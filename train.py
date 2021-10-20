import os
import json
import logging
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from tqdm.auto import tqdm, trange
from matplotlib import pyplot as plt

from lib.nets import Generator, Discriminator, Denormalize, convert_model
from lib.fid import FID
from lib.utils import *
from lib.plots import plot_images


def train_epoch(net_g, net_d, opt_g, opt_d, dataloader, criterion,
                z_dim, device, log, double_step=False):
    losses_g = []
    losses_d = []

    total = len(dataloader)
    total_char_len = len(str(total))
    batch_format = lambda batch: f"[{batch:{total_char_len}d}/{total}] "
    log_interval = total // 10

    net_g.train()
    net_d.train()

    for batch, images in enumerate(tqdm(dataloader, desc="Batch", ncols=70, leave=False)):
        batch_size = images.size(0)

        real_label = torch.ones(batch_size, device=device)
        fake_label = torch.zeros(batch_size, device=device)

        real_images = images.to(device)
        noise = torch.randn(batch_size, z_dim, device=device)
        fake_images = net_g(noise)

        # Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        real_output = net_d(real_images)
        fake_output = net_d(fake_images.detach())

        loss_d_real = criterion(real_output, real_label)
        loss_d_fake = criterion(fake_output, fake_label)
        loss_d = loss_d_real + loss_d_fake
        losses_d.append((loss_d.item(), batch_size))

        net_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # Update Generator network: maximize log(D(G(z)))
        fake_output1 = net_d(fake_images)
        loss_g1 = criterion(fake_output1, real_label)

        net_g.zero_grad()
        loss_g1.backward()
        opt_g.step()

        if double_step:
            fake_images2 = net_g(noise)
            fake_output2 = net_d(fake_images2)
            loss_g2 = criterion(fake_output2, real_label)

            net_g.zero_grad()
            loss_g2.backward()
            opt_g.step()

            # Statistics
            loss_g = (loss_g1 + loss_g2) / 2
            d_x    = real_output.mean().item()
            d_g_z1 = fake_output1.mean().item()
            d_g_z2 = fake_output2.mean().item()
        else:
            # Statistics
            loss_g = loss_g1
            d_x    = real_output.mean().item()
            d_g_z1 = fake_output.mean().item()
            d_g_z2 = fake_output1.mean().item()

        losses_g.append((loss_g, batch_size))

        # Logging
        if batch % log_interval == 0:
            log(batch_format(batch) +
                f"Loss D: {loss_d.item():.4f}  Loss G: {loss_g.item():.4f}  "
                f"D(x): {d_x:.4f}  D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}")

    return losses_g, losses_d


def main(args, logger):
    base_path = f"./out/{args.train_name}"

    # Set device
    if torch.cuda.is_available() and args.num_gpu > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    # Load dataset
    num_channels = 3
    dataloader = get_celeba_dataloader(
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, train=True, shuffle=True,
    )
    test_dataloader = get_celeba_dataloader(
        image_size=args.image_size, batch_size=args.num_test,
        num_workers=0, train=False, shuffle=True,
    )
    test_iter = iter(test_dataloader)
    logger.debug("Dataset loaded")

    # Save sample images
    if not args.resume:
        images = next(iter(dataloader))[:64]
        fig = plot_images(images)
        fig.savefig(f"{base_path}/sample_images.png")
        plt.close(fig)
        logger.debug(f"Save sample images to '{base_path}/sample_images.png'")

    # Model
    denormalize = Denormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).to(device)

    net_g = Generator(args.z_dim, args.num_features_generator, num_channels)
    net_g = convert_model(net_g, device, args.num_gpu)
    logger.debug(format_block(net_g))

    net_d = Discriminator(args.num_features_discriminator, num_channels)
    net_d = convert_model(net_d, device, args.num_gpu)
    logger.debug(format_block(net_d))

    fid_model = FID(dims=args.inception_dim).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()

    opt_g = optim.Adam(net_g.parameters(), lr=args.lr_generator,     betas=(args.beta1, 0.999))
    opt_d = optim.Adam(net_d.parameters(), lr=args.lr_discriminator, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, args.z_dim, device=device)

    if args.resume:
        # find the folder with the highest epoch number
        folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        max_epoch = sorted(folders)[-1]
        logger.info(f"Resume training from epoch {max_epoch}")
        # load the checkpoint
        ckpt_net_g = torch.load(f"{base_path}/{max_epoch}/net_g.pth")
        ckpt_net_d = torch.load(f"{base_path}/{max_epoch}/net_d.pth")
        ckpt_opt_g = torch.load(f"{base_path}/{max_epoch}/opt_g.pth")
        ckpt_opt_d = torch.load(f"{base_path}/{max_epoch}/opt_d.pth")
        # load the model
        net_g.load_state_dict(ckpt_net_g)
        net_d.load_state_dict(ckpt_net_d)
        # load the optimizer
        opt_g.load_state_dict(ckpt_opt_g)
        opt_d.load_state_dict(ckpt_opt_d)
        # start from the next epoch
        start_epoch = int(max_epoch) + 1
    else:
        start_epoch = 0

    # Train
    progress_losses_g = []
    progress_losses_d = []

    epoch_char_len = len(str(args.num_epochs))
    epoch_format = lambda epoch: f"[{epoch:{epoch_char_len}d}/{args.num_epochs}] "

    for epoch in trange(start_epoch, args.num_epochs, desc="Epoch", ncols=70):
        log = lambda msg: logger.info(epoch_format(epoch) + msg)
        losses = train_epoch(net_g, net_d, opt_g, opt_d, dataloader, criterion,
                             args.z_dim, device, log, args.double_step)

        losses_g, losses_d = losses

        total_loss_g = sum([x[0] * x[1] for x in losses_g]) / sum([x[1] for x in losses_g])
        total_loss_d = sum([x[0] * x[1] for x in losses_d]) / sum([x[1] for x in losses_d])
        log(f"Loss D: {total_loss_d:.4f}  Loss G: {total_loss_g:.4f}")

        progress_losses_g.extend(losses_g)
        progress_losses_d.extend(losses_d)

        epoch_path = f"{base_path}/{epoch:03d}"
        os.makedirs(epoch_path, exist_ok=True)
        logger.debug(f"Folder created: {epoch_path}/")

        ## Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(net_g.state_dict(), f"{epoch_path}/net_g.pth")
            logger.debug(f"Save generator network to       '{epoch_path}/net_g.pth'")
            torch.save(net_d.state_dict(), f"{epoch_path}/net_d.pth")
            logger.debug(f"Save discriminator network to   '{epoch_path}/net_d.pth'")
            torch.save(opt_g.state_dict(), f"{epoch_path}/opt_g.pth")
            logger.debug(f"Save generator optimizer to     '{epoch_path}/opt_g.pth'")
            torch.save(opt_d.state_dict(), f"{epoch_path}/opt_d.pth")
            logger.debug(f"Save discriminator optimizer to '{epoch_path}/opt_d.pth'")

        ## Save output
        with torch.no_grad():
            net_g.eval()

            fake_images = net_g(fixed_noise)
            fig = plot_images(fake_images)
            fig.savefig(f"{epoch_path}/generated.png")
            plt.close(fig)
            logger.debug(f"Save progress images to '{epoch_path}/generated.png'")

            if (epoch + 1) % 2 == 0:
                for t in range(1):
                    try:
                        real_images = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_dataloader)
                        real_images = next(test_iter)

                    noise = torch.randn(args.num_test, args.z_dim, device=device)
                    fake_images = denormalize(net_g(noise))

                    os.makedirs(f"{epoch_path}/images/", exist_ok=True)
                    logger.debug(f"Folder created: {epoch_path}/images")

                    torch.save(real_images, f"{epoch_path}/images/real-{t}.pth")
                    logger.debug(f"Save real images to '{epoch_path}/images/real-{t}.pth'")
                    torch.save(fake_images, f"{epoch_path}/images/fake-{t}.pth")
                    logger.debug(f"Save fake images to '{epoch_path}/images/fake-{t}.pth'")

                    fid_score = fid_model(real_images.to(device), fake_images.to(device))
                    log(f"FID: {fid_score:.4f}")

                    with open(f"{epoch_path}/images/fid-{t:03d}.txt", "w") as f:
                        f.write(f"{fid_score:.6f}\n")


if __name__ == "__main__":
    import multiprocessing

    NGPU = torch.cuda.device_count()
    NCPU = multiprocessing.cpu_count()
    NW = NCPU // 2

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment
    parser.add_argument("-n",   "--train-name",                 type=str,   default="",   help="Name of training run")
    parser.add_argument("-g",   "--num-gpu",                    type=int,   default=NGPU, help="Number of GPUs (0 = CPU)")
    parser.add_argument("-s",   "--seed",                       type=int,   default=109,  help="Random seed")
    # DataLoader
    parser.add_argument("-w",   "--num-workers",                type=int,   default=NW,   help="Number of workers for dataloader")
    parser.add_argument("-bs",  "--batch-size",                 type=int,   default=128,  help="Batch size during training")
    parser.add_argument("-is",  "--image-size",                 type=int,   default=64,   help="Spatial size of training images")
    # Model
    parser.add_argument("-fg",  "--num-features-generator",     type=int,   default=64,   help="Size of feature maps in generator")
    parser.add_argument("-fd",  "--num-features-discriminator", type=int,   default=64,   help="Size of feature maps in discriminator")
    parser.add_argument("-z",   "--z-dim",                      type=int,   default=128,  help="Dimension of z latent vector")
    parser.add_argument("-id",  "--inception-dim",              type=int,   default=2048, help="Dimension of inception module")
    # Training
    parser.add_argument("-e",   "--num-epochs",                 type=int,   default=100,  help="Number of training epochs")
    parser.add_argument("-b1",  "--beta1",                      type=float, default=0.5,  help="Beta1 for Adam optimizers")
    parser.add_argument("-lrg", "--lr-generator",               type=float, default=2e-4, help="Learning rate for optimizers")
    parser.add_argument("-lrd", "--lr-discriminator",           type=float, default=2e-4, help="Learning rate for optimizers")
    parser.add_argument("-ds",  "--double-step",                action="store_true",      help="Double learning step for generator")
    parser.add_argument("-t",   "--num-test",                   type=int,   default=1000, help="Number of test images")
    # Misc
    parser.add_argument("-det", "--deterministic",              action="store_true",      help="Deterministic training")
    parser.add_argument("-d",   "--debug",                      action="store_true",      help="Turn on debug messages")
    parser.add_argument("-r",   "--resume",                     action="store_true",      help="Resume training from checkpoint")
    # Parse
    args = parser.parse_args()

    train_exists = os.path.exists(f"./out/{args.train_name}")

    if args.train_name == "":
        args.train_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    elif train_exists and not args.resume:
        print(f"Train name '{args.train_name}' already exist.\n"
               "Use --resume to resume training or use another name.")
        exit(1)
    elif not train_exists and args.resume:
        print(f"Train name '{args.train_name}' does not exist.\n"
               "Use a valid train name to resume training.")
        exit(1)
    elif train_exists and args.resume:
        print(f"Resuming training from '{args.train_name}'.\n"
               "Warning: Reproducibility is not guaranteed.")
        # load_args_from_checkpoint(f"./out/{args.train_name}/train.log")  # TODO

    os.makedirs(f"./out/{args.train_name}", exist_ok=True)
    with open(f"./out/{args.train_name}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Set random seed
    set_seed(args.seed, deterministic=args.deterministic)

    # Set logger
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger_fmt = "[%(asctime)s] [%(levelname)s] %(message)s"

    console_handler = TqdmHandler()
    file_handler = logging.FileHandler(f"./out/{args.train_name}/train.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler.setFormatter(logging.Formatter(fmt=logger_fmt, datefmt="%H:%M:%S"))
    file_handler.setFormatter(logging.Formatter(fmt=logger_fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    args_log_str = (
        f"Arguments:\n"
        f"  * Environment\n"
        f"    - train-name: {args.train_name}\n"
        f"    - num-gpu:    {args.num_gpu}\n"
        f"    - seed:       {args.seed}\n"
        f"  * DataLoader\n"
        f"    - num-workers: {args.num_workers}\n"
        f"    - batch-size:  {args.batch_size}\n"
        f"    - image-size:  {args.image_size}\n"
        f"  * Model\n"
        f"    - num-features-generator:     {args.num_features_generator}\n"
        f"    - num-features-discriminator: {args.num_features_discriminator}\n"
        f"    - z-dim:                      {args.z_dim}\n"
        f"    - inception-dim:              {args.inception_dim}\n"
        f"  * Training\n"
        f"    - num-epochs:       {args.num_epochs}\n"
        f"    - beta1:            {args.beta1}\n"
        f"    - lr-generator:     {args.lr_generator}\n"
        f"    - lr-discriminator: {args.lr_discriminator}\n"
        f"    - double-step:      {args.double_step}\n"
        f"    - num-test:         {args.num_test}"
    )

    logger.info(format_block(args_log_str))

    try:
        main(args, logger)
    except KeyboardInterrupt:
        logger.info("Stopped")
        exit()
