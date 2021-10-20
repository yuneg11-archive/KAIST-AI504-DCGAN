import numpy as np
from imageio import mimsave
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from torchvision.utils import make_grid
from IPython.display import HTML


def plot_images(images, figsize=(20, 20)):
    img_grid = make_grid(images.detach(), padding=2, normalize=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    fig.tight_layout()
    return fig


def plot_images_for_gif(images):
    img_grid = make_grid(images.detach(), padding=2) * 255.
    img_grid = img_grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return img_grid


def save_gif(images_list, gif_name):
    mimsave(gif_name, images_list)


def plot_gif(images_list, plot_length=10):
    plt.close()
    fig = plt.figure()

    total_len = len(images_list)
    for i in range(plot_length):
        im = plt.imshow(images_list[int(total_len/plot_length)*i])
        plt.show()


def show_gif(images_list):
    def make_frame(images):
        fig, ax = plt.subplots()
        ax.imshow(images, animated=True)
        return [fig]

    frames = [make_frame(images) for images in images_list]

    fig, _ = plt.subplots()
    anim = ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    html = anim.to_html5_video()
    return HTML(html)



# def show_gif(training_progress_images):
#     fig = plt.figure()

#     ims = []
#     for i in range(len(training_progress_images)):
#         im = plt.imshow(training_progress_images[i], animated=True)
#         ims.append([im])

#     anim = ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
#     html = anim.to_html5_video()
#     return HTML(html)
