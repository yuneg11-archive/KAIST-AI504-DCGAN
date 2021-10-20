import logging
from textwrap import indent

from tqdm.auto import tqdm

import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


__all__ = [
    "TqdmHandler",
    "format_block",
    "get_celeba_dataloader",
    "save_image_list",
    "set_seed",
]


class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def format_block(obj, indent_level: int = 4):
    return f"\n{indent(str(obj), ' ' * indent_level)}\n"


def get_celeba_dataloader(
    root: str = "./data", train: bool = True, shuffle: bool = True,
    image_size: int = 64, batch_size: int = 128, num_workers: int = 0,
):
    """
    You should download the following files from
    [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8).

    1. Download the following files:
        - `Anno/list_attr_celeba.txt`
        - `Anno/identity_CelebA.txt`
        - `Anno/list_bbox_celeba.txt`
        - `Anno/list_landmarks_align_celeba.txt`
        - `Eval/list_eval_partition.txt`
        - `Img/img_align_celeba.zip`
    2. Make folder `celeba` under `./data/` directory.
    3. Move downloaded files to `./data/celeba/` directory.
    Then extract the `img_align_celeba.zip`.
    4. Final directory structure under `./data/celeba/` should be like:
        - `img_align_celeba` (directory)
        - `img_align_celeba.zip` (can be deleted)
        - `identity_CelebA.txt`
        - `list_attr_celeba.txt`
        - `list_bbox_celeba.txt`
        - `list_eval_partition.txt`
        - `list_landmarks_align_celeba.txt`
    """

    transform = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    if train:
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    dataset = datasets.CelebA(root=root, split="all", transform=transforms.Compose(transform))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, collate_fn=(lambda batch: torch.stack([x for x, _ in batch])),
    )

    return dataloader


def save_image_list(base_path, dataset):
    dataset_path = []

    for i in range(len(dataset)):
        save_path =  f"{base_path}/{i:4d}.png"
        dataset_path.append(save_path)
        save_image(dataset[i], save_path)

    return base_path


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
