import os
import logging
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm.auto import tqdm, trange

from lib.utils import *
from lib.fid.fid_score import calculate_fid_given_paths
from lib import plots
from lib.fid import FID


# 42.87271405285509


if __name__ == "__main__":
    # print("fid start")
    # fid = calculate_fid_given_paths([
    #     "./out/20211021-071821/080/images/01/fake",
    #     "./out/20211021-071821/080/images/01/real",
    # ], batch_size=100, cuda=True, dims=2048)
    # print(fid)

    fid_model = FID(dims=2048).cuda()

    dataset = datasets.ImageFolder(
        root="./out/20211021-071821/080/images/01",
        transform=transforms.ToTensor(),
    )
    fakes = torch.stack([image for image, c in dataset if c == 0]).cuda()
    reals = torch.stack([image for image, c in dataset if c == 1]).cuda()

    print("fid start")

    fid_score = fid_model(fakes, reals)
    print(fid_score)
