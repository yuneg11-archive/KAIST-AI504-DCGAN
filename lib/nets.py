from typing import Callable, Optional

import torch
from torch import nn

torch.optim.Adam

__all__ = [
    "convert_model",
    "Generator",
    "Discriminator",
]


def _weights_init(m):
    class_name = m.__class__.__name__
    if "Conv" in class_name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in class_name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def convert_model(model, device, num_gpu):
    if device.type == "cuda" and num_gpu > 1:
        model = nn.DataParallel(model, list(range(num_gpu)))
    return model.to(device)


class Denormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.as_tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class Generator(nn.Sequential):
    def __init__(self,
        z_dim: int,
        num_features: int,
        num_channels: int = 3,
        target_transform: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.num_features = num_features
        self.num_channels = num_channels

        # Input: Z
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(z_dim, 1, 1))
        self.deconv1 = self._deconv(
            in_channels=z_dim,
            out_channels=(num_features * 8),
            stride=1, padding=0,
        )
        # State size: (ngf x 8) x 4 x 4
        self.deconv2 = self._deconv(
            in_channels=(num_features * 8),
            out_channels=(num_features * 4),
        )
        # State size: (ngf x 4) x 8 x 8
        self.deconv3 = self._deconv(
            in_channels=(num_features * 4),
            out_channels=(num_features * 2),
        )
        # State size: (ngf x 2) x 16 x 16
        self.deconv4 = self._deconv(
            in_channels=(num_features * 2),
            out_channels=num_features,
        )
        # State size: (ngf) x 32 x 32
        self.deconv5 = self._deconv(
            in_channels=num_features,
            out_channels=num_channels,
            batch_norm=False,
            act_factory=lambda: nn.Tanh(),
        )
        # Output size: (nc) x 64 x 64
        if target_transform is not None:
            self.target_transform = target_transform

        self.apply(_weights_init)

    @staticmethod
    def _deconv(
        in_channels: int, out_channels: int, kernel_size: int = 4,
        stride: int = 2, padding: int = 1, batch_norm: bool = True,
        act_factory: Callable = lambda: nn.ReLU(inplace=True),
    ):
        return nn.Sequential(
            *([nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, bias=False)]
            + ([nn.BatchNorm2d(out_channels)] if batch_norm else [])
            + [act_factory()])
        )


class Discriminator(nn.Sequential):
    def __init__(self,
        num_features: int,
        num_channels: int = 3,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_channels = num_channels

        # Input size: (nc) x 64 x 64
        self.conv1 = self._conv(
            in_channels=num_channels,
            out_channels=num_features,
        )
        # State size: (ngf) x 32 x 32
        self.conv2 = self._conv(
            in_channels=num_features,
            out_channels=(num_features * 2),
        )
        # State size: (ngf x 2) x 16 x 16
        self.conv3 = self._conv(
            in_channels=(num_features * 2),
            out_channels=(num_features * 4),
        )
        # State size: (ngf x 4) x 8 x 8
        self.conv4 = self._conv(
            in_channels=(num_features * 4),
            out_channels=(num_features * 8),
        )
        # State size: (ngf x 8) x 4 x 4
        self.conv5 = self._conv(
            in_channels=(num_features * 8),
            out_channels=1,
            stride=1, padding=0,
            batch_norm=False,
            act_factory=nn.Sigmoid,
        )
        self.flatten = nn.Flatten(start_dim=0)
        # Output: (1)

        self.apply(_weights_init)

    @staticmethod
    def _conv(
        in_channels: int, out_channels: int, kernel_size: int = 4,
        stride: int = 2, padding: int = 1, batch_norm: bool = True,
        act_factory: Callable = lambda: nn.LeakyReLU(0.2, inplace=True),
    ):
        return nn.Sequential(
            *([nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, bias=False)]
            + ([nn.BatchNorm2d(out_channels)] if batch_norm else [])
            + [act_factory()])
        )


if __name__ == "__main__":
    z_dim = 256
    num_features_g = 128
    num_features_d = 128
    num_channels = 3

    target_transform = Denormalize(torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5]))

    generator = Generator(z_dim, num_features_g, num_channels, target_transform)
    print(generator, end="\n\n")

    discriminator = Discriminator(num_features_d, num_channels)
    print(discriminator, end="\n\n")

    goutput = generator(torch.empty(100, z_dim))
    print(f"Generator ouput shape:     {list(goutput.shape)}")

    doutput = discriminator(torch.empty(100, num_channels, 64, 64))
    print(f"Discriminator ouput shape: {list(doutput.shape)}")
