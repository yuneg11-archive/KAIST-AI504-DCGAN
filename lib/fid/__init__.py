import numpy as np
from scipy import linalg

import torch
from torch import nn
from torch.nn import functional as F

from .inception import InceptionV3


def _cov(tensor):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor.to(dtype=torch.float64)
    tensor = (tensor - tensor.mean(dim=0, keepdim=True))
    factor = 1 / (tensor.shape[0] - 1)
    return factor * torch.matmul(tensor.T, tensor).conj()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the inception net
               (like returned by the function 'get_predictions') for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    device, dtype = mu1.device, mu1.dtype

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(torch.matmul(sigma1, sigma2).cpu().numpy(), disp=False)
    if not np.isfinite(covmean).all():
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm(torch.matmul(sigma1 + offset, sigma2 + offset).cpu().numpy())

    covmean = torch.from_numpy(covmean).to(device=device)

    # Numerical error might give slight imaginary component
    if covmean.is_complex():
        imag_diag = torch.diagonal(covmean).imag
        if not torch.allclose(imag_diag, torch.zeros_like(imag_diag), atol=1e-3):
            m = torch.max(torch.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    diff = mu1 - mu2
    fd = (torch.matmul(diff, diff) + torch.trace(sigma1) + torch.trace(sigma2)
          - 2 * torch.trace(covmean.to(dtype=dtype)))

    return fd


class FID(nn.Module):
    def __init__(self, batch_size: int = 128, dims: int = 2048):
        super().__init__()
        self.batch_size = batch_size
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx])
        self.inception.eval()

    def _compute_statistics(self, x):
        act_batches = []

        for i in range(0, len(x), self.batch_size):
            start, end = i, i + self.batch_size
            act_batch = self.inception(x[start:end])[0]
            if act_batch.size(2) != 1 or act_batch.size(3) != 1:
                act_batch = F.adaptive_avg_pool2d(act_batch, output_size=(1, 1))
            act_batch = act_batch.flatten(start_dim=1)
            act_batches.append(act_batch)

        act = torch.vstack(act_batches)  # N x D
        mu = torch.mean(act, dim=0)      # D
        sigma = _cov(act)                # D x D

        return mu, sigma

    def forward(self, x1, x2):
        mu1, sigma1 = self._compute_statistics(x1)
        mu2, sigma2 = self._compute_statistics(x2)
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid
