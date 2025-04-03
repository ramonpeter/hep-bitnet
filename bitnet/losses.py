""" Definition of Bayesian losses"""

import torch


def bMSE(prediction: torch.Tensor, targets: torch.Tensor):
    mu = prediction[:, 0]
    logsigma2 = prediction[:, 1]
    out = torch.pow(mu - targets, 2) / (2 * logsigma2.exp()) + 1.0 / 2.0 * logsigma2
    return torch.mean(out)
