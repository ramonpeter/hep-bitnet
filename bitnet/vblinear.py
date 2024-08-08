import math
import sys
import torch
import torch.nn as nn
import numpy as np


class VBLinear(nn.Module):
    """
    Bayesian linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_prec: float = 1.0,
        enable_map: bool = False,
        std_init: float = -9,
    ):
        """
        Constructs the Bayesian linear layer
        Args:
            in_features: Number of input dimensions
            out_features: Number of input dimensions
            prior_prec: Standard deviation of the Gaussian prior
            enable_map: If True, does not sample from posterior during evaluation
                        (maximum-a-posteriori)
            std_init: Logarithm of the initial standard deviation of the weights
        """
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = enable_map
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.std_init = std_init
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights of the layer.
        """
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        """
        Reset the random weights. New weights will be sampled the next time, forward is
        called in evaluation mode.
        """
        self.random = None

    def sample_random_state(self) -> np.ndarray:
        """
        Sample a random state and return it as a numpy array

        Returns:
            Tensor with sampled weights
        """
        return torch.randn_like(self.logsig2_w).detach().cpu().numpy()

    def import_random_state(self, state: np.ndarray):
        """
        Replace the random state

        Args:
            state: Numpy array with random numbers
        """
        self.random = torch.tensor(
            state, device=self.logsig2_w.device, dtype=self.logsig2_w.dtype
        )

    def kl(self) -> torch.Tensor:
        """
        KL divergence between posterior and prior.

        Returns:
            KL divergence
        """
        logsig2_w = self.logsig2_w.clamp(-11, 11)
        kl = (
            0.5
            * (
                self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                - logsig2_w
                - 1
                - np.log(self.prior_prec)
            ).sum()
        )
        return kl

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bayesian linear layer. In training mode, use the local
        reparameterization trick.
        Args:
            input: Input tensor
        Returns:
            Output tensor
        """
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = nn.functional.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = nn.functional.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.map:
                return nn.functional.linear(input, self.mu_w, self.bias)

            logsig2_w = self.logsig2_w.clamp(-11, 11)
            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return nn.functional.linear(input, weight, self.bias) + 1e-8

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.
        Returns:
            String representation of the layer
        """
        return f"{self.__class__.__name__} ({self.n_in}) -> ({self.n_out})"
