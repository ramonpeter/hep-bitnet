import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Networks
import Models
from einops import rearrange

import math
from typing import Type, Callable, Union, Optional
import torch
import torch.nn as nn
from torchdiffeq import odeint

class TransfusionAR(GenerativeModel):

    def __init__(self, params: dict, device, doc):
        super().__init__(params, device, doc)
        self.params = params
        trajectory = get(self.params, "trajectory", "linear_trajectory")
        try:
            self.trajectory = getattr(Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        self.dim_embedding = params["dim_embedding"]

        self.t_min = get(self.params, "t_min", 0)
        self.t_max = get(self.params, "t_max", 1)
        distribution = get(self.params, "distribution", "uniform")
        if distribution == "uniform":
            self.distribution = torch.distributions.uniform.Uniform(low=self.t_min, high=self.t_max)
        elif distribution == "beta":
            self.distribution = torch.distributions.beta.Beta(1.5, 1.5)
        else:
            raise NotImplementedError(f"build_model: Distribution type {distribution} not implemented")

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "ARtransformer")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        # x = input[0].clone()
        condition = input[1]
        weights = None
        # return x, condition, weights
        return input[0], condition, weights

    def batch_loss(self, input):
        """
        Args:
            x: input tensor, shape (n_events, dims_in)
            c: condition tensor, shape (n_events, dims_c)
            kl_scale: factor in front of KL loss term, default 0
        Returns:
            loss: batch loss
            loss_terms: dictionary with loss contributions
        """
        x, c, _ = self.get_condition_and_input(input)

        if self.latent: # encode x into autoencoder latent space
            x = self.ae.encode(x, c)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])
            x = self.ae.unflatten_layer_from_batch(x)
        # else:
        #     print(x.shape)
        #     x = x.movedim(1,2)
        #     print(x.shape)
            
        # add phantom layer dim to condition
        c = c.unsqueeze(-1)

        # Sample time steps
        t = self.distribution.sample(
            list(x.shape[:2]) + [1]*(x.ndim-2)).to(dtype=x.dtype, device=x.device)

        # Sample noise variables
        x_0 = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        # Calculate point and derivative on trajectory
        x_t, x_t_dot = self.trajectory(x_0, x, t)
        v_pred = self.net(c,x_t,t,x)
        # Mask out masses if not needed
        loss = ((v_pred - x_t_dot) ** 2).mean()

        return loss

    @torch.inference_mode()
    def sample_batch(self,c):
        sample = self.net(c.unsqueeze(-1), rev=True)
        if self.latent: # decode the generated sample
            sample, c = self.ae.flatten_layer_to_batch(sample, c)
            sample = self.ae.decode(sample.squeeze(), c)
        return sample

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot