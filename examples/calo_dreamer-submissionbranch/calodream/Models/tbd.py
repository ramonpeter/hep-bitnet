import torch
from torchdiffeq import odeint

from .base import GenerativeModel
from calodream import Networks

class TBD(GenerativeModel):
    """
     Class for trajectory-based diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params, device, model_dir):
        
        super().__init__(params, device, model_dir)

        self.t_min = self.params.get('t_min', 0)
        self.t_max = self.params.get('t_max', 1)
        self.distribution = torch.distributions.uniform.Uniform(low=self.t_min, high=self.t_max)
        self.add_noise = self.params.get('add_noise', False)
        self.gamma = self.params.get('gamma', 1.e-4)


    def build_net(self):
        """
        Build the network
        """
        network = self.params.get("network", "ViT")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")

    def get_condition_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        condition = input[1]
        weights = None
        return input[0], condition, weights

    def batch_loss(self, x):
        """
        Calculate batch loss as described by Peter
        """
        
        # get input and conditions
        x, condition, weights = self.get_condition_and_input(x)

        t = self.distribution.sample([x.shape[0]] + [1]*(x.dim() - 1)).to(x.device)
        
        x_0 = torch.randn_like(x)
        if self.add_noise:
            x = x + self.gamma * torch.randn_like(x, device=x.device, dtype=x.dtype)
        x_t, x_t_dot = linear_trajectory(x_0, x, t)
        self.net.kl = 0
        drift = self.net(x_t, t.view(-1, 1), condition)

        loss = torch.mean((drift - x_t_dot) ** 2)

        return loss

    @torch.inference_mode()
    def sample_batch(self, batch):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        dtype = batch.dtype
        device = batch.device

        x_T = torch.randn((batch.shape[0], *self.shape), dtype=dtype, device=device)

        def f(t, x_t):
            t_torch = t.repeat((x_t.shape[0],1)).to(self.device)
            return self.net(x_t, t_torch, batch)

        sample = odeint(
            f, x_T,
            torch.tensor([self.t_min, self.t_max], dtype=dtype, device=device),
            **self.params.get("solver_kwargs", {})
        )[-1]
            
        return sample

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot