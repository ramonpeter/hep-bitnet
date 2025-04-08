import numpy as np
import torch
from scipy.integrate import solve_ivp
import Networks
from Util.util import get
from Models.ModelBase import GenerativeModel
import Networks
import Models
from torchdiffeq import odeint
from torchsde import sdeint


class TBD(GenerativeModel):
    """
     Class for Trajectory Based Diffusion
     Inheriting from the GenerativeModel BaseClass
    """

    def __init__(self, params, device, doc):
        super().__init__(params, device, doc)
        trajectory = get(self.params, "trajectory", "linear_trajectory")
        try:
            self.trajectory =  getattr(Models.tbd, trajectory)
        except AttributeError:
            raise NotImplementedError(f"build_model: Trajectory type {trajectory} not implemented")

        self.C = get(self.params, "C", 1)
        if self.C != 1:
            print(f"C is {self.C}")

        self.bayesian = get(self.params, "bayesian", 0)
        self.t_min = get(self.params, "t_min", 0)
        self.t_max = get(self.params, "t_max", 1)
        self.distribution = torch.distributions.uniform.Uniform(low=self.t_min, high=self.t_max)
        self.add_noise = get(self.params, "add_noise", False)
        self.gamma = get(self.params, "gamma", 1.e-4)


    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "network", "Resnet")
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
        
        if self.latent:
            # encode x into autoencoder latent space
            x = self.ae.encode(x, condition)
            if self.ae.kl:
                x = self.ae.reparameterize(x[0], x[1])

        # t = self.distribution.sample((x.size(0),1)).to(x.device)
        t = self.distribution.sample([x.shape[0]] + [1]*(x.dim() - 1)).to(x.device)
        x_0 = torch.randn_like(x)
        if self.add_noise:
            x = x + self.gamma * torch.randn_like(x, device=x.device, dtype=x.dtype)
        x_t, x_t_dot = self.trajectory(x_0, x, t)
        self.net.kl = 0
        drift = self.net(x_t, t.view(-1, 1), condition)

        loss = torch.mean((drift - x_t_dot) ** 2)#* torch.exp(self.t_factor * t)) ?
        # self.regular_loss.append(loss.detach().cpu().numpy())
        # if self.C != 0:
            # kl_loss = self.C*self.net.kl / self.n_traindata
            # self.kl_loss.append(kl_loss.detach().cpu().numpy())
            # loss = loss + kl_loss

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

        solver = sdeint if self.params.get("use_sde", False) else odeint
        function = SDE(self.net) if self.params.get("use_sde", False) else f

        sample = solver(
            function, x_T,
            torch.tensor([self.t_min, self.t_max], dtype=dtype, device=device),
            **self.params.get("solver_kwargs", {})
        )[-1]

        if self.latent:
            # decode the generated sample
            sample = self.ae.decode(sample, batch)
            
        return sample

    def invert_n(self, samples):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        if self.net.bayesian:
            self.net.map = get(self.params,"fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        self.eval()
        batch_size = get(self.params, "batch_size", 8192)
        n_samples = samples.shape[0]

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((-1, *self.shape)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.inference_mode():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.inference_mode():
            for i in range(int(n_samples / batch_size)):
                sol = solve_ivp(f, (1, 0), samples[batch_size * i: batch_size * (i + 1)].flatten())
                s = sol.y[:, -1].reshape(batch_size, *self.shape)
                events.append(s)
            sol = solve_ivp(f, (1, 0), samples[batch_size * (i+1):].flatten())
            s = sol.y[:, -1].reshape(-1, *self.shape)
            events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]

    def sample_n_evolution(self, n_samples):

        n_frames = get(self.params, "n_frames", 1000)
        t_frames = np.linspace(0, 1, n_frames)

        batch_size = get(self.params, "batch_size", 8192)
        x_T = np.random.randn(n_samples + batch_size, *self.shape)

        def f(t, x_t):
            x_t_torch = torch.Tensor(x_t).reshape((batch_size, *self.shape)).to(self.device)
            t_torch = t * torch.ones_like(x_t_torch[:, [0]])
            with torch.inference_mode():
                f_t = self.net(x_t_torch, t_torch).detach().cpu().numpy().flatten()
            return f_t

        events = []
        with torch.inference_mode():
            for i in range(int(n_samples / batch_size) + 1):
                sol = solve_ivp(f, (0, 1), x_T[batch_size * i: batch_size * (i + 1)].flatten(), t_eval=t_frames)
                s = sol.y.reshape(batch_size, *self.shape, -1)
                events.append(s)
        return np.concatenate(events, axis=0)[:n_samples]



def sine_cosine_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c * x_0 + s * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = c_dot * x_0 + s_dot * x_1
    return x_t, x_t_dot

def sine2_cosine2_trajectory(x_0, x_1, t):
    c = torch.cos(t * np.pi / 2)
    s = torch.sin(t * np.pi / 2)
    x_t = c**2 * x_0 + s**2 * x_1

    c_dot = -np.pi / 2 * s
    s_dot = np.pi / 2 * c
    x_t_dot = 2 * c_dot * c * x_0 + 2 * s_dot * s * x_1
    return x_t, x_t_dot

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot

def vp_trajectory(x_0, x_1, t, a=19.9, b=0.1):

    e = -1./4. * a * (1-t)**2 - 1./2. * b * (1-t)
    alpha_t = torch.exp(e)
    beta_t = torch.sqrt(1-alpha_t**2)
    x_t = x_0 * alpha_t + x_1 * beta_t

    e_dot = 2 * a * (1-t) + 1./2. * b
    alpha_t_dot = e_dot * alpha_t
    beta_t_dot = -2 * alpha_t * alpha_t_dot / beta_t
    x_t_dot = x_0 * alpha_t_dot + x_1 * beta_t_dot
    return x_t, x_t_dot

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, net):
        super().__init__()
        self.net = net

    def f(self,t, x_t):
        t_torch = t * torch.ones_like(x_t[:, [0]])
        v = self.net(x_t, t_torch)

        return v
    def g(self,t,x_t):
        epsilon = 0.5 * torch.ones_like(x_t)
        return np.sqrt(2*epsilon)*x_t.shape[1]
