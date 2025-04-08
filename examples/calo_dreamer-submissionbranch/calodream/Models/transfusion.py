import torch
from .base import GenerativeModel
from calodream import Networks

class TransfusionAR(GenerativeModel):

    def __init__(self, params, device, model_dir):

        super().__init__(params, device, model_dir)

        self.t_min = self.params.get('t_min', 0)
        self.t_max = self.params.get('t_max', 1)
        distribution = self.params.get('distribution', 'uniform')
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
        network = self.params.get('network', "ARtransformer")
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

        # add phantom layer dim to condition
        c = c.unsqueeze(-1)

        # Sample time steps
        t = self.distribution.sample(
            list(x.shape[:2]) + [1]*(x.ndim-2)).to(dtype=x.dtype, device=x.device)

        # Sample noise variables
        x_0 = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        # Calculate point and derivative on trajectory
        x_t, x_t_dot = linear_trajectory(x_0, x, t)
        v_pred = self.net(c,x_t,t,x)
        # Mask out masses if not needed
        loss = ((v_pred - x_t_dot) ** 2).mean()

        return loss

    @torch.inference_mode()
    def sample_batch(self,c):
        sample = self.net(c.unsqueeze(-1), rev=True)
        return sample

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot