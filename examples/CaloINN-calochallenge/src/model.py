import math

import torch
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm

from myBlocks import *
from vblinear import VBLinear

import numpy as np

class Subnet(nn.Module):
    """ This class constructs a subnet for the coupling blocks """

    def __init__(self, num_layers, size_in, size_out, internal_size=None, dropout=0.0,
                 layer_class=nn.Linear, layer_args={}):
        """
            Initializes subnet class.

            Parameters:
            size_in: input size of the subnet
            size: output size of the subnet
            internal_size: hidden size of the subnet. If None, set to 2*size
            dropout: dropout chance of the subnet
        """
        super().__init__()
        if internal_size is None:
            internal_size = size_out * 2
        if num_layers < 1:
            raise(ValueError("Subnet size has to be 1 or greater"))
        self.layer_list = []
        for n in range(num_layers - 1):
            if isinstance(internal_size, list):
                input_dim, output_dim = internal_size[n], internal_size[n+1]
            else:
                input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in

            self.layer_list.append(layer_class[n](input_dim, output_dim, **(layer_args[n])))

            if dropout > 0:
                self.layer_list.append(nn.Dropout(p=dropout))

            #self.layer_list.append(nn.BatchNorm1d(output_dim))
            self.layer_list.append(nn.LeakyReLU()) ###nn.SiLU()) ###########
        
        # separating last linear/VBL layer
        #output_dim = size_out
        self.layer_list.append(layer_class[-1](output_dim, size_out, **(layer_args[-1]) ))

        self.layers = nn.Sequential(*self.layer_list)

        final_layer_name = str(len(self.layers) - 1)
        for name, param in self.layers.named_parameters():
            if name[0] == final_layer_name and "logsig2_w" not in name:
                param.data.zero_()

    def forward(self, x):
        return self.layers(x)

class MixedTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, alpha = 0., alpha_logit=0.):
        super().__init__(dims_in, dims_c)
        self.alpha = alpha
        self.alpha_logit = alpha_logit

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        if rev:
            z1 = torch.exp(x[:,:369]) - self.alpha
            z2 = torch.sigmoid(x[:, -4:])
            z2 = (z2 - self.alpha_logit)/(1-2*self.alpha_logit)

            z = torch.cat((z1,z2), dim=1)
            jac = torch.sum( x, dim=1)
        else:
            z1 = torch.log(x[:,:369] + self.alpha)
            z2 = torch.logit(x[:, -4:]*(1-2*self.alpha_logit) + self.alpha_logit)

            z = torch.cat((z1,z2), dim=1)
            jac = - torch.sum( z, dim=1)
        return (z, ), torch.tensor([0.], device=x.device) # jac

    def output_dims(self, input_dims):
        return input_dims


class LogTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, alpha = 0., alpha_logit=0.):
        super().__init__(dims_in, dims_c)
        self.alpha = alpha
        self.alpha_logit = alpha_logit

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        if rev:
            z = torch.exp(x) - self.alpha
            #z2 = torch.exp(x[:, -4:])
            #z3 = x[:, 369].reshape(-1, 1)

            #z = torch.cat((z1,z3,z2), dim=1)
            jac = torch.sum( x, dim=1)
        else:
            z = torch.log(x + self.alpha)
            #z3 = x[:,369].reshape(-1, 1)
            #z2 = torch.log(x[:, -4:])     #*(1-2*self.alpha_logit) + self.alpha_logit)

            #z = torch.cat((z1,z3,z2), dim=1)
            jac = - torch.sum( z, dim=1)
        return (z, ), torch.tensor([0.], device=x.device) # jac

    def output_dims(self, input_dims):
        return input_dims


class LogitTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, alpha = 0.):
        super().__init__(dims_in, dims_c)
        self.alpha = alpha

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        if not rev:
            x = x*(1-2*self.alpha) + self.alpha
            z = torch.logit(x)
        else:
            if not self.training:
                x[:,:-3] = self.norm_logit(x[:,:-3])
            z = torch.sigmoid(x)
            z = (z - self.alpha)/(1-2*self.alpha)
        return (z, ), torch.tensor([0.], device=x.device) # jac

    def norm_logit(self, t: torch.Tensor):
         f = lambda x: torch.sum(1/(1+torch.exp(-t-x)), dim=1) - 1 - self.alpha*(t.shape[1] - 2)
         f_ = lambda x: torch.sum(torch.exp(-t-x)/(1+torch.exp(-t-x))**2,dim=1)
         c = torch.zeros((t.shape[0], 1), device=t.device)
         for i in range(8):
             c = c - (f(c)/f_(c))[...,None]
         return t+c

    def output_dims(self, input_dims):
        return input_dims


class NormTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, log_cond=False):
        super().__init__(dims_in, dims_c)
        self.log_cond = log_cond

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        c, = c
        if self.log_cond:
            c = torch.exp(c)
        if rev:
            z = x/c
            jac = -torch.log(c)
        else:
            z = x*c
            jac = torch.log(c)
        return (z, ), torch.tensor([0.], device=x.device) # jac

    def output_dims(self, input_dims):
        return input_dims


class CINN(nn.Module):
    """ cINN model """

    def __init__(self, params, data, cond):
        """ Initializes model class.

        Parameters:
        params: Dict containing the network and training parameter
        data: Training data to initialize the norm layer
        cond: Conditions to the training data
        """
        super(CINN, self).__init__()
        self.params = params
        self.num_dim = data.shape[1]

        self.norm_m = None
        self.bayesian = params.get("bayesian", False)
        self.alpha = params.get("alpha", 1e-8)
        self.alpha_logit = params.get("alpha_logit", 1.0e-6)
        self.log_cond = params.get("log_cond", False)
        self.use_norm = self.params.get("use_norm", False) and not self.params.get("use_extra_dim", False)
        self.pre_subnet = None

        if self.bayesian:
            self.bayesian_layers = []

        self.initialize_normalization(data, cond)
        self.define_model_architecture(self.num_dim)
        print(self.model)

    def forward(self, x, c, rev=False, jac=True):
        if self.log_cond:
            c_norm = torch.log2(c)
        else:
            c_norm = c
        if self.pre_subnet:
            c_norm = self.pre_subnet(c_norm)
        return self.model.forward(x, c_norm, rev=rev, jac=jac)

    def get_layer_class(self, lay_params):
        lays = []
        for n in range(len(lay_params)):
            if lay_params[n] == 'vblinear':
                lays.append(VBLinear)
            if lay_params[n] == 'linear':
                lays.append(nn.Linear)
        return lays

    def get_layer_args(self, params):
        layer_class = params["sub_layers"]
        layer_args = []
        for n in range(len(layer_class)):
            n_args = {}
            if layer_class[n] == "vblinear":
                if "prior_prec" in params:
                    n_args["prior_prec"] = params["prior_prec"]
                if "std_init" in params:
                    n_args["std_init"] = params["std_init"]
            layer_args.append(n_args)
        return layer_args

    def get_constructor_func(self, params):
        """ Returns a function that constructs a subnetwork with the given parameters """
        if "sub_layers" in params:
            layer_class = params["sub_layers"]
            layer_class = self.get_layer_class(layer_class)
            layer_args = self.get_layer_args(params)
        else:
            layer_class = []
            layer_args = []
            for n in range(params.get("layers_per_block", 3)):
                dicts = {}
                if self.bayesian:
                    layer_class.append(VBLinear)
                    if "prior_prec" in params:
                        dicts["prior_prec"] = params["prior_prec"]
                    if "std_init" in params:
                        dicts["std_init"] = params["std_init"]
                else:
                    layer_class.append(nn.Linear)
                layer_args.append(dicts)
        #if "prior_prec" in params:
        #    layer_args["prior_prec"] = params["prior_prec"]
        #if "std_init" in params:
        #    layer_args["std_init"] = params["std_init"]
        #if "bias" in params:
        #    layer_args["bias"] = params["bias"]
        def func(x_in, x_out):
            subnet = Subnet(
                    params.get("layers_per_block", 3),
                    x_in, x_out,
                    internal_size = params.get("internal_size"),
                    dropout = params.get("dropout", 0.),
                    layer_class = layer_class,
                    layer_args = layer_args)
            if self.bayesian:
                self.bayesian_layers.extend(
                    layer for layer in subnet.layer_list if isinstance(layer, VBLinear))
            return subnet
        return func

    def get_coupling_block(self, params):
        """ Returns the class and keyword arguments for different coupling block types """
        constructor_fct = self.get_constructor_func(params)
        permute_soft = params.get("permute_soft")
        coupling_type = params.get("coupling_type", "affine")

        if coupling_type == "affine":
            CouplingBlock = fm.AllInOneBlock
            block_kwargs = {
                            "affine_clamping": params.get("clamping", 5.),
                            "subnet_constructor": constructor_fct,
                            "global_affine_init": 0.92,
                            "permute_soft" : permute_soft
                           }
        elif coupling_type == "cubic":
            CouplingBlock = CubicSplineBlock
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                            "bounds_init": params.get("bounds_init", 10),
                            "bounds_type": params.get("bounds_type", "SOFTPLUS"),
                            "permute_soft" : permute_soft
                           }
        elif coupling_type == "rational_quadratic":
            CouplingBlock = RationalQuadraticSplineBlock
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                            "bounds_init":  params.get("bounds_init", 10),
                            "permute_soft" : permute_soft
                           }
        elif coupling_type == "MADE":
            CouplingBlock = MADE
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "bounds_init":  params.get("bounds_init", 10),
                            "permute_soft" : permute_soft,
                            "hidden_features": params.get("internal_size"),
                            "num_blocks": params.get("layers_per_block", 3)-2,
                            "dropout": params.get("dropout", 0.)
                           }
        else:
            raise ValueError(f"Unknown Coupling block type {coupling_type}")

        return CouplingBlock, block_kwargs

    def initialize_normalization(self, data, cond):
        """ Calculates the normalization transformation from the training data and stores it. """
        data = torch.clone(data)
        if self.use_norm:
            data /= cond
        data = torch.log(data + self.alpha)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        self.norm_m = torch.diag(1 / std)
        self.norm_b = - mean/std

        print('num samples out of bounds:', torch.count_nonzero(torch.max(torch.abs(data), dim=1)[0] > self.params.get("bounds_init", 10)).item())

    def define_model_architecture(self, in_dim):
        """ Create a ReversibleGraphNet model based on the settings, using
        SubnetConstructor as the subnet constructor """

        self.in_dim = in_dim
        if self.norm_m is None:
            self.norm_m = torch.eye(in_dim)
            self.norm_b = torch.zeros(in_dim)

        nodes = [ff.InputNode(in_dim, name="inp")]
        cond_node = ff.ConditionNode(1, name="cond")

        if self.use_norm:
                nodes.append(ff.Node(
                [nodes[-1].out0],
                NormTransformation,
                {"log_cond": self.log_cond},
                conditions = cond_node,
                name = "norm"
            ))
        nodes.append(ff.Node(
            [nodes[-1].out0],
            LogTransformation,
            { "alpha": self.alpha, "alpha_logit": self.alpha_logit },
            name = "inp_log"
        ))
        #nodes.append(ff.Node(
        #    [nodes[-1].out0],
        #    fm.FixedLinearTransform,
        #    { "M": self.norm_m, "b": self.norm_b },
        #    name = "inp_norm"
        #))
        CouplingBlock, block_kwargs = self.get_coupling_block(self.params)
        Affine = fm.AllInOneBlock
        Affine_kwargs = {
                            "affine_clamping": self.params.get("clamping", 5.),
                            "subnet_constructor": self.get_constructor_func(self.params),
                            "global_affine_init": 0.92,
                            "permute_soft" : False
                           }
 
        for i in range(self.params.get("n_blocks", 10)):
            if self.params.get("norm", True): # i!=0:
                nodes.append(
                    ff.Node(
                        [nodes[-1].out0],
                        fm.ActNorm,
                        module_args = {},
                        name = f"act_{i}"
                        )
                    )
            nodes.append(
                ff.Node(
                    [nodes[-1].out0],
                    CouplingBlock,
                    block_kwargs,
                    conditions = cond_node,
                    name = f"block_{i}"
                )
            )
            #nodes.append(
            #    ff.Node(
            #        [nodes[-1].out0],
            #        Affine,
            #        Affine_kwargs,
            #        conditions = cond_node,
            #        name = f"affine_{i}"
            #    )
            #)
         
        nodes.append(ff.OutputNode([nodes[-1].out0], name='out'))
        nodes.append(cond_node)

        self.model = ff.GraphINN(nodes)
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        n_trainable = sum(p.numel() for p in self.params_trainable)
        print(f"number of parameters: {n_trainable}", flush=True)

    def set_bayesian_std_grad(self, requires_grad):
        for layer in self.bayesian_layers:
            layer.logsig2_w.requires_grad = requires_grad

    def sample_random_state(self):
        return [layer.sample_random_state() for layer in self.bayesian_layers]

    def import_random_state(self, state):
        [layer.import_random_state(s) for layer, s in zip(self.bayesian_layers, state)]

    def get_kl(self):
        return sum(layer.KL() for layer in self.bayesian_layers)

    def enable_map(self):
        for layer in self.bayesian_layers:
            layer.enable_map()

    def disenable_map(self):
        for layer in self.bayesian_layers:
            layer.disenable_map()

    def reset_random(self):
        """ samples a new random state for the Bayesian layers """
        for layer in self.bayesian_layers:
            layer.reset_random()

    def sample(self, num_pts, condition):
        """
            sample from the learned distribution

            Parameters:
            num_pts (int): Number of samples to generate for each given condition
            condition (tensor): Conditions

            Returns:
            tensor[len(condition), num_pts, dims]: Samples 
        """
        z = torch.normal(0, 1,
            size=(num_pts*condition.shape[0], self.in_dim),
            device=next(self.parameters()).device)
        c = condition.repeat(num_pts,1)
        x, _ = self.forward(z, c, rev=True)
        return x.reshape(num_pts, condition.shape[0], self.in_dim).permute(1,0,2)

    def log_prob(self, x, c):
        """
            evaluate conditional log-likelihoods for given samples and conditions

            Parameters:
            x (tensor): Samples
            c (tensor): Conditions

            Returns:
            tensor: Log-likelihoods
        """
        z, log_jac_det = self.forward(x, c, rev=False)
        log_prob = - 0.5*torch.sum(z**2, 1) + log_jac_det - z.shape[1]/2 * math.log(2*math.pi)
        return log_prob
