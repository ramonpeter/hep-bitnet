import torch
import torch.nn as nn
from Networks.vblinear import VBLinear


class Resnet(nn.Module):
    """
    Simple Conditional Resnet class to build from a params dict
    """

    def __init__(self, param):
        """
        :param param: file with all relevant network parameters
        """
        super().__init__()
        # Read in the network specifications from the params
        self.param = param

        # number of residual blocks
        self.n_blocks = param["n_blocks"]

        # latent dimension
        self.intermediate_dim = self.param["intermediate_dim"]

        # input dimension
        self.dim = self.param["shape"][0] # just a hack since resnet should only see 1d inputs

        # output dimension
        self.out_dim = self.param.get("out_dim", self.dim)

        # conditional dimension
        self.n_con = self.param["n_con"]

        # number of layers per block
        self.layers_per_block = self.param["layers_per_block"]

        # special layers
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)

        # activation function
        self.activation = self.param.get("activation", "SiLU")

        # encoding
        self.encode_t = self.param.get("encode_t", False)
        self.encode_x = self.param.get("encode_x", False)
        # if true, network input will be encoded to higher dimension
        if self.encode_x:
            self.block_out_dim = self.encode_x_dim = self.param.get("encode_x_dim", 32)
            self.add_final = True
        else:
            self.block_out_dim = self.out_dim
            self.add_final = False

        self.conditional = self.param.get("conditional", False)
        self.encode_condition = self.param.get("encode_condition", False)

        self.bayesian = self.param.get("bayesian", False)

        # keep track of kl term or regularisation term
        self.bayesian_layers = []
        self.deter_layers = []

        # parameters for bayesian setup
        self.prior_prec = self.param.get("prior_prec", 1.0)
        self.map = False
        self.learn_sigma = self.param.get("learn_sigma", False)

        self.timesteps = self.param.get("timesteps", 1000)

        # if true embed time to higher dimension
        if self.encode_t:
            self.encode_t_scale = self.param.get("encode_t_scale", 30)
            self.encode_t_dim = self.param.get("encode_t_dim", 64)
            if self.param.get("encode_t_type", "Gauss") == "Gauss":
                # Use GaussianFourierProjection for the time if specified
                self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.encode_t_dim,
                                                                     scale=self.encode_t_scale),
                                           nn.Linear(self.encode_t_dim, self.encode_t_dim))
            elif self.param.get("encode_t_type", "Gauss") == "Embedding":
                # Use normal Embedding layer for the time
                self.embed = nn.Sequential(nn.Embedding(self.timesteps, self.encode_t_dim), nn.Linear(self.encode_t_dim,
                                                                                                      self.encode_t_dim))

        else:
            self.encode_t_dim = 1

        # if true, embed condition to higher dimension
        if self.encode_condition:
            self.encode_c_dim = self.param.get("encode_c_dim", 64)
            # self.encode_c = nn.Linear(self.n_con, self.encode_c_dim)
            self.encode_c = nn.Sequential(
                nn.Linear(self.n_con, self.encode_c_dim),
                nn.ReLU(),
                nn.Linear(self.encode_c_dim, self.encode_c_dim)
            )
        else:
            self.encode_c_dim = self.n_con

        # Build the Resnet blocks
        if self.encode_x:
            self.embed_x = nn.Linear(self.dim, self.encode_x_dim)
        else:
            self.encode_x_dim = self.dim

        # if self.learn_sigma == True:
        #    self.out_dim = 2*self.dim
        # else:
        #    self.out_dim = self.dim

        # build blocks
        self.blocks = nn.ModuleList([
            self.make_block()
            for _ in range(self.n_blocks)])

        if self.add_final:
            if self.bayesian == 0:
                self.final = nn.Linear(self.block_out_dim, self.out_dim)
                self.deter_layers.append(self.final)
            else:
                self.final = VBLinear(self.block_out_dim, self.out_dim, prior_prec=self.prior_prec)
                self.bayesian_layers.append(self.final)

        # initialize all weights of last layer in each block to zero
        if self.bayesian > 1:
            for block in self.blocks:
                block[-1].mu_w.data *= 0
                block[-1].bias.data *= 0
                # block[-1].logsig2_w.data *= 10**(-5)
        else:
            for block in self.blocks:
                block[-1].weight.data *= 0
                block[-1].bias.data *= 0

    def make_block(self):
        """
        Method to build the Resnet blocks with the defined specifications
        """
        if self.bayesian > 1:
            # Use bayesian layer instead of linear layer
            bays_layer = VBLinear(self.encode_x_dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim,
                                  prior_prec=self.prior_prec)
            layers = [bays_layer, nn.SiLU()]
            self.bayesian_layers.append(bays_layer)

            for _ in range(1, self.layers_per_block - 1):
                bays_layer = VBLinear(self.intermediate_dim, self.intermediate_dim, prior_prec=self.prior_prec)
                layers.append(bays_layer)
                self.bayesian_layers.append(bays_layer)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())
            bays_layer = VBLinear(self.intermediate_dim, self.block_out_dim, prior_prec=self.prior_prec)
            layers.append(bays_layer)
            self.bayesian_layers.append(bays_layer)

        else:
            # Use linear layers
            linear = nn.Linear(self.encode_x_dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim)
            self.deter_layers.append(linear)
            layers = [linear, nn.SiLU()]

            for _ in range(1, self.layers_per_block - 1):
                linear = nn.Linear(self.intermediate_dim, self.intermediate_dim)
                layers.append(linear)
                self.deter_layers.append(linear)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())

            linear = nn.Linear(self.intermediate_dim, self.block_out_dim)
            layers.append(linear)
            self.deter_layers.append(linear)

        return nn.Sequential(*layers)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        # Calculate kl term for each forward call separately
        self.kl = 0

        # Embed time
        if self.encode_t:
            t = self.embed(t)
            if self.param.get("encode_t_type", "Gauss") == "Embedding":
                t = torch.squeeze(t)

        # Embed condition
        if self.conditional:
            if self.encode_condition:
                condition = self.encode_c(condition)
            add_input = torch.cat([t, condition], 1)
        else:
            add_input = t

        # Embed model input
        if self.encode_x:
            x = self.embed_x(x)

        # Set all bayesian layers to self.map (if true weights won't be sampled, instead mean will be used)
        for bay_layer in self.bayesian_layers:
            bay_layer.map = self.map

        # create network output
        x = self.blocks[0](torch.cat([x, add_input], 1))
        for block in self.blocks[1:]:
            x = x + block(torch.cat([x, add_input], 1))

        if self.add_final:
            x = nn.SiLU()(x)
            x = self.final(x)

        # get kl term
        for bay_layer in self.bayesian_layers:
            self.kl += bay_layer.KL()

        # get regularization term
        for deter_layer in self.deter_layers:
            self.kl += (deter_layer.weight.pow(2) / self.prior_prec ** 2).sum()

        return x

class Conditional_Resnet(nn.Module):
    """
    Simple Conditional Resnet class to build from a params dict
    """

    def __init__(self, param):
        """
        :param param: file with all relevant network parameters
        """
        super().__init__()
        # Read in the network specifications from the params
        self.param = param

        # number of residual blocks
        self.n_blocks = param["n_blocks"]

        # latent dimension
        self.intermediate_dim = self.param["intermediate_dim"]

        # input dimension
        self.dim = self.param["shape"][0]  # just a hack since resnet should only see 1d inputs

        # output dimension
        self.out_dim = self.param.get("out_dim", self.dim)

        # conditional dimension
        self.n_con = self.param["n_con"]

        # number of layers per block
        self.layers_per_block = self.param["layers_per_block"]

        # special layers
        self.dropout = self.param.get("dropout", None)
        self.normalization = self.param.get("normalization", None)

        # activation function
        self.activation = self.param.get("activation", "SiLU")

        # encoding
        self.encode_t = self.param.get("encode_t", False)
        self.encode_x = self.param.get("encode_x", False)
        # if true, network input will be encoded to higher dimension
        if self.encode_x:
            self.block_out_dim = self.encode_x_dim = self.param.get("encode_x_dim", 32)
            self.add_final = True
        else:
            self.block_out_dim = self.out_dim
            self.add_final = False

        self.conditional = self.param.get("conditional", False)
        self.encode_condition = self.param.get("encode_condition", False)

        self.bayesian = self.param.get("bayesian", False)

        # keep track of kl term or regularisation term
        self.bayesian_layers = []
        self.deter_layers = []

        # parameters for bayesian setup
        self.prior_prec = self.param.get("prior_prec", 1.0)
        self.map = False
        self.learn_sigma = self.param.get("learn_sigma", False)

        self.timesteps = self.param.get("timesteps", 1000)

        # if true embed time to higher dimension
        if self.encode_t:
            self.encode_t_scale = self.param.get("encode_t_scale", 30)
            self.encode_t_dim = self.param.get("encode_t_dim", 64)
            # Use GaussianFourierProjection for the time if specified
            self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.encode_t_dim,
                                                                     scale=self.encode_t_scale))
        else:
            self.encode_t_dim = 1

        # if true, embed condition to higher dimension
        if self.encode_condition:
            self.encode_c_dim = self.param.get("encode_c_dim", 64)
            self.conditional_layers = self.param.get("conditional_layers",4)
            self.encode_c = self.make_condition_block()
        else:
            self.encode_c_dim = self.n_con

        # Build the Resnet blocks
        if self.encode_x:
            self.embed_x = nn.Linear(self.dim, self.encode_x_dim)
        else:
            self.encode_x_dim = self.dim

        self.block = self.make_block()

        # initialize all weights of last layer in each block to zero
        if self.bayesian > 1:
            self.block[-1].mu_w.data *= 0
            self.block[-1].bias.data *= 0
        else:
            self.block[-1].weight.data *= 0
            self.block[-1].bias.data *= 0

        if self.add_final:
            if self.bayesian == 0:
                self.final = nn.Linear(self.block_out_dim, self.out_dim)
                self.deter_layers.append(self.final)
            else:
                self.final = VBLinear(self.block_out_dim, self.out_dim, prior_prec=self.prior_prec)
                self.bayesian_layers.append(self.final)
    def make_block(self):
        """
        Method to build the Resnet blocks with the defined specifications
        """
        if self.bayesian > 1:
            # Use bayesian layer instead of linear layer
            bays_layer = VBLinear(self.encode_x_dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim,
                                  prior_prec=self.prior_prec)
            layers = [bays_layer, getattr(nn, self.activation)()]
            self.bayesian_layers.append(bays_layer)

            for _ in range(1, self.layers_per_block - 1):
                bays_layer = VBLinear(self.intermediate_dim, self.intermediate_dim, prior_prec=self.prior_prec)
                layers.append(bays_layer)
                self.bayesian_layers.append(bays_layer)
                if self.normalization is not None:
                    layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
                if self.dropout is not None:
                    layers.append(nn.Dropout(p=self.dropout))
                layers.append(getattr(nn, self.activation)())
            bays_layer = VBLinear(self.intermediate_dim, self.block_out_dim, prior_prec=self.prior_prec)
            layers.append(bays_layer)
            self.bayesian_layers.append(bays_layer)

        else:
            # Use linear layers
            linear = nn.Linear(self.encode_x_dim + self.encode_c_dim + self.encode_t_dim, self.intermediate_dim)
            self.deter_layers.append(linear)
            layers = [linear]

            for _ in range(1, self.layers_per_block - 1):
                linear = nn.Linear(self.intermediate_dim+self.encode_c_dim+ self.encode_t_dim, self.intermediate_dim)
                layers.append(linear)
                self.deter_layers.append(linear)

            linear = nn.Linear(self.intermediate_dim+self.encode_c_dim+ self.encode_t_dim, self.block_out_dim)
            layers.append(linear)
            self.deter_layers.append(linear)

        return nn.Sequential(*layers)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        # Calculate kl term for each forward call separately
        self.kl = 0

        # Embed time
        if self.encode_t:
            t = self.embed(t)
            if self.param.get("encode_t_type", "Gauss") == "Embedding":
                t = torch.squeeze(t)

        t = t.float()

        # Embed condition
        if self.conditional:
            if self.encode_condition:
                condition = self.encode_c(condition)
            add_input = torch.cat([t, condition], 1)
        else:
            add_input = t

        # Embed model input
        if self.encode_x:
            x = self.embed_x(x)

        # Set all bayesian layers to self.map (if true weights won't be sampled, instead mean will be used)
        for bay_layer in self.bayesian_layers:
            bay_layer.map = self.map

        # create network output
        x_0 = x.clone()
        for layer in self.block[:-1]:
            x_0 = layer(torch.cat([x_0, add_input], 1))
            if self.normalization:
                x_0 = getattr(nn, self.normalization)(self.intermediate_dim, device=x_0.device)(x_0)
            x_0 = getattr(nn, self.activation)()(x_0)

        x = x + self.block[-1](torch.cat([x_0, add_input], 1))

        if self.add_final:
            x = nn.SiLU()(x)
            x = self.final(x)

        # get kl term
        for bay_layer in self.bayesian_layers:
            self.kl += bay_layer.KL()

        # get regularization term
        #for deter_layer in self.deter_layers:
        #    self.kl += (deter_layer.weight.pow(2) / self.prior_prec ** 2).sum()

        return x

    def make_condition_block(self):
        # Use linear layers
        linear = nn.Linear(self.n_con, self.encode_c_dim)

        layers = [linear, getattr(nn, self.activation)()]

        for _ in range(1, self.conditional_layers - 1):
            linear = nn.Linear(self.encode_c_dim, self.encode_c_dim)
            layers.append(linear)
            layers.append(getattr(nn, self.activation)())

        linear = nn.Linear(self.encode_c_dim, self.encode_c_dim)
        layers.append(linear)

        return nn.Sequential(*layers)

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
