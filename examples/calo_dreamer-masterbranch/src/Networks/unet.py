from itertools import pairwise
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dim_from_conv(conv_class):
    for i in 2, 3:
        if f'{i}d' in conv_class.__name__:
            return i

def add_coord_channels(x, break_dims=None):
    ndim = len(x.shape)  # TODO: move to init? and other optimisations
    channels = [x]
    for d in break_dims:
        coord = torch.linspace(0, 1, x.shape[d], device=x.device)
        cast_shape = np.where(np.arange(ndim) == d, -1, 1)
        expand_shape = np.where(np.arange(ndim) == 1, 1, x.shape)
        channels.append(coord.view(*cast_shape).expand(*expand_shape))
    return torch.cat(channels, dim=1)

class DownBlock(nn.Module):
    """
    Downsampling block for U-Net.
    modified from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py

    __init__ parameters:
        in_channels  -- number of input channels
        out_channels -- desired number of output channels
        down_kernel  -- kernel for the downsampling operation
        down_stride  -- stride for the downsampling operation
        down_pad     -- size of the circular padding (ignored if `cylindrical` is True)
        cond_dim     -- dimension of conditional input
        bottleneck   -- whether this is the bottlneck block (excludes downsampling)
        break_dims   -- the indices of dimensions at which translation symmetry should be broken
        conv         -- the convolution operation to use (Conv 2d or 3d)
        down_op      -- the downsampling operation to use (Conv 2d or 3d) # TODO: MaxPool?
        norm         -- the normalization operation to use (BatchNorm 2d or 3d)
        cylindrical  -- whether to use cylindrical convolutions
    """

    def __init__(self, in_channels, out_channels, down_kernel=None, down_stride=None, down_pad=None,
                 cond_dim=None, bottleneck=False, break_dims=None, conv=nn.Conv3d, down_op=nn.Conv3d,
                 norm=nn.BatchNorm3d, cylindrical=False):

        super(DownBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.break_dims = break_dims or []
        self.cylindrical = cylindrical
        self.act = nn.SiLU()
        self.bn1 = norm(num_features=out_channels)
        self.bn2 = norm(num_features=out_channels)

        # reconcile padding based on dimension and cylindrical
        self.dim = dim_from_conv(conv)
        rect_pad = [1] * self.dim
        if cylindrical:
            rect_pad[-2] = 0
            self.circ_pad = ((lambda x, p: F.pad(x, (0, 0, 0, p), mode='circular'))
                             if self.dim == 2 else
                             (lambda x, p: F.pad(x, (0, 0, 0, p, 0, 0), mode='circular')))

        # initialize convolutions                
        self.conv1 = conv(
            in_channels=in_channels+len(self.break_dims), bias=False, out_channels=out_channels,
            kernel_size=3, padding=rect_pad
        )
        self.conv2 = conv(
            in_channels=out_channels+len(self.break_dims), bias=False, out_channels=out_channels,
            kernel_size=3, padding=rect_pad
        )

        # add downsampling unless this is the bottleneck block
        self.bottleneck = bottleneck
        if not bottleneck:
            # reconcile padding
            down_pad = [down_pad] * self.dim if type(down_pad) is int else down_pad
            if cylindrical:
                down_pad[-2] = 0
            self.down_pad = (
                (down_kernel if type(down_kernel) is int else down_kernel[-2])
                - (down_stride if type(down_stride) is int else down_stride[-2])
            )      
            # initialize pooling          
            self.pooling = down_op(
                in_channels=out_channels+len(self.break_dims), out_channels=out_channels,
                kernel_size=down_kernel, stride=down_stride, padding=down_pad
            )

    def forward(self, input, condition=None):

        # conv1
        res = add_coord_channels(input, self.break_dims)
        if self.cylindrical:
            res = self.circ_pad(res, 2)
        res = self.conv1(res)

        # conditioning
        if condition is not None:
            res = res + self.cond_layer(condition).view(-1, self.out_channels, *([1]*self.dim))
        res = self.act(self.bn1(res))

        # conv2
        res = add_coord_channels(res, self.break_dims)
        if self.cylindrical:
            res = self.circ_pad(res, 2)
        res = self.act(self.bn2(self.conv2(res)))

        # pooling
        out = None
        if not self.bottleneck:
            out = add_coord_channels(res, self.break_dims)
            if self.cylindrical:
                out = self.circ_pad(out, self.down_pad)
            out = self.pooling(out)
        else:
            out = res
        return out, res

class UpBlock(nn.Module):

    """
    Upsampling block for U-Net.
    modified from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py

    __init__ parameters:
        in_channels    -- number of input channels
        out_channels   -- desired number of output channels
        up_kernel      -- kernel for the upsampling operation
        up_stride      -- stride for the upsampling operation
        up_crop        -- size of cropping in the circular dimension (ignored if `cylindrical` is True)
        cond_dim       -- dimension of conditional input
        output_padding -- argument forwarded to ConvTranspose
        break_dims     -- the indices of dimensions at which translation symmetry should be broken
        conv           -- the convolution operation to use (Conv 2d or 3d)
        up_op          -- the upsampling operation to use (Conv 2d or 3d) # TODO: MaxPool?
        norm           -- the normalization operation to use (BatchNorm 2d or 3d)  
        cylindrical    -- whether to use cylindrical convolutions
    """

    def __init__(self, in_channels, out_channels, up_kernel=None, up_stride=None, up_crop=0,
                 cond_dim=None, output_padding=0, break_dims=None, conv=nn.Conv3d,
                 up_op=nn.ConvTranspose3d, norm=nn.BatchNorm3d, cylindrical=False):

        super(UpBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        self.break_dims = break_dims or []
        self.cylindrical = cylindrical
        self.act = nn.SiLU()
        self.bn1 = norm(num_features=out_channels)
        self.bn2 = norm(num_features=out_channels)

        # reconcile padding based on dimension and cylindrical
        self.dim = dim_from_conv(conv)
        rect_pad = [1] * self.dim
        up_crop = [up_crop] * self.dim if type(up_crop) is int else up_crop
        if cylindrical:
            up_crop[-2] = 0
            rect_pad[-2] = 0
            self.up_crop = ((up_kernel if type(up_kernel) is int else up_kernel[-2])
                          - (up_stride if type(up_stride) is int else up_stride[-2]))
            self.circ_pad = ((lambda x, p: F.pad(x, (0, 0, 0, p), mode='circular'))
                             if self.dim == 2 else
                             (lambda x, p: F.pad(x, (0, 0, 0, p, 0, 0), mode='circular')))
        
        # initialize convolutions
        self.upconv1 = up_op(
            in_channels=in_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=up_kernel, stride=up_stride, padding=up_crop, output_padding=output_padding
        )
        self.conv1 = conv(
            in_channels=out_channels+len(self.break_dims), bias=False, out_channels=out_channels,
            kernel_size=3, padding=rect_pad
        )
        self.conv2 = conv(
            in_channels=out_channels+len(self.break_dims), bias=False, out_channels=out_channels,
            kernel_size=3, padding=rect_pad
        )


    def forward(self, input, residual=None, condition=None):

        # upsample
        out = add_coord_channels(input, self.break_dims)
        out = self.upconv1(out)
        if self.cylindrical:
            out = self.circ_crop(out)

        # residual connection
        if residual != None:          
            out = out + residual

        # conv1
        out = add_coord_channels(out, self.break_dims)
        if self.cylindrical:
            out = self.circ_pad(out, 2)
        out = self.conv1(out)

        # conditioning
        if condition is not None:
            out = out + self.cond_layer(condition).view(-1, self.out_channels, *([1]*self.dim))
        out = self.act(self.bn1(out))

        # conv2
        out = add_coord_channels(out, self.break_dims)
        if self.cylindrical:
            out = self.circ_pad(out, 2)
        out = self.act(self.bn2(self.conv2(out)))

        return out

    def circ_crop(self, x):
        """
        Cropping operation that averages over cirular padding
                          X0 | X1 | ... | X7 | X8 | x0 | x1
                                     |
                                     V
            (X0+x0)/2 | (X1+x1)/2 | ... | X7 | X8
        """
        C = self.up_crop
        # store edge
        r_edge = x[..., -C:, :]
        # crop
        x = x[..., :-C, :]
        # average with cropped edge
        x[..., :C, :] = (x[..., :C, :] + r_edge)/2
        return x        


class UNet(nn.Module):
    """
    Model class for convolutional U-Net.
    modified from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py

    :param param: A dictionary containing the relevant network parameters:
                  
        dim       -- Number of spatial dimensions in the input (determines convolution dim).
        condition_dim  -- Dimension of conditional input
        in_channels    -- Number of channels in the input
        out_channels   -- Number of channels in the network output
        level_channels -- Number of channels at each level (count top-down)
        level_kernels  -- Kernel shape for the up/down sampling operations
        level_strides  -- Stride shape for the up/down sampling operations
        level_pads     -- Padding for the up/down sampling operations
        encode_t       -- Whether or not to embed the time input
        encode_t_dim   -- Dimension of the time embedding
        encode_t_scale -- Scale for the Gaussian Fourier projection
        encode_c       -- Whether or not to embed the conditional input
        encode_c_dim   -- Dimension of the condition embedding            
        activation     -- Activation function for hidden layers
        break_dims     -- the indices of dimensions at which translation symmetry
                          should be broken                  
        bayesian       -- Whether or not to use bayesian layers
    """

    def __init__(self, param):

        super(UNet, self).__init__()

        defaults = {
            'dim': 3,
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'level_channels': [32, 64, 128],
            'break_dims': None,
            'cylindrical': False,
            'level_kernels': [[3, 2, 3], [3, 2, 3]],
            'level_strides': [[3, 2, 3], [3, 2, 3]],
            'level_pads': [0, 0],
            'encode_t': False,
            'encode_t_dim': 32,
            'encode_t_scale': 30,
            'encode_c': False,
            'encode_c_dim': 32,
            'bayesian': False,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)
        self.break_dims = self.break_dims or []

        # select conv and norm layers based on input dimension
        if self.dim == 2:
            norm    = nn.BatchNorm2d
            conv    = nn.Conv2d
            down_op = nn.Conv2d
            up_op   = nn.ConvTranspose2d
        elif self.dim == 3:
            norm    = nn.BatchNorm3d
            conv    = nn.Conv3d
            down_op = nn.Conv3d
            up_op   = nn.ConvTranspose3d
        else: raise ValueError(self.dim)

        # Conditioning
        self.total_condition_dim = (self.encode_t_dim if self.encode_t else 1) \
                                 + (self.encode_c_dim if self.encode_c else self.condition_dim)
        if self.encode_t_dim:
            fourier_proj = GaussianFourierProjection(
                embed_dim=self.encode_t_dim, scale=self.encode_t_scale
            )
            self.t_encoding = nn.Sequential(
                fourier_proj, nn.Linear(self.encode_t_dim, self.encode_t_dim)
            )
        if self.encode_c_dim:
            self.c_encoding = nn.Sequential(
                nn.Linear(self.condition_dim, self.encode_c_dim),
                nn.ReLU(),
                nn.Linear(self.encode_c_dim, self.encode_c_dim)
            )

        *level_channels, bottle_channel = self.level_channels

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(
                n, m, self.level_kernels[i], self.level_strides[i], self.level_pads[i],
                break_dims=self.break_dims, cond_dim=self.total_condition_dim, norm=norm, conv=conv,
                down_op=down_op, cylindrical=self.cylindrical
            ) for i, (n, m) in enumerate(pairwise([self.in_channels] + level_channels))
        ])

        # Bottleneck block
        self.bottleneck_block = DownBlock(
            level_channels[-1], bottle_channel, bottleneck=True, break_dims=self.break_dims,
            cond_dim=self.total_condition_dim, norm=norm, conv=conv, cylindrical=self.cylindrical
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(
                n, m, self.level_kernels[-1 -i], self.level_strides[-1-i], self.level_pads[-1-i],
                break_dims=self.break_dims, cond_dim=self.total_condition_dim, norm=norm, conv=conv,
                up_op=up_op, cylindrical=self.cylindrical
            ) for i, (n, m) in enumerate(pairwise([bottle_channel] + level_channels[::-1]))
        ])

        # Output layer
        self.output_layer = conv(
            in_channels=level_channels[0]+len(self.break_dims), out_channels=self.out_channels,
            kernel_size=[1]*self.dim
        )
        
        self.kl = torch.zeros(())

    def forward(self, x, t, c=None):

        if self.encode_t:
            t = self.t_encoding(t)
        if c is None:
            condition = t
        else:
            if self.encode_c:
                c = self.c_encoding(c)
            condition = torch.cat([t, c], 1)

        residuals = []
        out = x

        # down path
        for down in self.down_blocks:
            out, res = down(out, condition)
            residuals.append(res)

        # bottleneck
        out, _ = self.bottleneck_block(out, condition)

        # up path
        for up in self.up_blocks:
            out = up(out, residuals.pop(), condition)

        # output
        out = add_coord_channels(out, self.break_dims)
        out = self.output_layer(out)

        return out

# TODO: Move this (and defn in resnet.py) to separate file
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)