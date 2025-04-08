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
    ndim = len(x.shape)
    channels = [x]
    for d in break_dims:
        coord = torch.linspace(0, 1, x.shape[d], device=x.device)
        cast_shape = np.where(np.arange(ndim) == d, -1, 1)
        expand_shape = np.where(np.arange(ndim) == 1, 1, x.shape)
        channels.append(coord.view(*cast_shape).expand(*expand_shape))
    return torch.cat(channels, dim=1)

class DownBlock(nn.Module):
    """
    Similar to the downsampling UNet block
    """

    def __init__(self, in_channels, out_channels, down_kernel=None, down_stride=None,
                 down_pad=None, cond_dim=None, bottleneck=False, break_dims=None,
                 conv=nn.Conv3d, down_op=nn.Conv3d, norm=nn.BatchNorm3d, cylindrical=False):

        super(DownBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
        
        self.break_dims = break_dims or []
        self.act = nn.SiLU()
        self.cylindrical = cylindrical

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


        self.conv1 = conv(
            in_channels=in_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=3, padding=rect_pad
        )
        self.conv2 = conv(
            in_channels=out_channels+len(self.break_dims), out_channels=out_channels,
            kernel_size=3, padding=rect_pad
        )

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
            res = res + self.cond_layer(condition).view(
                -1, self.out_channels, *([1]*self.dim)
                )
        res = self.act(self.bn1(res))

        # conv2
        res = add_coord_channels(res, self.break_dims)
        if self.cylindrical:
            res = self.circ_pad(res, 2)
        res = self.conv2(res)
        res = self.act(self.bn2(res))

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
    Similar to upsampling UNet block
    """

    def __init__(self, in_channels, out_channels, up_kernel=None, up_stride=None,
                 up_crop=0, cond_dim=None, output_padding=0, break_dims=None,
                 conv=nn.Conv3d, up_op=nn.ConvTranspose3d, norm=nn.BatchNorm3d, cylindrical=False):

        super(UpBlock, self).__init__()

        self.out_channels = out_channels
        self.cond_layer = nn.Linear(cond_dim, out_channels)
 
        self.break_dims = break_dims or []
        self.cylindrical = cylindrical
        self.bn1 = norm(num_features=out_channels)
        self.bn2 = norm(num_features=out_channels)
        self.act = nn.SiLU()
        
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
            out = self.circ_crop(out)
        out = self.conv1(out)

        # conditioning
        if condition is not None:
            out = out + self.cond_layer(condition).view(
                -1, self.out_channels, *([1]*self.dim)
            )
        out = self.act(self.bn1(out))

        # conv2
        out = add_coord_channels(out, self.break_dims)
        if self.cylindrical:
            out = self.circ_crop(out)
        out = self.conv2(out)
        out = self.act(self.bn2(out))

        return out

class AutoEncoder(nn.Module):

    def __init__(self, param):

        super(AutoEncoder, self).__init__()

        defaults = {
            'dim': 3,
            'condition_dim': 0,
            'in_channels': 1,
            'out_channels': 1,
            'cylindrical': False,
            'ae_level_channels': [32, 1],
            'ae_level_kernels': [[2, 3]],
            'ae_level_strides': [[2, 3]],
            'ae_level_pads': [0],
            'ae_encode_c': False,
            'ae_encode_c_dim': 32,
            'ae_break_dims': None,
            'activation': nn.SiLU(),
            'ae_kl': False,
            'ae_latent_dim': 100,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)
        
        self.ae_break_dims = self.ae_break_dims or None
        
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
        self.total_condition_dim = self.ae_encode_c_dim if self.ae_encode_c else self.condition_dim

        if self.ae_encode_c_dim:
            self.c_encoding = nn.Sequential(
                nn.Linear(self.condition_dim, self.ae_encode_c_dim),
                nn.ReLU(),
                nn.Linear(self.ae_encode_c_dim, self.ae_encode_c_dim)
            )

        *level_channels, bottle_channel = self.ae_level_channels

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            DownBlock(
                n, m, self.ae_level_kernels[i], self.ae_level_strides[i],
                self.ae_level_pads[i], cond_dim=self.total_condition_dim,
                break_dims=self.ae_break_dims, norm=norm, conv=conv, down_op=down_op,
                cylindrical=self.cylindrical
            ) for i, (n, m) in enumerate(pairwise([self.in_channels] + level_channels))
        ])

        # Bottleneck block
        
        
        if self.ae_kl:
            self.conv_mu = conv(
                    in_channels=bottle_channel, out_channels=bottle_channel,
                    kernel_size=1
                    )
            self.conv_logvar = conv(
                    in_channels=bottle_channel, out_channels=bottle_channel,
                    kernel_size=1
                    )

            self.bottleneck = nn.ModuleList([
                conv(
                    in_channels=level_channels[-1]+len(self.ae_break_dims),
                    out_channels=bottle_channel, kernel_size=1
                ),
            ])
        else:
            self.bottleneck = nn.ModuleList([
                conv(
                    in_channels=level_channels[-1]+len(self.ae_break_dims),
                    out_channels=bottle_channel, kernel_size=1
                )
            ])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            UpBlock(
                n, m, self.ae_level_kernels[-1 -i], self.ae_level_strides[-1-i],
                self.ae_level_pads[-1-i], cond_dim=self.total_condition_dim,
                break_dims=self.ae_break_dims, norm=norm, conv=conv, up_op=up_op,
                cylindrical=self.cylindrical
            ) for i, (n, m) in enumerate(pairwise([bottle_channel] + level_channels[::-1]))
        ])

        # Output layer
        self.output_layer = conv(
            in_channels=level_channels[0]+len(self.ae_break_dims),
            out_channels=1, kernel_size=1
        )

    def forward(self, x, c=None):

        if self.ae_encode_c:
            c = self.c_encoding(c)

        z = self.encode(x, c=c)
        x = self.decode(z, c=c)

        return x


    def encode(self, x, c=None):

        out = x
        for down in self.down_blocks:
            out, _ = down(out, c)
        out = add_coord_channels(out, self.ae_break_dims)
        for btl in self.bottleneck:
            out = btl(out)
        if self.ae_kl:
            mu = self.conv_mu(out)
            logvar = self.conv_logvar(out)
            return mu, logvar
        return out

    def decode(self, z, c=None):

        out = z
        for up in self.up_blocks:
            out = up(out, residual=None, condition=c)
        out = add_coord_channels(out, self.ae_break_dims)
        out = self.output_layer(out)
        return torch.sigmoid(out)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z
