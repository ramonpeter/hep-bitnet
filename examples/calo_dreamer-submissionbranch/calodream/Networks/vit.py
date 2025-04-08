"""Modified from github.com/facebookresearch/DiT/blob/main/models.py""" 

import math
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
#from timm.models.vision_transformer import Mlp

from bitnet.bitlinear import BitLinear158b as BitLinear

class ViT(nn.Module):
    """
    Vision transformer-based diffusion network.
    """
    # TODO: Embed time and condition separately into hidden_dim//2 then concat
    # TODO: Implement padding

    def __init__(self, param):

        super(ViT, self).__init__()

        defaults = {
            'dim': 3,
            'shape': (1, 45, 16, 9),
            'patch_shape': (3, 2, 3),
            'condition_dim': 46,
            'hidden_dim': 24 * 24,
            'depth': 4,
            'num_heads': 4,
            'mlp_ratio': 4.0,
            'attn_drop': 0.,
            'proj_drop': 0.,
            'pos_embedding_coords': 'cartesian',
            'learn_pos_embed': False,
            'causal_attn': False,
            'cos_attn': False
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)
        self.use_bitnet = param.get("use_bitnet_shape", "None")

        in_channels, *axis_sizes = self.shape

        # check shapes
        for i, (s, p) in enumerate(zip(axis_sizes, self.patch_shape)):
            assert s % p == 0, \
                f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
        assert not self.hidden_dim % (2*self.dim), \
            f"Hidden dim should be divisible by {2*self.dim} (for fourier position embeddings)"
        
        # initialize x,t,c embeddings
        patch_dim = math.prod(self.patch_shape) * in_channels
        if self.use_bitnet in ["aggressive"]:
            self.x_embedder = BitLinear(patch_dim, self.hidden_dim)
        else:
            self.x_embedder = nn.Linear(patch_dim, self.hidden_dim)
        self.t_embedder = TimestepEmbedder(self.hidden_dim, use_bitnet=self.use_bitnet)

        if self.use_bitnet in ["aggressive"]:
            self.c_embedder = nn.Sequential(
                BitLinear(self.condition_dim, self.hidden_dim),
                nn.SiLU(),
                BitLinear(self.hidden_dim, self.hidden_dim),
            )
        elif self.use_bitnet in ["nonlast", "nonlastfc2"]:
            self.c_embedder = nn.Sequential(
                BitLinear(self.condition_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.c_embedder = nn.Sequential(
                nn.Linear(self.condition_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        
        self.num_patches = [s // p for s, p in zip(axis_sizes, self.patch_shape)]
        
        # initialize position embeddings
        if self.learn_pos_embed:
            l, a, r = self.num_patches
            self.pos_embed_freqs = nn.Parameter(torch.randn(self.hidden_dim//2))
            self.register_buffer('lgrid', torch.arange(l)/l)
            self.register_buffer('agrid', torch.arange(a)/a)
            self.register_buffer('rgrid', torch.arange(r)/r)
        else:
            self.register_buffer(
                'pos_embed',
                get_2d_cylindrical_sincos_pos_embed(self.num_patches, self.hidden_dim)
                if self.pos_embedding_coords == 'cylindrical' and self.dim == 2 else
                get_2d_cartesian_sincos_pos_embed(self.num_patches, self.hidden_dim)
                if self.pos_embedding_coords == 'cartesian' and self.dim == 2 else            
                get_3d_cylindrical_sincos_pos_embed(self.num_patches, self.hidden_dim)
                if self.pos_embedding_coords == 'cylindrical' and self.dim == 3 else
                get_3d_cartesian_sincos_pos_embed(self.num_patches, self.hidden_dim)
                if self.pos_embedding_coords == 'cartesian' and self.dim == 3 else None
            )

        # compute layer-causal attention mask
        if self.causal_attn:
            assert self.dim == 3, "A layer-causal attention mask should only be used in 3d"
            l, a, r = self.num_patches
            patch_idcs = torch.arange(l*a*r)
            self.attn_mask = nn.Parameter(
                patch_idcs[:,None]//(a*r) >= patch_idcs[None,:]//(a*r), # tril (causal)
                requires_grad=False
            )

        # initialize transformer stack
        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_dim, self.num_heads, mlp_ratio=self.mlp_ratio, use_bitnet=self.use_bitnet,
                attn_drop=self.attn_drop, proj_drop=self.proj_drop, cos_attn=self.cos_attn,
                attn_mask=self.attn_mask if self.causal_attn else None
            ) for _ in range(self.depth)
        ])

        # initialize output layer
        if self.use_bitnet in ["aggressive"]:
            use_bit=True
        else:
            use_bit=False
        self.final_layer = FinalLayer(self.hidden_dim, self.patch_shape, in_channels, use_bitnet=use_bit)

        # custom weight initialization
        self.initialize_weights()

    def learnable_pos_embedding(self):
        wz, wy, wx = (self.pos_embed_freqs * 2 * math.pi).chunk(3)
        z, y, x = torch.meshgrid(self.lgrid, self.agrid, self.rgrid, indexing='ij')
        z = z.flatten()[:, None] * wz[None, :]
        y = y.flatten()[:, None] * wy[None, :]
        x = x.flatten()[:, None] * wx[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)
        return pe

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Forward pass of DiT.
        x: (B, C, *axis_sizes) tensor of spatial inputs
        t: (B,) tensor of diffusion timesteps
        c: (B, K) tensor of conditions
        """
        x = self.to_patches(x)
        if self.learn_pos_embed:
            x = self.x_embedder(x) + self.learnable_pos_embedding()
        else:
            x = self.x_embedder(x) + self.pos_embed # (B, T, D), where T = (L*A*R)/prod(patch_size)

        t = self.t_embedder(t)                   # (B, D)
        c = self.c_embedder(c)                   # (B, D)
        c = t + c                                # (B, D)
        
        for block in self.blocks:
            x = block(x, c)                      # (B, T, D)
        x = self.final_layer(x, c)               # (B, T, prod(patch_shape) * out_channels)
        x = self.from_patches(x)                 # (B, out_channels, *axis_sizes)            

        return x

    def from_patches(self, x):
        if self.dim == 3:
            x = rearrange(
                x, 'b (l a r) (p1 p2 p3 c) -> b c (l p1) (a p2) (r p3)',
                **dict(zip(('l', 'a', 'r', 'p1', 'p2', 'p3'), self.num_patches+self.patch_shape))
            )
        elif self.dim == 2:
            x = rearrange(
                x, 'b (a r) (p1 p2 c) -> b c (a p1) (r p2)',
                **dict(zip(( 'a', 'r', 'p1', 'p2'), self.num_patches+self.patch_shape))
            )
        else:
            raise ValueError(self.dim)
        return x

    def to_patches(self, x):
        if self.dim == 3:
            x = rearrange(
                x, 'b c (l p1) (a p2) (r p3) -> b (l a r) (p1 p2 p3 c)',
                **dict(zip(('p1', 'p2', 'p3'), self.patch_shape))
            )
        elif self.dim == 2:
            x = rearrange(
                x, 'b c (a p1) (r p2) -> b (a r) (p1 p2 c)',
                **dict(zip(('p1', 'p2'), self.patch_shape))
            )
        else:
            raise ValueError(self.dim)
        return x       

                      

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_bitnet="None", **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, use_bitnet=use_bitnet, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0, use_bitlinear=use_bitnet)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, patch_shape, out_channels, use_bitnet=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        if use_bitnet:
            self.linear = BitLinear(hidden_dim, math.prod(patch_shape) * out_channels)
        else:
            self.linear = nn.Linear(hidden_dim, math.prod(patch_shape) * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, use_bitnet="None"):
        super().__init__()
        if use_bitnet in ["aggressive"]:
            self.mlp = nn.Sequential(
                BitLinear(frequency_embedding_size, hidden_size),
                nn.SiLU(),
                BitLinear(hidden_size, hidden_size),
            )
        elif use_bitnet in ["nonlast", "nonlastfc2"]:
            self.mlp = nn.Sequential(
                BitLinear(frequency_embedding_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Attention(nn.Module):
    
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            use_bitnet: str = "None",
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_mask: torch.Tensor = None,
            norm_layer: nn.Module = nn.LayerNorm,
            cos_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if use_bitnet in ['full', 'qkv', 'aggressive', 'nonlast', 'nonlastfc2', 'noemb']:
            self.qkv = BitLinear(dim, dim * 3)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        if use_bitnet in ["aggressive", "noemb"]:
            self.proj = BitLinear(dim, dim)
        else:
            self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_mask = attn_mask
        self.cos_attn = cos_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.cos_attn:
            x = cosine_distance(q, k, v)
        else:
            x = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=self.attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def cosine_distance(q, k ,v):
    k = k.transpose(-2, -1)
    dots = q @ k
    scale = torch.einsum('bhi, bhj -> bhij',
            (torch.norm(q, 2, dim = -1), torch.norm(k, 2, dim = -2)))
    out = torch.matmul(dots/scale, v)
    return out

def get_2d_cylindrical_sincos_pos_embed(num_patches, dim, temperature=10000):
    """
    Embeds patch positions based directly on input indices, which are assumed
    to be depth, angle, radius.
    """
    A, R = num_patches
    y, x = torch.meshgrid(
        torch.arange(A) / A, torch.arange(R) / R, indexing='ij'
    )

    fourier_dim = dim // 4
    omega = torch.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    # padding can be implemented here

    return pe


def get_2d_cartesian_sincos_pos_embed(num_patches, dim, temperature=10000):
    """
    Embeds patch positions after converting input indices from polar to cartesian
    coordinates. i.e. depth, angle, radius -> depth, height, width
    """
    A, R = num_patches
    alpha, r = torch.meshgrid(
        torch.arange(A) * (2 * math.pi / A), torch.arange(R) / R,
        indexing='ij'
    )
    x = r * alpha.cos()
    y = r * alpha.sin()

    fourier_dim = dim // 4
    omega = torch.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    # padding can be implemented here

    return pe


def get_3d_cylindrical_sincos_pos_embed(num_patches, dim, temperature=10000):    
    """
    Embeds patch positions based directly on input indices, which are assumed
    to be depth, angle, radius.
    """
    L, A, R = num_patches
    z, y, x = torch.meshgrid(
        torch.arange(L)/L, torch.arange(A)/A,torch.arange(R)/R, indexing='ij'
    )

    fourier_dim = dim // 6
    omega = torch.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)    
    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)
    # padding can be implemented here

    return pe


def get_3d_cartesian_sincos_pos_embed(num_patches, dim, temperature=10000):    
    """
    Embeds patch positions after converting input indices from polar to cartesian
    coordinates. i.e. depth, angle, radius -> depth, height, width
    """
    L, A, R = num_patches
    z, alpha, r = torch.meshgrid(
        torch.arange(L)/L, torch.arange(A)*(2*math.pi/A), torch.arange(R)/R,
        indexing='ij'
    )
    x = r*alpha.cos()
    y = r*alpha.sin()

    fourier_dim = dim // 6
    omega = torch.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)    
    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)
    # padding can be implemented here

    return pe  

# new by Claudius, taken from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py                                                                                           

from functools import partial
from timm.layers.helpers import to_2tuple

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            use_bitlinear="None",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        if use_bitlinear in ['full', 'mlp', "aggressive", "nonlast", "nonlastfc2", "noemb"]:
            self.fc1 = BitLinear(in_features, hidden_features)
        else:
            self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        if use_bitlinear in ['full', 'mlp', "aggressive", "nonlastfc2", "noemb"]:
            self.fc2 = BitLinear(hidden_features, out_features)
        else:
            self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
