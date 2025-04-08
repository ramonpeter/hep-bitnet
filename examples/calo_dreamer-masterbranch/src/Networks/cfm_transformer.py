import torch
import torch.nn as nn
import math

class CFMTransformer(nn.Module):
    """
    Simple Conditional Resnet class to build from a params dict
    """
    def __init__(self, params):
        super().__init__()
        # Read in the network specifications from the params
        self.params = params

        self.dim_embedding = self.params["dim_embedding"]
        self.dims_in = self.params["shape"][0]
        self.dims_c = self.params["n_con"]
        self.bayesian = False


        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=self.params["n_head"],
            num_encoder_layers=self.params["n_encoder_layers"],
            num_decoder_layers=self.params["n_decoder_layers"],
            dim_feedforward=self.params["dim_feedforward"],
            dropout=self.params.get("dropout", 0.0),
            activation=self.params.get("activation", "relu"),
            batch_first=True,
        )
        self.layer = nn.Linear(self.dim_embedding + 1, 1)
        # Initalize to zero
        self.layer.weight.data *= 0
        self.layer.bias.data *= 0

        self.embeds = self.params.get("embeds", False)
        if self.embeds:
            self.x_embed = nn.Linear(1, self.dim_embedding)
            self.c_embed = nn.Linear(1, 2 * self.dim_embedding)
            self.pos_embed_x = nn.Embedding(self.dims_in, self.dim_embedding)
            self.pos_embed_c = nn.Embedding(self.dims_c, 2 * self.dim_embedding)
            #self.time_embed = nn.Linear(1, self.dim_embedding)
            self.encode_t_scale = self.params.get("encode_t_scale", 30)
            self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.dim_embedding,
                                                                      scale=self.encode_t_scale),
                                       nn.Linear(self.dim_embedding, self.dim_embedding))
            self.layer = nn.Linear(3 * self.dim_embedding, 1)

            print("Using embeds")
            self.transformer = nn.Transformer(
                d_model=2 * self.dim_embedding,
                nhead=self.params["n_head"],
                num_encoder_layers=self.params["n_encoder_layers"],
                num_decoder_layers=self.params["n_decoder_layers"],
                dim_feedforward=self.params["dim_feedforward"],
                dropout=self.params.get("dropout", 0.0),
                activation=self.params.get("activation", "relu"),
                batch_first=True,
            )
            #self.x_embed = SinCos_embedding(n_frequencies=self.dim_embedding / 2)
            #self.pos_embed = PositionalEncoding(d_model=self.dim_embedding)




    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        if self.embeds:
            if t is not None:
                p = self.x_embed(p.unsqueeze(-1))
                p = p + self.pos_embed_x(torch.arange(n_components, device=p.device))
                t = self.time_embed(t).unsqueeze(1)
                p = torch.cat([t.repeat(1, p.size(1), 1), p], dim=-1)
                return p
            else:
                p = self.c_embed(p.unsqueeze(-1))
                p = p + self.pos_embed_c(torch.arange(n_components,device=p.device))
                return p

        else:
            one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
                None, : p.shape[1], :
            ].expand(p.shape[0], -1, -1)
            if t is None:
                p = p.unsqueeze(-1)
            else:
                p = torch.cat([p.unsqueeze(-1), t.unsqueeze(1).expand(t.shape[0], p.shape[1], self.encode_t_dim)], dim=-1)
            n_rest = self.dim_embedding - n_components - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, x, t, condition=None):
        """
        forward method of our Resnet
        """
        self.kl = 0

        if condition is None:
            embedding = self.transformer.decoder(
                self.compute_embedding(
                    x,
                    n_components=self.dims_in,
                    t=t
                ),
                torch.zeros((x.size(0), x.size(1), 2*self.dim_embedding), device=x.device, dtype=x.dtype)
            )
        else:
            embedding = self.transformer(
                src=self.compute_embedding(
                    condition,
                    n_components=self.dims_c
                ),
                tgt=self.compute_embedding(
                    x,
                    n_components=self.dims_in,
                    t=t
                )
            )
        if self.embeds:
            t = self.time_embed(t)
        v_pred = self.layer(torch.cat([t.unsqueeze(1).repeat(1, self.dims_in, 1), embedding], dim=-1)).squeeze()
        return v_pred

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SinCos_embedding(nn.Module):

    def __init__(self, n_frequencies: int, sigmoid=True):
        super().__init__()
        self.arg = nn.Parameter(2 * math.pi * 2**torch.arange(n_frequencies), requires_grad=False)
        self.sigmoid = sigmoid

    def forward(self, x):
        if self.sigmoid:
            x_pp = nn.functional.sigmoid(x)
        else:
            x_pp = x
        frequencies = (x_pp.unsqueeze(-1)*self.arg).reshape(x_pp.size(0), x_pp.size(1), -1)
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)


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