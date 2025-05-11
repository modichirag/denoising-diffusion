import torch
from torch import nn, einsum
from torch.func import vmap
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from functools import partial
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from attend import Attend
from utils import *
from networks import Linear, PositionalEmbedding

# small helper modules
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


def l2norm(t):
    return F.normalize(t, dim = -1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules
class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = RMSNorm(dim, scale = False)
        dim_hidden = dim * mult

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange('b d -> b 1 d')
        )

        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias = False),
            nn.SiLU()
        )

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    def forward(self, x, t):
        x = self.norm(x)
        x = self.proj_in(x)

        scale, shift = self.to_scale_shift(t).chunk(2, dim = -1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)

class SimpleFeedForward(nn.Module):
    def __init__(
        self, dim, hidden_sizes = [256, 256], activation=torch.nn.SiLU, latent_dim=None
    ):
        super().__init__()
        self.latent_dim = latent_dim if latent_dim is not None else 0
        layers = []
        prev_dim = dim + latent_dim + 1 # 1 for t
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_size))
            layers.append(activation())
            prev_dim = hidden_size

        # final layer
        layers.append(torch.nn.Linear(prev_dim, dim))

        # Wrap all layers in a Sequential module
        self.net = torch.nn.Sequential(*layers)

    def _single_forward(self, x, t, latent):
        t = t.unsqueeze(-1)
        return self.net(torch.cat((x, t, latent)))

    def forward(self, x, t, latents=None):
        batch_size = x.shape[0]
        if latents is not None:
            if latents.shape[0] != batch_size:
                raise ValueError(f"Latents batch size {latents.shape[0]} does not match x/t batch size {batch_size}")
            if latents[0].numel() != self.latent_dim:
                raise ValueError(f"Latents feature dimension {latents[0].numel()} does not match model's feature_dim_latent {self.latent_dim}")
            latents = latents.reshape(batch_size, -1)
        else:
            latents = torch.zeros(x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype)
        return vmap(self._single_forward, in_dims=(0, 0, 0), out_dims=(0))(x, t, latents)


class FeedForwardwithEMB(nn.Module):
    def __init__(
        self, dim, emb_channels=64, hidden_sizes = [256, 256], activation=torch.nn.SiLU, latent_dim=None
    ):
        super().__init__()
        self.latent_dim = latent_dim if latent_dim is not None else 0
        self.emb_channels = emb_channels
        layers = []
        prev_dim = dim + emb_channels # emb for time
        if latent_dim > 0:
            prev_dim += emb_channels # emb for latent
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_size))
            layers.append(activation())
            prev_dim = hidden_size
        # final layer
        layers.append(torch.nn.Linear(prev_dim, dim))
        # Wrap all layers in a Sequential module
        self.final_net = torch.nn.Sequential(*layers)
        self.map_t = PositionalEmbedding(num_channels=emb_channels, max_positions=1)

        if latent_dim > 0:
            self.map_latents = torch.nn.Sequential(
                Linear(in_features=latent_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(latent_dim)),
                torch.nn.SiLU(),
                Linear(in_features=emb_channels, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(latent_dim)),
                torch.nn.SiLU(),
                Linear(in_features=emb_channels, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(latent_dim))
            )

    def forward(self, x, t, latents=None):
        batch_size = x.shape[0]
        if latents is not None:
            if latents.shape[0] != batch_size:
                raise ValueError(f"Latents batch size {latents.shape[0]} does not match x/t batch size {batch_size}")
            if latents[0].numel() != self.latent_dim:
                raise ValueError(f"Latents feature dimension {latents[0].numel()} does not match model's feature_dim_latent {self.latent_dim}")
            latents = latents.reshape(batch_size, -1)
        t_emb = self.map_t(t)
        latent_emb = self.map_latents(latents) if self.latent_dim > 0 else torch.zeros_like(t_emb)
        return self.final_net(torch.cat((x, t_emb, latent_emb), dim=-1))