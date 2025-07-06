
import glob
import logging
import math
import os
import random
from collections import namedtuple
from functools import partial

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.fft as fft

from accelerate import Accelerator
from einops import rearrange, reduce

from ema_pytorch import EMA

from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader

from torchvision import utils
from tqdm.auto import tqdm

from torchaudio import load
from soundfile import write
from os.path import join

# from src.inference import evaluate_model
from src.other import si_sdr, pad_spec
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi




ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def spec_fwd(self, spec):
        # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
        # and introduced numerical error
        e = 0.5
        spec = spec.abs() ** e * torch.exp(1j * spec.angle())
        spec = spec * 0.15
        return spec

    def forward(self, pred_stft, true_stft):
        device = pred_stft.device
        pred_stft_real, pred_stft_imag = pred_stft.real, pred_stft.imag
        true_stft_real, true_stft_imag = true_stft.real, true_stft.imag
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (0.7))
        pred_imag_c = pred_stft_imag / (pred_mag ** (0.7))
        true_real_c = true_stft_real / (true_mag ** (0.7))
        true_imag_c = true_stft_imag / (true_mag ** (0.7))
        real_loss = nn.MSELoss()(pred_real_c, true_real_c)
        imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
        mag_loss = nn.MSELoss()(pred_mag ** (0.3), true_mag ** (0.3))
        y_pred = (pred_stft_real + 1j * pred_stft_imag).squeeze()
        y_true = (true_stft_real + 1j * true_stft_imag).squeeze()

        y_pred = torch.istft(self.spec_fwd(y_pred), 510, 128, window=torch.hann_window(510).pow(0.5).to(device))
        y_true = torch.istft(self.spec_fwd(y_true), 510, 128, window=torch.hann_window(510).pow(0.5).to(device))
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (
                    torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-8)
        sisnr = - torch.log10(torch.norm(y_true, dim=-1, keepdim=True) ** 2 / (
                    torch.norm(y_pred - y_true, dim=-1, keepdim=True) ** 2 + 1e-8) + 1e-8).mean()

        return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr
def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(spec):
    if isinstance(spec, list):
        return [spec[k] * 2 - 1 for k in range(len(spec))]
    else:
        return spec * 2 - 1


def unnormalize_to_zero_to_one(spec):
    if isinstance(spec, list):
        return [(spec[k] + 1) * 0.5 for k in range(len(spec))]
    else:
        return (spec + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=(2, 2), mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), (4, 4), (2, 2), 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()  # 归一化权重

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # 定义了一个可学习的参数

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# 先进行了归一化的操作，再进行其他的操作
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        # 对时间进行嵌入
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

 # 调用Block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 得到qkv
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, [0], :, :] * mask[:, [0], :, :]- spec[:, [1], :, :] * mask[:, [1], :, :]
        s_imag = spec[:, [1], :, :] * mask[:, [0], :, :] + spec[:, [0], :, :] * mask[:, [1], :, :]
        s = torch.cat([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class cplxdecoder(nn.Module):
    def __init__(self):
        super(cplxdecoder, self).__init__()

        self.convr = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1))
        self.scconvr = ScConv(16)
        self.scconvi = ScConv(16)
        self.convi = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1))


    def forward(self, x):
        real, imag = torch.chunk(x, chunks=2, dim=1)
        real1 = self.scconvr(real) - self.scconvi(imag)
        imag1 = self.scconvr(imag) + self.scconvi(real)

        real = self.convr(real1)
        imag = self.convi(imag1)
        x = torch.cat([real, imag], dim=1)
        return x

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=2,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        input_condition=False,

    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + channels * \
            (1 if self_condition else 0) + channels * \
            (1 if condition else 0) + channels * (1 if input_condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.cplxdecoder = cplxdecoder()
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)



    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        # 将复数重组为实数
        x = torch.cat((x[:, [0], :, :].real, x[:, [0], :, :].imag, x[:, [1], :, :].real, x[:, [1], :, :].imag), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []


        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            h_ = h.pop()
            x = torch.cat((x, h_), dim=1)
            x = block1(x, t)

            h_ = h.pop()

            x = torch.cat((x, h_), dim=1)
            x = block2(x, t)

            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)  # 最后进行残差连接
        x = self.final_res_block(x, t)
        # x = self.cplxdecoder(x)
        x = self.final_conv(x)

        # 转变回复数
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        x = torch.view_as_complex(x)[:, None, :, :]

        return x


class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=2,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=True,
        input_condition=False,
        objective='pred_res',
        test_res_or_noise="res"
    ):
        super().__init__()
        self.condition = condition
        self.input_condition = input_condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions
        self.unet0 = Unet(dim,
                          init_dim=init_dim,
                          out_dim=out_dim,
                          dim_mults=dim_mults,
                          channels=channels,
                          self_condition=self_condition,
                          resnet_block_groups=resnet_block_groups,
                          learned_variance=learned_variance,
                          learned_sinusoidal_cond=learned_sinusoidal_cond,
                          random_fourier_features=random_fourier_features,
                          learned_sinusoidal_dim=learned_sinusoidal_dim,
                          condition=condition,
                          input_condition=input_condition)

    def forward(self, x, time, x_self_cond=None):

        if self.objective == 'pred_res_noise':
            # num_unet=2
            pass
        elif self.objective == 'pred_x0_noise':
            # num_unet=2
            pass
        elif self.objective == "pred_noise":
            time = time[1]
        elif self.objective == "pred_res":
            time = time[0]
        return [self.unet0(x, time, x_self_cond=x_self_cond)]


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 生成系数
def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def stratified_sampling(num_samples=5, num_strata=5, max_time=1000, device=None):
    """将时间轴分为num_strata层，每层均匀采样整数，范围0到max_time"""
    t = []
    samples_per_stratum = num_samples // num_strata
    remainder = num_samples % num_strata

    for i in range(num_strata):
         # 动态分配每层样本数（处理余数）
        if i < remainder:
            stratum_samples_num = samples_per_stratum + 1
        else:
            stratum_samples_num = samples_per_stratum

        # 计算当前层的实际时间范围
        low = i * (max_time // num_strata)
        high = (i + 1) * (max_time // num_strata)

        # 生成该层样本
        stratum_samples = torch.randint(low, high, (stratum_samples_num,), device=device)
        t.append(stratum_samples)
    t = torch.cat(t)
    return t  # 返回排序后的时间步

class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        spec_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='mse',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        input_condition=False,
        resume=False,
        use_mean_value=True,
        midpoint_time=399,
    ):
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.spec_size = spec_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.resume = resume
        # loss
        self.loss_func = HybridLoss()

        # 使用均值采样
        self.use_mean_value = use_mean_value
        self.midpoint_time = midpoint_time

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        convert_to_ddim=True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = 0
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = 0
        else:
            alphas = gen_coefficients(timesteps, schedule="decreased")
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def init(self):
        timesteps = 1000

        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = alphas[1]
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = betas2[1]

        else:
            alphas = gen_coefficients(timesteps, schedule="average", ratio=1)
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale, ratio=3)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(
                alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])
            betas2_cumsum_prev = F.pad(
                betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_self_cond=None, time_next=None):

        x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                      self.betas_cumsum[t]*self.num_timesteps],
                                  x_self_cond)  # 输出当前输入对应的res和noise


        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]

                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)  # 通过网络的输出res和noise得出x0
                # x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                # pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                # x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                # x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                # pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            # pred_res = maybe_clip(pred_res)
            # x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            # x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            # pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res

        # if time_next > 0:
        #     x_start = x_start + self.alphas_cumsum[t].view(x_start.shape[0], 1, 1, 1) * x_input

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_spec = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_spec, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            spec = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = spec
        else:
            spec = torch.randn(shape, device=device)

        x_start = None

        if not last:
            spec_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            spec, x_start = self.p_sample(
                x_input, spec, t, x_input_condition, self_cond)

            if not last:
                spec_list.append(spec)

        if self.condition:
            if not last:
                spec_list = [input_add_noise]+spec_list
            else:
                spec_list = [input_add_noise, spec]
            return unnormalize_to_zero_to_one(spec_list)
        else:
            if not last:
                spec_list = spec_list
            else:
                spec_list = [spec]
            return unnormalize_to_zero_to_one(spec_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            spec = x_input+math.sqrt(self.sum_scale) * \
                torch.randn_like(x_input, device=device)  # 给图像添加噪声
            input_add_noise = spec
        else:
            spec = torch.randn(shape, device=device)

        x_start = None
        # pred_x0 = []
        type = "use_pred_noise"
        last = False

        if not last:
            spec_list = []
            x_start_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, spec, time_cond, self_cond, time_next)  # 得到三个输出

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                spec = x_start
                if not last:
                    spec_list.append(spec)
                    x_start_list.append(x_start)

                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(spec)

            if type == "use_pred_noise":
                spec = spec - alpha*pred_res + sigma2.sqrt()*noise

            elif type == "use_x_start":
                spec = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*spec + \
                    (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                spec = spec - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                spec = spec - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise
            if not last:
                spec_list.append(spec)
                x_start_list.append(x_start)


        if self.condition:
            if not last:
                spec_list = [input_add_noise]+spec_list
            else:
                spec_list = [input_add_noise, spec]
            return spec_list, x_start_list
        else:
            if not last:
                spec_list = spec_list
            else:
                spec_list = [spec]
            return spec_list, x_start_list


    @torch.no_grad()
    def valid_ddim_sample(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            spec = x_input+math.sqrt(self.sum_scale) * \
                torch.randn_like(x_input, device=device)  # 给图像添加噪声
            input_add_noise = spec
        else:
            spec = torch.randn(shape, device=device)

        x_start = None
        # pred_x0 = []
        type = "use_pred_noise"
        last = False

        if not last:
            spec_list = []
            x_start_list = []

        for time, time_next in time_pairs:
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, spec, time_cond, self_cond, time_next)  # 得到三个输出

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                spec = x_start
                if not last:
                    spec_list.append(spec)
                    x_start_list.append(x_start)

                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(spec)

            if type == "use_pred_noise":
                spec = spec - alpha*pred_res + sigma2.sqrt()*noise

            elif type == "use_x_start":
                spec = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*spec + \
                    (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                spec = spec - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                spec = spec - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise
            if not last:
                spec_list.append(spec)
                x_start_list.append(x_start)


        if self.condition:
            if not last:
                spec_list = [input_add_noise]+spec_list
            else:
                spec_list = [input_add_noise, spec]
            return spec_list, x_start_list
        else:
            if not last:
                spec_list = spec_list
            else:
                spec_list = [spec]
            return spec_list, x_start_list


    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        spec_size, channels = self.spec_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, spec_size, spec_size)
        return sample_fn(x_input, size, last=last)

    @torch.no_grad()
    def valid_sample(self, x_input, batch_size=16, last=True):
        spec_size, channels = self.spec_size, self.channels
        if self.condition:
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, spec_size, spec_size)
        return self.valid_ddim_sample(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type =='mea':
            losses = err.abs()
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def p_losses(self, specs, t1, t2, noise=None):

        x_start = specs[0]
        x_input = specs[1]

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # noise sample
        x1 = self.q_sample(x_start, x_res, t1, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        # 准备将第一部分数据输入到模型中
        x_in_one = torch.cat((x1, x_input), dim=1)
        model_out_one = self.model(x_in_one, self.alphas_cumsum[t1]*self.num_timesteps,
                               x_self_cond)  # 预测噪声和残差的输入网络的时间不一样

        pred_res_one = model_out_one[0]


        # 准备将第二部分数据输入到模型中
        x2 = self.q_sample(x_start, x_res, t2, noise=noise)
        x_in_two = torch.cat((x2, x_input), dim=1)
        model_out_two = self.model(x_in_two, self.alphas_cumsum[t2]*self.num_timesteps, x_self_cond)  # 预测噪声和残差的输入网络的时间不一样

        pred_res_two = model_out_two[0]


        # calculate loss
        loss_list = []
        loss_list.append(self._loss(pred_res_one - x_res) + self._loss(pred_res_one - pred_res_two))
        return loss_list

    def resume_losses(self, specs, time_pairs, device, noise=None):
        x_start = specs[0]
        x_input = specs[1]

        # noise sample
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = x_input + math.sqrt(self.sum_scale) * noise  # 给图像添加噪声

        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        for time, time_next in time_pairs:
            time_cond = torch.full((b,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, x_t, time_cond, self_cond, time_next=time_next)  # 得到三个输出

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            pred_x_start = preds.pred_x_start

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = self.ddim_sampling_eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if self.ddim_sampling_eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(x_t)

            x_t = x_t - alpha * pred_res + sigma2.sqrt() * noise

        loss_list = []
        err = pred_x_start - x_start
        loss = self.loss_func(pred_x_start, x_start)
        # loss = self._loss(err)
        loss_list.append(loss)
        return loss_list



    def forward(self, spec, resume=False, *args, **kwargs):
        if isinstance(spec, list):
            b, c, h, w, device, spec_size, = * \
                spec[0].shape, spec[0].device, self.spec_size
        else:
            b, c, h, w, device, spec_size, = *spec.shape, spec.device, self.spec_size
        if self.resume:
            step = random.randint(2, 4)
            times = torch.linspace(-1, 999, step)
            times = list(reversed(times.int().tolist()))
            # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            time_pairs = list(zip(times[:-1], times[1:]))
            loss = self.resume_losses(spec, time_pairs, device=device, *args, **kwargs)
        else:

            t1 = torch.randint(0, self.num_timesteps//2, (b,), device=device).long()
            t2 = torch.randint(self.num_timesteps//2, self.num_timesteps, (b,), device=device).long()
            # t1 = stratified_sampling(b, b, max_time=self.num_timesteps, device=device)
            # t2 = stratified_sampling(b, b, max_time=self.num_timesteps, device=device)
            loss = self.p_losses(spec, t1=t1, t2=t2, *args, **kwargs)
        return loss

# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        Dataset,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        condition=True,
        sub_dir=False,
        crop_patch=False,
        num_unet=1,
        stage='train',
        lambda_x=0.5,
        resume=False
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.spec_size = diffusion_model.spec_size
        self.condition = condition
        self.num_unet = num_unet

        # Dataset
        self.Dataset = Dataset
        if stage == 'fit' or stage is None:
            # train_dataset
            self.Dataset.setup(stage="fit")
            self.dl = cycle(self.accelerator.prepare(
                DataLoader(self.Dataset.train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=0)))
            self.valid_set = self.Dataset.valid_set

        if stage == 'test' or stage is None:
            # test_dataset
            self.Dataset.setup(stage="test")
            self.test_set = self.Dataset.test_set


        # optimizer

        if self.num_unet == 1:
            # self.opt0 = RAdam(diffusion_model.parameters(),
            #                   lr=train_lr, weight_decay=0.0)
            self.opt0 = Adam(diffusion_model.parameters(),
                            lr=train_lr, betas=adam_betas, capturable=True)
        elif self.num_unet == 2:
            self.opt0 = RAdam(
                diffusion_model.model.unet0.parameters(), lr=train_lr, weight_decay=0.0)
            self.opt1 = RAdam(
                diffusion_model.model.unet1.parameters(), lr=train_lr, weight_decay=0.0)


        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        if self.num_unet == 1:
            self.model, self.opt0 = self.accelerator.prepare(
                self.model, self.opt0)
        elif self.num_unet == 2:
            self.model, self.opt0, self.opt1 = self.accelerator.prepare(
                self.model, self.opt0, self.opt1)
        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        if self.num_unet == 1:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        elif self.num_unet == 2:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'opt1': self.opt1.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')

        if path.exists():
            data = torch.load(
                str(path), map_location=self.device)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']

            if self.num_unet == 1:
                self.opt0.load_state_dict(data['opt0'])
                self.opt0.optimizer.param_groups[0]['capturable'] = True

            elif self.num_unet == 2:
                self.opt0.load_state_dict(data['opt0'])
                self.opt1.load_state_dict(data['opt1'])
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))
        else:
            print("未找到已保存的模型权重，开始新的训练..")
        # self.ema.to(self.device)

    def train(self, results_folder):
        accelerator = self.accelerator

        logging.basicConfig(filename=os.path.join(results_folder, 'training.log'), level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
        epoch_loss = [0]
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                if self.num_unet == 1:
                    total_loss = [0]
                elif self.num_unet == 2:
                    total_loss = [0, 0]
                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        for i in range(self.num_unet):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            total_loss[i] = total_loss[i] + loss[i].item()
                            epoch_loss[i] = epoch_loss[i] + loss[i].item()

                    for i in range(self.num_unet):
                        self.accelerator.backward(loss[i])

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)  # 进行梯度裁剪

                accelerator.wait_for_everyone()

                self.opt0.step()
                self.opt0.zero_grad()


                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every

                        if self.step != 0 and self.step % (self.save_and_sample_every * 10) == 0:
                            self.save(milestone)
                            results_folder = self.results_folder
                            self.set_results_folder(results_folder)
                            self.evaluate_model(milestone)

                pbar.set_description(f'loss_unet: {total_loss[0]:.4f}')
                    # logging.info(f'Step [{self.step + 1}/{self.train_num_steps}],Step_Loss: {total_loss[0]:.4f}')

                if (self.step + 1) % (self.save_and_sample_every * 10) == 0:  # 每隔一定步数计算并保存平均损失
                    avg_loss = epoch_loss[0] / (self.save_and_sample_every * 10)
                    logging.info(f'Step [{self.step + 1}/{self.train_num_steps}],Average Loss: {avg_loss:.4f}')
                    epoch_loss = [0]

                pbar.update(1)

        accelerator.print('training complete')

    def evaluate_model(self, milestone, sr=16000, last=True):
        self.ema.to(self.device)
        self.ema.ema_model.eval()

        print("valid start")
        clean_files = self.valid_set.clean_files
        noisy_files = self.valid_set.noisy_files

        # Select test files uniformly accros validation files
        total_num_files = len(clean_files)
        indices = torch.arange(total_num_files)
        clean_files = list(clean_files[i] for i in indices)
        noisy_files = list(noisy_files[i] for i in indices)

        _pesq = 0
        _si_sdr = 0
        _estoi = 0

        # 添加进度条

        for idx in tqdm(range(total_num_files), position=0, desc="Processing files"):
            # Load wavs
            try:
                clean_path = clean_files[idx]
                noisy_path = noisy_files[idx]

                x, sr_x = load(clean_path)
                y, sr_y = load(noisy_path)
                T_orig = x.size(-1)

                # 确保采样率正确
                if sr_x != sr or sr_y != sr:
                    raise ValueError(f"Sample rate mismatch: clean {sr_x}, noisy {sr_y}")

                # Normalize per utterance
                norm_factor = y.abs().max()
                y = y / norm_factor
                y = y.to(self.device)

                Y = torch.unsqueeze(self.Dataset.spec_fwd(self.Dataset.stft(y.cuda())), 0)
                Y = pad_spec(Y)
                batches = self.num_samples
                all_spec_list, all_start_list = list(self.ema.ema_model.valid_sample([Y], batch_size=batches, last=last))
                x_hat = all_spec_list[-1]

                # Backward transform in time domain
                x_hat = self.to_audio(x_hat.squeeze(), length=T_orig) * norm_factor
                x_hat = x_hat.cpu().squeeze().numpy()
                x = x.cpu().squeeze().numpy()

                # 确保音频长度对齐
                min_len = min(x.shape[-1], x_hat.shape[-1])
                x = x[..., :min_len]
                x_hat = x_hat[..., :min_len]

                # 将指标进行加和
                _si_sdr += si_sdr(x, x_hat)
                _pesq += pesq(sr, x, x_hat, 'wb')
                _estoi += stoi(x, x_hat, sr, extended=True)

            except Exception as e:
                print(f"Error processing {clean_files[idx]}: {str(e)}")
                total_num_files -= 1  # 调整有效文件计数

        PESQ = _pesq / total_num_files
        ESTOI = _estoi / total_num_files
        SI_SDR = _si_sdr / total_num_files

        if total_num_files > 0:
            logging.info(f'milestone:{milestone}, PESQ: {PESQ:.4f}, ESTOI: {ESTOI:.4f}, SI_SDR: {SI_SDR:.4f}')
        else:
            logging.warning("No valid files processed")

        print("valid end")



    def to_audio(self, spec, length=None):
        spec = self.Dataset.spec_back(spec)
        audio = self.Dataset.istft(spec, length)
        return audio

    def test(self, sample=False, last=True, gamma=0.05):
        self.ema.ema_model.init()
        self.ema.to(self.device)

        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(dataset=self.test_set, batch_size=1)
            i = 0
            for items in loader:
                # load other
                file_name, norm_factor, T_orig = self.test_set.load_other(i)
                i += 1

                with torch.no_grad():
                    batches = self.num_samples

                    # load data
                    x_input_sample = [item.to(self.device)
                                          for item in items]
                    show_x_input_sample = x_input_sample
                    Y = pad_spec(x_input_sample[1])
                    x_input_sample = [Y]


                    if sample:
                        all_images_list, all_start_list= show_x_input_sample + \
                            list(self.ema.ema_model.sample(
                                x_input_sample, batch_size=batches))
                    else:
                        all_images_list, all_start_list = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last))
                        all_images_list = [all_start_list[-1]]

                all_images = torch.cat(all_images_list, dim=0)
                # Backward transform in time domain
                x_hat = self.to_audio(all_images.squeeze(), length=T_orig)

                #Renormalize
                x_hat = x_hat * norm_factor

                #Write enhanced wav file
                write(join(self.results_folder, file_name), x_hat.cpu().numpy(), 16000)
                print("test-save "+file_name)

        print("test end")



    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)


if __name__ == '__main__':

    from thop import profile,clever_format

    condition = True
    input_condition = False
    input_condition_mask = False

    # 随机生成输入数据
    x = torch.randn(1, 2, 256, 256)
    complex_x = torch.complex(x, torch.randn_like(x))
    time_cond = torch.randint(0, 100, (1,))

    # 初始化模型
    model = Unet(dim=32, dim_mults=(1, 1, 1, 1), condition=True)

    # 计算模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # 前向传播计算量
    macs, params = profile(model, inputs=(complex_x, time_cond, ), verbose=False)
    print(f"Total FLOPs: {macs / 1e9} GFLOPs")


