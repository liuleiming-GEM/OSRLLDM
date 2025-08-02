from abc import abstractmethod
import torch
from torch.nn import functional as f

import math
import numpy as np
import torch as th
import torch.nn.functional as F
from .swin_transformer import BasicLayer
from einops import rearrange
from inspect import isfunction
from torch import nn, einsum
from models.LAnet_diffusion import LAnet_diff
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def posemb_sincos_2d(patches, temperature = 10000, dtype = th.float32):
    '''
    Borrowed from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    '''
    _, dim, h, w, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = th.meshgrid(th.arange(h, device = device), th.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = th.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = th.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype) # (n, hd)


class ZeroSFT(nn.Module):
    def __init__(self, control_channels, en_channels, de_channels):
        super().__init__()

        ks = 3
        pw = ks // 2

        self.param_free_norm = normalization(en_channels + de_channels)

        self.zero_conv = zero_module(conv_nd(2, control_channels, en_channels, 1, 1, 0))

        self.channel = nn.Sequential(
            nn.Conv2d(control_channels, de_channels, 1, 1, 0),
            nn.SiLU()
        )

        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(control_channels, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU()
        )
        self.zero_mul = zero_module(nn.Conv2d(nhidden, en_channels + de_channels, kernel_size=ks, padding=pw))
        self.zero_add = zero_module(nn.Conv2d(nhidden, en_channels + de_channels, kernel_size=ks, padding=pw))

    def forward(self, c, h, de=None):
        h = h + self.zero_conv(c)
        h = th.cat([de, h], dim=1)
        actv = self.mlp_shared(c)  # Xc生成归一化gamma和beta
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        h = self.param_free_norm(h) * (gamma + 1) + beta  # groupnorm
        return h


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, Mat, LR_size):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):  # 继承模块         self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
    """一个顺序模块，将时间步长嵌入传递给子级将其作为额外的输入进行支持。
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, Mat, LR_size):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, Mat, LR_size)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                      1, 1, 0, bias=True)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, Mat, LR_size):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # if x.shape[2] == LR_size:
        #     Mat = Mat
        # elif x.shape[2] == LR_size / 2:
        #     Mat = F.avg_pool2d(Mat, kernel_size=(2, 1), stride=(2, 1))
        # elif x.shape[2] == LR_size / 4:
        #     Mat = F.avg_pool2d(Mat, kernel_size=(4, 1), stride=(4, 1))
        # elif x.shape[2] == LR_size / 8:
        #     Mat = F.avg_pool2d(Mat, kernel_size=(8, 1), stride=(8, 1))

        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
        emb_out = self.emb_layers(emb).type(h.dtype)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class SpatialCrossAttentionWithPosEmb(nn.Module):
    '''
    Cross-attention block for image-like data.
    First image reshape to b, t, d.
    Perform self-attention if context is None, else cross-attention.
    The dims of the input and output of the block are the same (arg query_dim).
    '''

    def __init__(self, in_channels=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(inner_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Dropout(dropout)
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

        self.norm = nn.LayerNorm(inner_dim)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        context = default(context, x)
        x = self.proj_in(x)  # (b,d,h,w)
        context = self.proj_in(context)  # (b,d,h,w)

        # positional embedding
        pe = posemb_sincos_2d(x)  # (n,d)

        # re-arrange image data to b, n, d.
        x = rearrange(x, 'b c h w -> b (h w) c')
        if (len(context.shape) == 4):
            context = rearrange(context, 'b c h w -> b (h w) c')

        # add pos emb
        x += pe
        if context.shape[1] != x.shape[1]:
            context[:, :h * w] += pe
            context[:, h * w:] += pe
        else:
            context += pe

        heads = self.heads

        x = self.norm(x)
        context = self.norm(context)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        out = self.to_out(out)

        # restore image shape
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.proj_out(out)

        return x_in + out


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x ch
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            # q,k, v: (b*heads) x ch x length
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards     # (b*heads) x M x M
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v)  # (b*heads) x ch x length
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            # qkv: b x length x heads x 3ch
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).to(memory_format=th.contiguous_format)
            q, k, v = qkv.split(ch, dim=3)  # b x length x heads x ch
            a = xop.memory_efficient_attention(q, k, v, p=0.0)  # b x length x heads x length
            out = a.permute(0, 2, 3, 1).to(memory_format=th.contiguous_format).reshape(bs, -1, length)
        else:
            q, k, v = qkv.chunk(3, dim=1)  # b x heads*ch x length
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = th.einsum(
                "bct,bcs->bts",
                (q * scale).view(bs * self.n_heads, ch, length),
                (k * scale).view(bs * self.n_heads, ch, length),
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
            out = a.reshape(bs, -1, length)
        return out

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModelHAT(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :patch_norm: patch normalization in swin transformer
    :swin_embed_norm: embed_dim in swin transformer
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 2, 4),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        STB_depth=2,
        STB_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_mask=False,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert STB_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_mask = cond_mask
        self.LAnet_diff = LAnet_diff(
                 img_size=64,
                 patch_size=1,
                 in_chans=4,
                 out_chans=4,
                 embed_dim=192,
                 depths=(4, 4, 4),
                 num_heads=(4, 4, 4),
                 window_size=8,
                 mlp_ratio=4,
                 drop_rate=0.,
                 img_range=1.,
                 c_dim=192,
                 window_condition=True,
                 dcn_condition=[1, 1, 1, 1],
                 dcn_condition_type='2conv',
                 )

        time_embed_dim = model_channels * 2
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # self.mat = nn.Sequential(
        #     nn.Conv2d(1, channel_mult[3] * model_channels, 1, 1, 0, bias=True),
        #     nn.SiLU(),
        #     nn.Conv2d(channel_mult[3] * model_channels, channel_mult[3] * model_channels, 1, 1, 0, bias=True)
        # )

        self.feature_extractor = nn.Identity()
        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=STB_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else STB_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=STB_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                    in_chans=ch,
                    embed_dim=STB_embed_dim,
                    num_heads=num_heads if num_head_channels == -1 else STB_embed_dim // num_head_channels,
                    window_size=window_size,
                    depth=STB_depth,
                    img_size=ds,
                    patch_size=1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=dropout,
                    attn_drop=0.,
                    drop_path=0.,
                    use_checkpoint=False,
                    norm_layer=normalization,
                    patch_norm=patch_norm,
                     ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=STB_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else STB_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=STB_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

        self.project_modules = nn.ModuleList()
        control_channels = [320] * 3 + [320] * 6 + [160] * 3
        en_channels = [320] * 2 + [320] * 6 + [160] * 4
        de_channels = [320] * 4 + [320] * 6 + [160] * 2
        for i in range(len(en_channels)):
            self.project_modules.append(ZeroSFT(control_channels[i], en_channels[i], de_channels[i]))

        # self.layer_fussion = LAM_Module_v2(in_dim=32)

    def forward(self, x, timesteps, LR, Mat_64, z_BIC):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param LR: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)
        # Mat = self.mat(Mat_64)
        Mat = Mat_64[:, :, :, :1]
        if LR.shape[2] == 32:
            LR = F.interpolate(LR, scale_factor=2, mode='bicubic', align_corners=False)
        elif LR.shape[2] == 128:
            LR = F.pixel_unshuffle(LR, 2)
        LR_size = LR.shape[2]

        control64, control32, control16, control8, LAnet_out = self.LAnet_diff(z_BIC, Mat_64)
        LR = self.feature_extractor(LR.type(self.dtype))
        x = th.cat([x, z_BIC, LR, Mat_64], dim=1)  # 带噪声图像恶化lq图像一起

        h = x.type(self.dtype)
        hs.append(h)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb, Mat, LR_size)
            hs.append(h)

        h = self.middle_block(h, emb, Mat, LR_size)

        adapter_idx = 0

        for module in self.output_blocks:
            _h = hs.pop()
            if _h.shape[2] == LR_size:
                h = self.project_modules[adapter_idx](control64, _h, h)
                adapter_idx += 1
                h = module(h, emb, Mat, LR_size)
            elif _h.shape[2] == LR_size / 2:
                h = self.project_modules[adapter_idx](control32, _h, h)
                adapter_idx += 1
                h = module(h, emb, Mat, LR_size)
            elif _h.shape[2] == LR_size / 4:
                h = self.project_modules[adapter_idx](control16, _h, h)
                adapter_idx += 1
                h = module(h, emb, Mat, LR_size)
            elif _h.shape[2] == LR_size / 8:
                h = self.project_modules[adapter_idx](control8, _h, h)
                adapter_idx += 1
                h = module(h, emb, Mat, LR_size)

        h = h.type(x.dtype)
        out = self.out(h)
        return out, LAnet_out
