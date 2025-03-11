import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.quant_util import QConv2d
from .self_attention import QSelfAttention


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1, 
                 quantization=False, sequence=None, args=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        
        if quantization and sequence is not None:
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
            self.conv1 = QConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
            self.dropout = nn.Dropout(dropout)
            self.conv2 = QConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
            
            if self.in_channels != self.out_channels:
                if self.use_conv_shortcut:
                    self.conv_shortcut = QConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                               w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
                else:
                    self.nin_shortcut = QConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                              w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
        else:
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
            self.dropout = nn.Dropout(dropout)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            
            if self.in_channels != self.out_channels:
                if self.use_conv_shortcut:
                    self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                else:
                    self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.1,
                 quantization=False, sequence=None, args=None, use_attention=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.res1 = ResidualBlock(in_channels, out_channels, dropout=dropout,
                                 quantization=quantization, sequence=sequence, args=args)
        self.res2 = ResidualBlock(out_channels, out_channels, dropout=dropout,
                                 quantization=quantization, sequence=sequence, args=args)
        
        if use_attention:
            self.attn = QSelfAttention(out_channels, quantization=quantization, 
                                      sequence=sequence, args=args)
        else:
            self.attn = nn.Identity()
        
        if time_emb_dim is not None:
            if quantization and sequence is not None:
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    QConv2d(time_emb_dim, out_channels, kernel_size=1, stride=1, padding=0,
                          w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
                )
            else:
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(time_emb_dim, out_channels, kernel_size=1, stride=1, padding=0)
                )
        else:
            self.time_mlp = None
    
    def forward(self, x, time_emb=None):
        x = self.maxpool(x)
        x = self.res1(x)
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            x = x + time_emb
        x = self.res2(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.1,
                 quantization=False, sequence=None, args=None, use_attention=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.res1 = ResidualBlock(in_channels + out_channels, out_channels, dropout=dropout,
                                 quantization=quantization, sequence=sequence, args=args)
        self.res2 = ResidualBlock(out_channels, out_channels, dropout=dropout,
                                 quantization=quantization, sequence=sequence, args=args)
        
        if use_attention:
            self.attn = QSelfAttention(out_channels, quantization=quantization, 
                                      sequence=sequence, args=args)
        else:
            self.attn = nn.Identity()
        
        if time_emb_dim is not None:
            if quantization and sequence is not None:
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    QConv2d(time_emb_dim, out_channels, kernel_size=1, stride=1, padding=0,
                          w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
                )
            else:
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(time_emb_dim, out_channels, kernel_size=1, stride=1, padding=0)
                )
        else:
            self.time_mlp = None
    
    def forward(self, x, skip_x, time_emb=None):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.res1(x)
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            x = x + time_emb
        x = self.res2(x)
        x = self.attn(x)
        return x


class Model(nn.Module):
    def __init__(self, config, quantization=False, sequence=None, args=None):
        super().__init__()
        self.config = config
        self.quantization = quantization
        self.sequence = sequence
        self.args = args
        
        # Time embedding
        time_embed_dim = config.model.time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim * 4),
        )
        
        # Initial convolution
        ch = config.model.ch
        if quantization and sequence is not None:
            self.init_conv = QConv2d(config.data.channels, ch, kernel_size=3, stride=1, padding=1,
                                    w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
        else:
            self.init_conv = nn.Conv2d(config.data.channels, ch, kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch_mult = config.model.ch_mult
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(config.model.num_res_blocks):
                self.down_blocks.append(
                    DownBlock(now_ch, out_ch, time_emb_dim=time_embed_dim * 4, dropout=config.model.dropout,
                             quantization=quantization, sequence=sequence, args=args,
                             use_attention=(i >= config.model.attention_resolutions))
                )
                now_ch = out_ch
            if i < len(ch_mult) - 1:
                self.down_blocks.append(
                    DownBlock(now_ch, now_ch, time_emb_dim=time_embed_dim * 4, dropout=config.model.dropout,
                             quantization=quantization, sequence=sequence, args=args,
                             use_attention=False)
                )
        
        # Middle blocks
        self.middle_block1 = ResidualBlock(now_ch, now_ch, dropout=config.model.dropout,
                                          quantization=quantization, sequence=sequence, args=args)
        self.middle_attn = QSelfAttention(now_ch, quantization=quantization, sequence=sequence, args=args)
        self.middle_block2 = ResidualBlock(now_ch, now_ch, dropout=config.model.dropout,
                                          quantization=quantization, sequence=sequence, args=args)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(config.model.num_res_blocks + 1):
                self.up_blocks.append(
                    UpBlock(now_ch, out_ch, time_emb_dim=time_embed_dim * 4, dropout=config.model.dropout,
                           quantization=quantization, sequence=sequence, args=args,
                           use_attention=(i >= config.model.attention_resolutions))
                )
                now_ch = out_ch
            if i > 0:
                self.up_blocks.append(
                    UpBlock(now_ch, ch * ch_mult[i-1], time_emb_dim=time_embed_dim * 4, dropout=config.model.dropout,
                           quantization=quantization, sequence=sequence, args=args,
                           use_attention=False)
                )
                now_ch = ch * ch_mult[i-1]
        
        # Output layers
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=now_ch, eps=1e-6)
        if quantization and sequence is not None:
            self.conv_out = QConv2d(now_ch, config.data.channels, kernel_size=3, stride=1, padding=1,
                                   w_bit=args.bitwidth, a_bit=args.bitwidth, sequence=sequence, args=args)
        else:
            self.conv_out = nn.Conv2d(now_ch, config.data.channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = get_timestep_embedding(t, self.config.model.time_embed_dim)
        t_emb = self.time_embed(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Downsampling
        skip_connections = [h]
        for layer in self.down_blocks:
            h = layer(h, t_emb)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block1(h)
        h = self.middle_attn(h)
        h = self.middle_block2(h)
        
        # Upsampling
        for layer in self.up_blocks:
            h = layer(h, skip_connections.pop(), t_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
