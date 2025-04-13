import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.quant_util import QConv2d
from .self_attention import EnhancedQSelfAttention


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
            self.attn = EnhancedQSelfAttention(out_channels, quantization=quantization, 
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
        # Check spatial dimensions before max pooling
        if x.size(-1) <= 1 or x.size(-2) <= 1:
            # Skip max pooling if spatial dimensions are already 1x1
            x = self.res1(x)
            if self.time_mlp is not None and time_emb is not None:
                time_emb = self.time_mlp(time_emb)
                x = x + time_emb
            x = self.res2(x)
            x = self.attn(x)
            return x
        else:
            # Apply max pooling for normal dimensions
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
            self.attn = EnhancedQSelfAttention(out_channels, quantization=quantization, 
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
        
        # Resize spatial dimensions if needed
        if x.shape[2:] != skip_x.shape[2:]:
            x = F.interpolate(x, size=skip_x.shape[2:], mode='nearest')
        
        # Add projection layer if channels don't match expectations
        expected_channels = self.res1.in_channels
        actual_channels = x.shape[1] + skip_x.shape[1]
        
        if actual_channels != expected_channels:
            # Project channels to match expected input
            combined = torch.cat([x, skip_x], dim=1)
            if not hasattr(self, 'channel_proj'):
                self.channel_proj = torch.nn.Conv2d(
                    actual_channels, expected_channels, 
                    kernel_size=1, stride=1, padding=0).to(x.device)
            x = self.channel_proj(combined)
        else:
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
        
        # Add default values if not present in config
        if not hasattr(config.model, 'time_embed_dim'):
            config.model.time_embed_dim = 256  # Set a reasonable default value
        
        if not hasattr(config.model, 'attention_resolutions'):
            config.model.attention_resolutions = 1  # Default value
        
        time_embed_dim = config.model.time_embed_dim
        
        # Time embedding
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
        self.middle_attn = EnhancedQSelfAttention(now_ch, quantization=quantization, sequence=sequence, args=args)
        self.middle_block2 = ResidualBlock(now_ch, now_ch, dropout=config.model.dropout,
                                          quantization=quantization, sequence=sequence, args=args)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            # Calculate the correct input channels for each upblock
            for j in range(config.model.num_res_blocks + 1):
                # The first block in each resolution needs to handle skip connection
                if j == 0:
                    # Input is now_ch + matching skip connection channels
                    skip_ch = ch * mult  # This should match the skip connection
                    self.up_blocks.append(
                        UpBlock(now_ch + skip_ch, out_ch, time_emb_dim=time_embed_dim * 4, 
                               dropout=config.model.dropout, quantization=quantization, 
                               sequence=sequence, args=args,
                               use_attention=(i >= config.model.attention_resolutions))
                    )
                else:
                    self.up_blocks.append(
                        UpBlock(now_ch, out_ch, time_emb_dim=time_embed_dim * 4,
                               dropout=config.model.dropout, quantization=quantization,
                               sequence=sequence, args=args,
                               use_attention=(i >= config.model.attention_resolutions))
                    )
                now_ch = out_ch
        
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
        for i, layer in enumerate(self.up_blocks):
            # Check if we've run out of skip connections but still have upsampling to do
            if len(skip_connections) == 0:
                # Create a suitable zero tensor for skip connection
                skip_h = torch.zeros_like(h)
                h = layer(h, skip_h, t_emb)
            else:
                h = layer(h, skip_connections.pop(), t_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h

    def apply_arch_weights(self, arch_weights):
        """Apply architecture weights to different components during forward pass"""
        self.current_arch_weights = arch_weights
    
    def forward_with_weights(self, x, t):
        """Modified forward pass that applies architecture weights"""
        # Original timestep embedding
        t_emb = get_timestep_embedding(t, self.config.model.time_embed_dim)
        t_emb = t_emb * self.current_arch_weights['timestep_embed']
        
        h = x
        for i, block in enumerate(self.down_blocks):
            # Apply weights to each resblock
            weight = self.current_arch_weights['resblocks'][i % len(self.current_arch_weights['resblocks'])]
            h = h + weight * block(h, t_emb)
        
        # Apply weights to attention blocks
        for i, attn in enumerate(self.middle_attn):
            weight = self.current_arch_weights['attention'][i % len(self.current_arch_weights['attention'])]
            h = h + weight * attn(h)
        
        return h
