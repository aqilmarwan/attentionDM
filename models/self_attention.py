import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant_util import QConv2d, AsymmetricQuantFunction

class EnhancedQSelfAttention(nn.Module):
    """
    Enhanced self-attention module with specialized quantization support
    - Supports different bit-widths for query, key, value, and output projections
    - Implements softmax-aware quantization
    - Adds support for timestep-adaptive precision
    """
    def __init__(self, in_channels, quantization=False, sequence=None, args=None,
                 qkv_bit_config=None):
        super(EnhancedQSelfAttention, self).__init__()
        self.quantization = quantization
        self.in_channels = in_channels
        self.key_channels = in_channels // 8
        self.value_channels = in_channels
        self.heads = 8
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Default configuration if not specified
        if qkv_bit_config is None:
            if hasattr(args, 'bitwidth'):
                q_bit = k_bit = v_bit = o_bit = args.bitwidth
            else:
                q_bit = k_bit = v_bit = o_bit = 8
        else:
            q_bit, k_bit, v_bit, o_bit = qkv_bit_config
        
        if quantization and sequence is not None:
            # Quantized convolution layers with specialized bit-widths
            self.query_conv = QConv2d(in_channels, self.key_channels, kernel_size=1, 
                                     w_bit=q_bit, a_bit=q_bit, 
                                     sequence=sequence, args=args)
            self.key_conv = QConv2d(in_channels, self.key_channels, kernel_size=1, 
                                   w_bit=k_bit, a_bit=k_bit, 
                                   sequence=sequence, args=args)
            self.value_conv = QConv2d(in_channels, self.value_channels, kernel_size=1, 
                                     w_bit=v_bit, a_bit=v_bit, 
                                     sequence=sequence, args=args)
            self.output_conv = QConv2d(self.value_channels, in_channels, kernel_size=1, 
                                      w_bit=o_bit, a_bit=o_bit, 
                                      sequence=sequence, args=args)
        else:
            self.query_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels, self.value_channels, kernel_size=1)
            self.output_conv = nn.Conv2d(self.value_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
        # Timestep-specific precision adaptation parameters
        self.timestep_gates = nn.Parameter(torch.ones(len(sequence), 4) if sequence else torch.ones(1000, 4))
        
    def quantized_softmax(self, attention, bit_width=8):
        """
        Specialized softmax implementation with quantization awareness
        """
        attention_max = torch.max(attention, dim=-1, keepdim=True)[0]
        attention = attention - attention_max  # For numerical stability
        attention_exp = torch.exp(attention)
        
        # Quantize the exponentials
        if self.quantization and bit_width < 32:
            attention_min = attention_exp.min()
            attention_max = attention_exp.max()
            scale = (2**bit_width - 1) / (attention_max - attention_min)
            zero_point = scale * attention_min
            
            # Quantize
            attention_exp_q = torch.round(scale * attention_exp - zero_point)
            attention_exp_q = torch.clamp(attention_exp_q, 0, 2**bit_width - 1)
            # Dequantize
            attention_exp = (attention_exp_q + zero_point) / scale
        
        # Normalize
        attention_sum = attention_exp.sum(dim=-1, keepdim=True)
        return attention_exp / attention_sum
        
    def forward(self, x, t=None):
        batch_size, channels, height, width = x.size()
        
        # Determine bit-width based on timestep if available
        softmax_bit = 8  # Default softmax precision
        if t is not None and hasattr(self, 'timestep_gates'):
            # Get timestep index
            t_idx = t[0].long() if isinstance(t, torch.Tensor) else 0
            if t_idx < self.timestep_gates.shape[0]:
                # Adaptive precision based on timestep importance
                t_importance = F.softmax(self.timestep_gates[t_idx], dim=0)
                softmax_bit = max(4, min(8, int(4 + 4 * t_importance[0].item())))
        
        # Add timestep conditioning if available
        if t is not None:
            t = t.view(-1, 1, 1, 1).repeat(1, 1, height, width)
            x = torch.cat([x, t], dim=1)
        
        # Compute query, key, and value
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Reshape for attention computation
        query = query.view(batch_size, self.key_channels, -1).permute(0, 2, 1)  # B x HW x C
        key = key.view(batch_size, self.key_channels, -1)  # B x C x HW
        value = value.view(batch_size, self.value_channels, -1).permute(0, 2, 1)  # B x HW x C
        
        # Compute attention scores with scaling
        attention = torch.bmm(query, key)  # B x HW x HW
        attention = attention * (self.key_channels ** -0.5)  # Scale by sqrt(d_k)
        
        # Apply quantization-aware softmax
        attention = self.quantized_softmax(attention, bit_width=softmax_bit)
        
        # Apply attention to values
        out = torch.bmm(attention, value)  # B x HW x C
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.value_channels, height, width)
        
        # Apply output convolution
        out = self.output_conv(out)
        
        # Residual connection with learnable weight
        original_channels = min(channels, x.size(1))
        out = self.gamma * out + x[:, :original_channels]
        
        return out