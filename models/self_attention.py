import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant_util import QConv2d

class QSelfAttention(nn.Module):
    """
    Self-attention module with support for quantization
    """
    def __init__(self, in_channels, quantization=False, sequence=None, args=None):
        super(QSelfAttention, self).__init__()
        self.quantization = quantization
        self.in_channels = in_channels
        self.key_channels = in_channels // 8
        self.value_channels = in_channels
        self.heads = 8
        self.temperature = nn.Parameter(torch.ones(1))
        
        if quantization and sequence is not None:
            # Quantized convolution layers
            self.query_conv = QConv2d(in_channels, self.key_channels, kernel_size=1, 
                                     w_bit=args.bitwidth, a_bit=args.bitwidth, 
                                     sequence=sequence, args=args)
            self.key_conv = QConv2d(in_channels, self.key_channels, kernel_size=1, 
                                   w_bit=args.bitwidth, a_bit=args.bitwidth, 
                                   sequence=sequence, args=args)
            self.value_conv = QConv2d(in_channels, self.value_channels, kernel_size=1, 
                                     w_bit=args.bitwidth, a_bit=args.bitwidth, 
                                     sequence=sequence, args=args)
            self.output_conv = QConv2d(self.value_channels, in_channels, kernel_size=1, 
                                      w_bit=args.bitwidth, a_bit=args.bitwidth, 
                                      sequence=sequence, args=args)
        else:
            self.query_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels, self.value_channels, kernel_size=1)
            self.output_conv = nn.Conv2d(self.value_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, t=None):
        batch_size, channels, height, width = x.size()
        
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
        
        # Compute attention scores
        attention = torch.bmm(query, key)  # B x HW x HW
        attention = attention * (self.key_channels ** -0.5)  # Scale by sqrt(d_k)
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(attention, value)  # B x HW x C
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.value_channels, height, width)
        
        # Apply output convolution
        out = self.output_conv(out)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x[:, :channels] 
        
        return out