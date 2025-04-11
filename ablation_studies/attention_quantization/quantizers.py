import torch
import torch.nn as nn
import numpy as np

class Quantizer(nn.Module):
    """Base quantizer class"""
    def __init__(self, bit_width=8, symmetric=True, per_channel=False):
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.n_levels = 2**bit_width
        self.scale = None
        self.zero_point = None
        
    def _calculate_qparams(self, x):
        """Calculate quantization parameters"""
        if self.per_channel:
            axis = 0 if x.dim() == 2 else 1
            x_min = x.min(dim=-1, keepdim=True)[0]
            x_max = x.max(dim=-1, keepdim=True)[0]
        else:
            x_min = x.min()
            x_max = x.max()
            
        if self.symmetric:
            abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
            x_min = -abs_max
            x_max = abs_max
            
        scale = (x_max - x_min) / (self.n_levels - 1)
        scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
        
        if self.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = torch.round(-x_min / scale)
            
        return scale, zero_point
    
    def forward(self, x):
        if self.bit_width == 32:  # No quantization
            return x
            
        if self.training:
            self.scale, self.zero_point = self._calculate_qparams(x)
            
        # Quantize
        x_q = torch.round(x / self.scale + self.zero_point)
        
        # Clamp
        x_q = torch.clamp(x_q, 0, self.n_levels - 1)
        
        # Dequantize
        x_dq = (x_q - self.zero_point) * self.scale
        
        return x_dq
    
    def calibrate(self, x):
        """Calculate and set scale and zero_point based on calibration data"""
        self.scale, self.zero_point = self._calculate_qparams(x)
        return self

class PTQQuantizer(Quantizer):
    """Post-training quantization implementation"""
    def __init__(self, bit_width=8, symmetric=True, per_channel=False, percentile=99.99):
        super().__init__(bit_width, symmetric, per_channel)
        self.percentile = percentile
        
    def _calculate_qparams(self, x):
        """Use percentile for outlier removal"""
        if self.per_channel:
            axis = list(range(1, x.dim()))
            x_flattened = x.transpose(0, 1).flatten(1)
            x_min = torch.tensor([torch.quantile(col, 1-self.percentile/100) for col in x_flattened]).view(-1, 1)
            x_max = torch.tensor([torch.quantile(col, self.percentile/100) for col in x_flattened]).view(-1, 1)
        else:
            x_min = torch.quantile(x.flatten(), 1-self.percentile/100)
            x_max = torch.quantile(x.flatten(), self.percentile/100)
            
        if self.symmetric:
            abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
            x_min = -abs_max
            x_max = abs_max
            
        scale = (x_max - x_min) / (self.n_levels - 1)
        scale = torch.clamp(scale, min=1e-8)
        
        if self.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = torch.round(-x_min / scale)
            
        return scale, zero_point