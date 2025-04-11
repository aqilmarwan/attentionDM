import torch
import torch.nn as nn
from .quantizers import Quantizer, PTQQuantizer

class QuantizedAttention(nn.Module):
    """Wrapper module that applies quantization to attention layers"""
    def __init__(self, attention_module, bit_width=8, qk_bit_width=None, v_bit_width=None, 
                 out_bit_width=None, symmetric=True, per_channel=False):
        super().__init__()
        self.attention_module = attention_module
        self.original_forward = attention_module.forward
        
        # Use default bit_width if specific ones aren't provided
        qk_bit_width = qk_bit_width or bit_width
        v_bit_width = v_bit_width or bit_width
        out_bit_width = out_bit_width or bit_width
        
        # Create quantizers for different parts of attention
        self.q_quantizer = PTQQuantizer(qk_bit_width, symmetric, per_channel)
        self.k_quantizer = PTQQuantizer(qk_bit_width, symmetric, per_channel)
        self.v_quantizer = PTQQuantizer(v_bit_width, symmetric, per_channel)
        self.out_quantizer = PTQQuantizer(out_bit_width, symmetric, per_channel)
        
        # Replace forward method
        attention_module.forward = self.quantized_forward
        
    def quantized_forward(self, *args, **kwargs):
        # Store original weights
        original_q_weight = None
        original_k_weight = None 
        original_v_weight = None
        original_out_weight = None
        
        attention = self.attention_module
        
        # Handle weights for cross-attention and self-attention differently
        if hasattr(attention, 'to_q'):  # Self-attention pattern
            original_q_weight = attention.to_q.weight.data.clone()
            original_k_weight = attention.to_k.weight.data.clone()
            original_v_weight = attention.to_v.weight.data.clone()
            original_out_weight = attention.to_out[0].weight.data.clone()
            
            # Quantize weights
            attention.to_q.weight.data = self.q_quantizer(attention.to_q.weight.data)
            attention.to_k.weight.data = self.k_quantizer(attention.to_k.weight.data)
            attention.to_v.weight.data = self.v_quantizer(attention.to_v.weight.data)
            attention.to_out[0].weight.data = self.out_quantizer(attention.to_out[0].weight.data)
            
        elif hasattr(attention, 'q_proj'):  # Cross-attention pattern 
            original_q_weight = attention.q_proj.weight.data.clone()
            original_k_weight = attention.k_proj.weight.data.clone()
            original_v_weight = attention.v_proj.weight.data.clone()
            original_out_weight = attention.out_proj.weight.data.clone()
            
            # Quantize weights
            attention.q_proj.weight.data = self.q_quantizer(attention.q_proj.weight.data)
            attention.k_proj.weight.data = self.k_quantizer(attention.k_proj.weight.data)
            attention.v_proj.weight.data = self.v_quantizer(attention.v_proj.weight.data)
            attention.out_proj.weight.data = self.out_quantizer(attention.out_proj.weight.data)
        
        # Call original forward
        result = self.original_forward(*args, **kwargs)
        
        # Restore original weights
        if hasattr(attention, 'to_q'):
            attention.to_q.weight.data = original_q_weight
            attention.to_k.weight.data = original_k_weight
            attention.to_v.weight.data = original_v_weight
            attention.to_out[0].weight.data = original_out_weight
        elif hasattr(attention, 'q_proj'):
            attention.q_proj.weight.data = original_q_weight
            attention.k_proj.weight.data = original_k_weight
            attention.v_proj.weight.data = original_v_weight
            attention.out_proj.weight.data = original_out_weight
            
        return result
        
    def calibrate(self, calibration_data):
        """Calibrate quantizers using representative data"""
        attention = self.attention_module
        
        # Process calibration data through each part of attention
        if hasattr(attention, 'to_q'):
            q_activations = []
            k_activations = []
            v_activations = []
            out_activations = []
            
            # Collect activations for calibration
            with torch.no_grad():
                for x in calibration_data:
                    q = attention.to_q(x)
                    k = attention.to_k(x)
                    v = attention.to_v(x)
                    q_activations.append(q)
                    k_activations.append(k)
                    v_activations.append(v)
                    
                    # Process through attention to get output activations
                    # This is simplified and may need adjustment based on actual implementation
                    out = attention.to_out[0](torch.bmm(q, k.transpose(-2, -1)) * v)
                    out_activations.append(out)
                    
            # Calibrate weights using collected activations
            q_tensor = torch.cat(q_activations)
            k_tensor = torch.cat(k_activations)
            v_tensor = torch.cat(v_activations)
            out_tensor = torch.cat(out_activations)
            
            self.q_quantizer.calibrate(q_tensor)
            self.k_quantizer.calibrate(k_tensor)
            self.v_quantizer.calibrate(v_tensor)
            self.out_quantizer.calibrate(out_tensor)
            
        # Similar process for cross-attention pattern would be implemented here
        
        return self