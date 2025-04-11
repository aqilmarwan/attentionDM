import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedPrecisionAttention(nn.Module):
    """
    Implements specialized quantization for self-attention computation in diffusion models.
    Supports timestep-dependent precision adjustment and optimized softmax computation.
    """
    def __init__(self, head_dim, num_heads, bit_width, scaling_factor=None):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.base_bit_width = bit_width
        self.scaling_factor = scaling_factor or (head_dim ** -0.5)
        
        # Quantization parameters for different stages of attention
        self.register_buffer('quant_scale_qk', torch.ones(1))
        self.register_buffer('quant_zero_qk', torch.zeros(1))
        self.register_buffer('quant_scale_attn', torch.ones(1))
        self.register_buffer('quant_zero_attn', torch.zeros(1))
        
        # Timestep-dependent bit allocation (can be learned)
        self.timestep_importance = nn.Parameter(torch.zeros(1000))
        self.timestep_importance.data.fill_(0.5)  # Initialize to middle value
        
        # Softmax stabilization parameters
        self.softmax_scale = nn.Parameter(torch.ones(1))
        
    def quantize_tensor(self, x, scale, zero_point, bits):
        """Apply quantization to a tensor with given parameters"""
        qmin, qmax = 0, (1 << bits) - 1
        scale = scale.to(x.device)
        zero_point = zero_point.to(x.device)
        
        x_q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq
    
    def get_effective_bits(self, timestep=None):
        """Determine effective bit-width based on timestep importance"""
        if timestep is None:
            return self.base_bit_width
            
        # Adjust bit-width based on timestep importance (higher for early steps)
        importance = self.timestep_importance[timestep]
        # Add up to 2 bits more precision for important timesteps
        bit_adjustment = 2.0 * torch.sigmoid(importance)
        return self.base_bit_width + bit_adjustment
    
    def forward(self, query, key, value, timestep=None):
        """
        Compute attention with customized quantization.
        
        Args:
            query: Query tensor [B, HW, C]
            key: Key tensor [B, C, HW]
            value: Value tensor [B, HW, C]
            timestep: Current diffusion timestep for adaptive precision
            
        Returns:
            Attention output tensor
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(2)
        
        # Reshape for multi-head attention
        q = query.reshape(batch_size, seq_len_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = key.reshape(batch_size, -1, self.num_heads, seq_len_k).permute(0, 2, 3, 1)
        v = value.reshape(batch_size, seq_len_q, self.num_heads, -1).permute(0, 2, 1, 3)
        
        # Get effective bit width for this timestep
        effective_bits = self.get_effective_bits(timestep)
        
        # Quantize QK computation with effective bits
        attn_weights = torch.matmul(q, k)
        attn_weights = attn_weights * self.scaling_factor
        
        # Optional quantization of attention weights for very low bit-widths
        if effective_bits <= 6:
            attn_weights = self.quantize_tensor(
                attn_weights, 
                self.quant_scale_qk, 
                self.quant_zero_qk, 
                max(4, int(effective_bits))
            )
        
        # Apply softmax with stability enhancements
        attn_weights = F.softmax(attn_weights * self.softmax_scale, dim=-1)
        
        # Optional quantization of attention probabilities
        if effective_bits <= 4:
            attn_weights = self.quantize_tensor(
                attn_weights, 
                self.quant_scale_attn, 
                self.quant_zero_attn, 
                max(3, int(effective_bits - 1))
            )
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape output to original dimensions
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len_q, -1)
        
        return output
    
    def update_quantization_params(self, qk_min, qk_max, attn_min, attn_max):
        """Update quantization parameters based on observed ranges"""
        # Set scale and zero point for QK computation
        qk_range = qk_max - qk_min
        self.quant_scale_qk = qk_range / (2**self.base_bit_width - 1)
        self.quant_zero_qk = -qk_min / self.quant_scale_qk
        
        # Set scale and zero point for attention probabilities (always in [0,1])
        self.quant_scale_attn = 1.0 / (2**self.base_bit_width - 1)
        self.quant_zero_attn = 0.0


class AttentionCalibrator:
    """Helper class for calibrating attention module quantization parameters"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attention_modules = []
        
        # Find all attention modules in the model
        for module in model.modules():
            if hasattr(module, 'attention_processor') and hasattr(module, 'mixed_precision'):
                if module.mixed_precision and module.quantization:
                    self.attention_modules.append(module)
    
    def calibrate(self, sample_batch, timesteps=None):
        """Calibrate attention modules using sample inputs"""
        if not self.attention_modules:
            print("No mixed-precision attention modules found to calibrate")
            return
            
        # Default timesteps to calibrate if not provided
        if timesteps is None:
            timesteps = [0, 250, 500, 750, 999]  # Sample across diffusion process
        
        # Collect ranges for each timestep
        for t in timesteps:
            t_tensor = torch.tensor([t], device=self.device).repeat(sample_batch.size(0))
            
            # Forward pass with hooks to capture min/max values
            with torch.no_grad():
                qk_mins, qk_maxs = [], []
                attn_mins, attn_maxs = [], []
                
                def qk_hook(module, input, output):
                    qk_mins.append(output.min().item())
                    qk_maxs.append(output.max().item())
                
                def attn_hook(module, input, output):
                    attn_mins.append(output.min().item())
                    attn_maxs.append(output.max().item())
                
                # Register hooks
                hooks = []
                for module in self.attention_modules:
                    # Need to instrument the actual attention computation
                    hooks.append(module.register_forward_hook(qk_hook))
                
                # Run model forward pass
                _ = self.model(sample_batch, t=t_tensor)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Update quantization parameters with collected statistics
                for i, module in enumerate(self.attention_modules):
                    if hasattr(module, 'attention_processor') and hasattr(module.attention_processor, 'update_quantization_params'):
                        module.attention_processor.update_quantization_params(
                            min(qk_mins), max(qk_maxs), 
                            0.0, 1.0  # Attention weights always in [0,1] after softmax
                        )
        
        print(f"Calibrated {len(self.attention_modules)} attention modules")
