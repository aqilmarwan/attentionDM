import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant_util import QConv2d
from utils.attention_quant_util import MixedPrecisionAttention

class EnhancedQSelfAttention(nn.Module):
    """
    Enhanced self-attention module with specialized quantization strategies
    for different components (query, key, value, output)
    """
    def __init__(self, in_channels, quantization=False, sequence=None, args=None, 
                 mixed_precision=False, bit_config=None):
        super(EnhancedQSelfAttention, self).__init__()
        self.quantization = quantization
        self.in_channels = in_channels
        self.key_channels = in_channels // 8
        self.value_channels = in_channels
        self.heads = 8
        self.temperature = nn.Parameter(torch.ones(1))
        self.mixed_precision = mixed_precision
        
        # Default bit configuration if not provided
        if bit_config is None:
            self.bit_config = {
                "query": args.bitwidth if args else 8,  # Higher precision for query
                "key": max(4, args.bitwidth - 2) if args else 6,     # Lower precision for key
                "value": args.bitwidth if args else 8,  # Higher precision for value
                "output": args.bitwidth if args else 8  # Full precision for output
            }
        else:
            self.bit_config = bit_config
        
        if quantization and sequence is not None:
            # Differentiated bit-width for different projections
            self.query_conv = QConv2d(in_channels, self.key_channels, kernel_size=1, 
                                     w_bit=self.bit_config["query"], 
                                     a_bit=self.bit_config["query"], 
                                     sequence=sequence, args=args)
            self.key_conv = QConv2d(in_channels, self.key_channels, kernel_size=1, 
                                   w_bit=self.bit_config["key"], 
                                   a_bit=self.bit_config["key"], 
                                   sequence=sequence, args=args)
            self.value_conv = QConv2d(in_channels, self.value_channels, kernel_size=1, 
                                     w_bit=self.bit_config["value"], 
                                     a_bit=self.bit_config["value"], 
                                     sequence=sequence, args=args)
            self.output_conv = QConv2d(self.value_channels, in_channels, kernel_size=1, 
                                      w_bit=self.bit_config["output"], 
                                      a_bit=self.bit_config["output"], 
                                      sequence=sequence, args=args)
            
            # Configure specialized group quantization settings
            self.configure_group_quantization()
        else:
            self.query_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels, self.key_channels, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels, self.value_channels, kernel_size=1)
            self.output_conv = nn.Conv2d(self.value_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Use specialized softmax for quantization if specified
        if mixed_precision and quantization:
            self.attention_processor = MixedPrecisionAttention(
                head_dim=self.key_channels // self.heads,
                num_heads=self.heads,
                bit_width=min(self.bit_config.values()),
                scaling_factor=self.key_channels ** -0.5
            )
        else:
            self.softmax = nn.Softmax(dim=-1)
    
    def configure_group_quantization(self):
        """
        Configure specialized group sizes for attention projections.
        This aligns quantization groups with attention head structure for better precision.
        """
        # For query/key projections, align groups with head dimensions
        if hasattr(self, 'query_conv') and isinstance(self.query_conv, QConv2d):
            # Each query/key group corresponds to one attention head
            # This preserves intra-head relationships better during quantization
            self.query_conv.group_num = self.heads  
            self.key_conv.group_num = self.heads
            
            # For value tensors, we can use coarser quantization (fewer groups)
            # This reduces memory footprint while still preserving essential features
            self.value_conv.group_num = max(2, self.heads // 2)
            
            # Output projection can use standard grouping as it recombines all heads
            self.output_conv.group_num = 8  # Default group number
            
            # Update alpha_activ parameters to match new group dimensions
            # (necessary because nn.Parameter sizes must match)
            for conv, name in [(self.query_conv, 'query'), 
                              (self.key_conv, 'key'), 
                              (self.value_conv, 'value'), 
                              (self.output_conv, 'output')]:
                # Create new alpha_activ with proper dimensions
                old_alpha = conv.alpha_activ.data
                new_alpha = torch.zeros(
                    old_alpha.size(0),  # len_seq dimension stays the same
                    conv.group_num,     # new group_num 
                    conv.in_channels    # in_channels stays the same
                ).to(old_alpha.device)
                
                # Initialize with same values spread across new groups
                for i in range(old_alpha.size(0)):  # For each timestep
                    for g in range(conv.group_num):
                        # Distribute old values proportionally into new grouping
                        old_group_idx = min(g * old_alpha.size(1) // conv.group_num, 
                                          old_alpha.size(1) - 1)
                        new_alpha[i, g] = old_alpha[i, old_group_idx]
                
                # Replace parameter
                conv.alpha_activ = nn.Parameter(new_alpha)
    
    def forward(self, x, t=None, timestep=None):
        batch_size, channels, height, width = x.size()
        
        # Process timestep embedding if provided
        if t is not None:
            t = t.view(-1, 1, 1, 1).repeat(1, 1, height, width)
            x = torch.cat([x, t], dim=1)
        
        # Compute projections with potentially different precisions
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Reshape for attention computation
        query = query.view(batch_size, self.key_channels, -1).permute(0, 2, 1)  # B x HW x C
        key = key.view(batch_size, self.key_channels, -1)  # B x C x HW
        value = value.view(batch_size, self.value_channels, -1).permute(0, 2, 1)  # B x HW x C
        
        # Compute attention with specialized processing if mixed precision
        if self.mixed_precision and self.quantization:
            out = self.attention_processor(query, key, value, timestep)
        else:
            # Standard attention computation
            attention = torch.bmm(query, key)  # B x HW x HW
            attention = attention * (self.key_channels ** -0.5)  # Scale by sqrt(d_k)
            attention = self.softmax(attention)
            out = torch.bmm(attention, value)  # B x HW x C
        
        # Reshape and apply output convolution
        out = out.permute(0, 2, 1).contiguous().view(batch_size, self.value_channels, height, width)
        out = self.output_conv(out)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x[:, :channels]
        
        return out

def create_enhanced_attention(in_channels, sequence, args):
    return EnhancedQSelfAttention(
        in_channels,
        quantization=True,
        sequence=sequence,
        args=args,
        mixed_precision=True,
        bit_config={
            "query": args.bitwidth,  # Full precision for query projections
            "key": max(4, args.bitwidth - 2),  # Lower precision for keys
            "value": args.bitwidth,  # Full precision for values
            "output": args.bitwidth  # Full precision for output projection
        }
    )