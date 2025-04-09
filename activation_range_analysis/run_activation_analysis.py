# q_diffusion_analyzer.py
import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, OrderedDict
import re

def load_model(model_path):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    return state_dict

def extract_layer_number(key):
    """Extract meaningful layer numbers with consistent spacing of 5."""
    # Extract layer type and position
    if 'down' in key:
        match = re.search(r'down\.(\d+)', key)
        if match:
            # Down blocks start at layer 5 and increment by 5
            return 5 + int(match.group(1)) * 5
    elif 'mid' in key:
        # Mid blocks come after down blocks
        match = re.search(r'mid\.block\.(\d+)', key)
        if match:
            return 40 + int(match.group(1)) * 5
    elif 'up' in key:
        match = re.search(r'up\.(\d+)', key)
        if match:
            # Up blocks come after mid blocks
            return 50 + int(match.group(1)) * 5
    elif 'input' in key or 'conv_in' in key:
        return 1  # Input layer
    elif 'output' in key or 'conv_out' in key:
        return 100  # Output layer
    
    # Additional internal structure
    match_block = re.search(r'block\.(\d+)', key)
    if match_block:
        block_num = int(match_block.group(1))
        # Determine base layer from context
        if 'down' in key:
            base = 5 + int(key.split('.')[1]) * 5
        elif 'mid' in key:
            base = 40
        elif 'up' in key:
            base = 50 + int(key.split('.')[1]) * 5
        else:
            base = 75
        # Add small offset for sub-blocks
        return base + block_num
    
    # Handle attention layers similarly
    match_attn = re.search(r'attn\.(\d+)', key)
    if match_attn:
        attn_num = int(match_attn.group(1))
        # Determine base layer from context
        if 'down' in key:
            base = 5 + int(key.split('.')[1]) * 5
        elif 'mid' in key:
            base = 40
        elif 'up' in key:
            base = 50 + int(key.split('.')[1]) * 5
        else:
            base = 75
        # Add small offset for attention blocks
        return base + attn_num + 3  # Offset from regular blocks
    
    # Default numbering for other layers
    if any(x in key for x in ['norm', 'temb']):
        return 2  # Early layers
    
    return 95  # Default for unmatched layers

def safe_sample_tensor(tensor, max_samples=10000):
    """Safely sample a tensor to avoid memory issues."""
    if tensor.numel() <= max_samples:
        return tensor.flatten().tolist()
    else:
        # Random sampling for large tensors
        indices = torch.randperm(tensor.numel())[:max_samples]
        return tensor.flatten()[indices].tolist()

def analyze_model(state_dict):
    """Analyze the model and extract activation statistics by layer."""
    # Group parameters by layer number
    layers = defaultdict(list)
    
    # First pass: assign layer numbers
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dim() == 0:
            continue
            
        layer_num = extract_layer_number(key)
        if layer_num > 0:  # Valid layer number
            # Store the key and its range
            min_val = float(tensor.min())
            max_val = float(tensor.max())
            range_val = max_val - min_val
            
            # Store a small sample for distribution
            samples = safe_sample_tensor(tensor)
            
            layers[layer_num].append({
                'key': key,
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'samples': samples
            })
    
    # Aggregate results by layer
    layer_stats = {}
    for layer_num, params in layers.items():
        if not params:
            continue
            
        # Collect all samples for this layer
        all_samples = []
        for p in params:
            all_samples.extend(p['samples'])
            
        # Use only a subset if there are too many
        if len(all_samples) > 50000:
            all_samples = np.random.choice(all_samples, 50000, replace=False).tolist()
            
        layer_stats[layer_num] = {
            'samples': all_samples,
            'range': max([p['max'] for p in params]) - min([p['min'] for p in params])
        }
    
    return layer_stats

def plot_q_diffusion_style(layer_stats, model_name, output_path):
    """Create a plot similar to the Q-Diffusion paper."""
    # Sort layers by number
    layer_numbers = sorted(layer_stats.keys())
    
    # Create the figure with specific dimensions
    plt.figure(figsize=(15, 4))
    
    # Prepare data for boxplot
    data = [layer_stats[ln]['samples'] for ln in layer_numbers]
    
    # Create boxplot
    boxplot = plt.boxplot(data, 
                         patch_artist=True,
                         showfliers=True,  # Show outliers
                         flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.5})
    
    # Customize appearance
    for box in boxplot['boxes']:
        box.set(color='blue', linewidth=1)
        box.set(facecolor='lightblue')
    for whisker in boxplot['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in boxplot['caps']:
        cap.set(color='black', linewidth=1)
    for median in boxplot['medians']:
        median.set(color='red', linewidth=1)
    for flier in boxplot['fliers']:
        flier.set(color='darkgrey', marker='o', markersize=2)
    
    # Add labels and title
    plt.xlabel('Layer number')
    plt.ylabel('Activation range')
    plt.title(model_name)
    
    # Set x-axis ticks
    plt.xticks(range(1, len(layer_numbers) + 1), layer_numbers)
    
    # Add dotted horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='dotted', alpha=0.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save and close
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Q-Diffusion style activation range analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="LDM", help="Model name for the plot")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and analyze model
    state_dict = load_model(args.model_path)
    layer_stats = analyze_model(state_dict)
    
    # Save layer statistics
    stats_path = os.path.join(args.output_dir, f"{args.model_name}_layer_stats.txt")
    with open(stats_path, 'w') as f:
        for layer_num in sorted(layer_stats.keys()):
            f.write(f"Layer {layer_num}:\n")
            f.write(f"  Range: {layer_stats[layer_num]['range']}\n")
            f.write(f"  Sample count: {len(layer_stats[layer_num]['samples'])}\n")
            f.write("\n")
    
    # Create visualization
    plot_path = os.path.join(args.output_dir, f"{args.model_name}_q_diffusion_style.png")
    plot_q_diffusion_style(layer_stats, args.model_name, plot_path)

if __name__ == "__main__":
    main()