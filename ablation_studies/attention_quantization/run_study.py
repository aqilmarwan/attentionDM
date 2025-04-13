import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add parent directory to path to access attentionDM modules
sys.path.append("/workspace/attentionDM")

# Import essential utilities
from utils import (
    find_attention_layers,
    apply_quantization,
    measure_inference_speed,
    prepare_calibration_data,
    calculate_simple_metrics
)

# Import necessary modules from attentionDM (adjust paths as needed)
from model.unet import UNet  # Update with correct path
from diffusion import GaussianDiffusion  # Update with correct path

def run_attention_quantization_study(args):
    """Run the attention layer quantization study with real model inference"""
    device = torch.device(args.device)
    
    # Load the full checkpoint - properly reconstruct the model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Reconstruct the model based on your attentionDM architecture
    # This is an example - adjust according to your model's actual structure
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        attention_resolutions=(1, 2, 4),
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),
        num_heads=8
    ).to(device)
    
    # Load state dict - adjust key naming if needed
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Try direct loading
    
    # Create diffusion model with the loaded UNet
    diffusion = GaussianDiffusion(
        model,
        image_size=32,  # CIFAR-10 size, adjust if needed
        timesteps=1000   # Adjust based on your model
    ).to(device)
    
    # Prepare calibration data
    calibration_data = prepare_calibration_data(args.calibration_data, args.num_calibration)
    
    # Define bit widths to test
    bit_widths = [2, 4, 6, 8, 16, 32]  # 32-bit means no quantization (baseline)
    
    results = {
        'bit_width': [],
        'mean_value': [],
        'std_value': [],
        'inference_time': [],
        'model_size_mb': []
    }
    
    # Run tests for each bit width
    for bit_width in tqdm(bit_widths, desc="Testing bit widths"):
        print(f"\nTesting {bit_width}-bit quantization:")
        
        # Create a copy of the model for this test
        model_copy = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=128,
            attention_resolutions=(1, 2, 4),
            num_res_blocks=2,
            channel_mult=(1, 2, 3, 4),
            num_heads=8
        ).to(device)
        model_copy.load_state_dict(model.state_dict())
        
        diffusion_copy = GaussianDiffusion(
            model_copy,
            image_size=32,  # CIFAR-10 size, adjust if needed
            timesteps=1000  # Adjust based on your model
        ).to(device)
        
        # Apply quantization
        if bit_width < 32:  # Skip quantization for the baseline
            quantized_modules = apply_quantization(
                model_copy, 
                bit_width=bit_width,
                symmetric=args.symmetric,
                per_channel=args.per_channel
            )
            
            # Calibrate quantizers if not in training mode
            if not args.training_aware:
                print("Calibrating quantizers...")
                for module in quantized_modules:
                    module.calibrate(calibration_data)
        
        # Generate real samples using the actual diffusion model
        print("Generating samples for evaluation...")
        generated_samples = []
        with torch.no_grad():
            for _ in tqdm(range(args.num_eval_samples)):
                # Use the actual sampling method from your diffusion model
                sample = diffusion_copy.sample(batch_size=1)
                generated_samples.append(sample)
        
        # Evaluate quality using simple metrics
        print("Evaluating quality...")
        metrics = calculate_simple_metrics(generated_samples)
        
        # Measure inference speed
        print("Measuring inference speed...")
        inference_time = measure_inference_speed(
            diffusion_copy.sample, 
            {"batch_size": 1}, 
            num_runs=args.speed_test_runs
        )
        
        # Estimate model size
        model_size_mb = sum(p.numel() * (bit_width/32 if bit_width < 32 else 1) * p.element_size() 
                           for p in model_copy.parameters()) / (1024**2)
        
        # Record results
        results['bit_width'].append(bit_width)
        results['mean_value'].append(metrics['mean_value'])
        results['std_value'].append(metrics['std_value'])
        results['inference_time'].append(inference_time)
        results['model_size_mb'].append(model_size_mb)
        
        print(f"Results for {bit_width}-bit:")
        print(f"  Mean pixel value: {metrics['mean_value']:.4f}")
        print(f"  Std deviation: {metrics['std_value']:.4f}")
        print(f"  Inference time: {inference_time*1000:.2f} ms")
        print(f"  Model size: {model_size_mb:.2f} MB")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, 'attention_quantization_results.csv'), index=False)
    
    # Create comprehensive visualization
    create_comprehensive_plot(results, args.output_dir)
    print(f"Results saved to {args.output_dir}")


def create_comprehensive_plot(results, output_dir):
    """Create a comprehensive visualization of quantization results"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    plt.figure(figsize=(15, 12))
    plt.style.use('ggplot')
    
    # Setup grid layout
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8])
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results['bit_width'])))
    
    # Plot 1: Mean Pixel Value
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(results['bit_width'], results['mean_value'], 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax1.set_title('Mean Pixel Value vs. Bit Width', fontsize=14)
    ax1.set_xlabel('Quantization Bit Width', fontsize=12)
    ax1.set_ylabel('Mean Value', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard Deviation
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(results['bit_width'], results['std_value'], 'o-', linewidth=2, markersize=8, color='#ff7f0e')
    ax2.set_title('Standard Deviation vs. Bit Width', fontsize=14)
    ax2.set_xlabel('Quantization Bit Width', fontsize=12)
    ax2.set_ylabel('Std Deviation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Inference Time
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(results['bit_width'], [t*1000 for t in results['inference_time']], 'o-', linewidth=2, markersize=8, color='#2ca02c')
    ax3.set_title('Inference Time vs. Bit Width', fontsize=14)
    ax3.set_xlabel('Quantization Bit Width', fontsize=12)
    ax3.set_ylabel('Inference Time (ms)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Size
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(results['bit_width'], results['model_size_mb'], 'o-', linewidth=2, markersize=8, color='#d62728')
    ax4.set_title('Model Size vs. Bit Width', fontsize=14)
    ax4.set_xlabel('Quantization Bit Width', fontsize=12)
    ax4.set_ylabel('Model Size (MB)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add visualizations of actual generated samples for each bit width
    # (This would require saving sample images during testing)
    
    # Bar chart comparing all metrics
    ax5 = plt.subplot(gs[2, :])
    
    # Normalize all metrics to the 32-bit baseline for comparison
    baseline_idx = results['bit_width'].index(max(results['bit_width']))
    rel_mean = [v/results['mean_value'][baseline_idx] for v in results['mean_value']]
    rel_std = [v/results['std_value'][baseline_idx] for v in results['std_value']]
    rel_time = [v/results['inference_time'][baseline_idx] for v in results['inference_time']]
    rel_size = [v/results['model_size_mb'][baseline_idx] for v in results['model_size_mb']]
    
    x = np.arange(len(results['bit_width']))
    width = 0.2
    
    bars1 = ax5.bar(x - width*1.5, rel_mean, width, label='Norm. Mean', color='#1f77b4')
    bars2 = ax5.bar(x - width/2, rel_std, width, label='Norm. Std Dev', color='#ff7f0e')
    bars3 = ax5.bar(x + width/2, rel_time, width, label='Norm. Inference Time', color='#2ca02c')
    bars4 = ax5.bar(x + width*1.5, rel_size, width, label='Norm. Model Size', color='#d62728')
    
    ax5.set_title('Normalized Metrics by Bit Width (Relative to 32-bit)', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels([str(bw) for bw in results['bit_width']])
    ax5.set_xlabel('Quantization Bit Width', fontsize=12)
    ax5.set_ylabel('Normalized Value', fontsize=12)
    ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)  # Reference line
    ax5.legend()
    ax5.grid(True, axis='y', alpha=0.3)
    
    # Overall title
    plt.suptitle('Impact of Attention Layer Quantization Precision', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'attention_quantization_results_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'attention_quantization_results_comprehensive.pdf'), bbox_inches='tight')
    print(f"Comprehensive plot saved to {output_dir}")
    
    plt.close()


# Update measure_inference_speed to handle callable with args
def measure_inference_speed(func, func_args, num_runs=100):
    """Measure inference speed of a function"""
    device = next(iter(func.__self__.parameters())).device  # Get device from model
    
    # Warmup
    for _ in range(10):
        func(**func_args)
        
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        func(**func_args)
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Layer Quantization Study")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--calibration_data", type=str, required=True, help="Path to calibration dataset")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--num_calibration", type=int, default=100, help="Number of calibration samples")
    parser.add_argument("--num_eval_samples", type=int, default=50, help="Number of samples to generate for evaluation")
    parser.add_argument("--speed_test_runs", type=int, default=100, help="Number of runs for speed test")
    parser.add_argument("--symmetric", action="store_true", help="Use symmetric quantization")
    parser.add_argument("--per_channel", action="store_true", help="Use per-channel quantization")
    parser.add_argument("--training_aware", action="store_true", help="Use training-aware quantization")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_attention_quantization_study(args)