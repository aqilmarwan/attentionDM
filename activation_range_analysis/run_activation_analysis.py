import os
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json

# Import your models (adjust imports based on your repository structure)
from models.ddim import DDIM
from models.ldm import LDM
from models.stable_diffusion import StableDiffusion
from activation_analysis import ModelAnalyzer, plot_activation_ranges

def analyze_model(model_type, output_dir, batch_size=4, image_size=64, channels=3, device='cuda'):
    """Analyze activation ranges for a specific model type."""
    print(f"Analyzing {model_type} model...")
    
    # Create dummy input
    sample_batch = torch.randn(batch_size, channels, image_size, image_size, device=device)
    
    # Generate timesteps to sample (you may want to use actual diffusion timesteps)
    # For this analysis, we'll sample timesteps from the beginning, middle, and end
    timesteps = list(range(0, 1000, 50))  # Adjust based on your diffusion schedule
    
    # Initialize appropriate model
    if model_type == 'DDIM':
        model = DDIM().to(device)
        image_size = 32  # CIFAR-10 size
    elif model_type == 'StableDiffusion':
        model = StableDiffusion().to(device)
        image_size = 512
    elif model_type == 'LDM-Bedroom':
        model = LDM(dataset='bedroom').to(device)
        image_size = 256
    elif model_type == 'LDM-Church':
        model = LDM(dataset='church').to(device)
        image_size = 256
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create analyzer
    analyzer = ModelAnalyzer(model, model_type)
    
    # Register hooks for activation statistics
    analyzer.register_hooks()
    
    # Run inference to collect statistics
    analyzer.run_inference(sample_batch, timesteps)
    
    # Collect and process results
    results = analyzer.collect_results()
    
    # Save results
    model_output_dir = os.path.join(output_dir, model_type)
    analyzer.save_results(model_output_dir)
    
    # Generate plots
    plot_activation_ranges(results, model_type, model_output_dir)
    
    # Clean up
    analyzer.cleanup()
    
    print(f"Analysis completed for {model_type}. Results saved to {model_output_dir}")
    return results

def cross_model_comparison_plot(all_results, output_dir):
    """Create comparison plots across all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Average activation range per model
    avg_ranges = {}
    for model_name, results in all_results.items():
        model_ranges = []
        for layer_name, stats in results.items():
            if 'range' in stats:
                if isinstance(stats['range'], dict):
                    # Average across timesteps
                    model_ranges.append(np.mean(list(stats['range'].values())))
                else:
                    model_ranges.append(stats['range'])
        
        avg_ranges[model_name] = np.mean(model_ranges)
    
    # Plot average ranges
    plt.figure(figsize=(10, 6))
    models = list(avg_ranges.keys())
    ranges = [avg_ranges[model] for model in models]
    
    plt.bar(models, ranges)
    plt.title("Average Activation Range Across Models")
    plt.ylabel("Activation Range (Max - Min)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "model_comparison_activation_ranges.png"), dpi=300)
    plt.close()
    
    # Create timestep progression comparison
    plt.figure(figsize=(12, 8))
    
    # We'll use the first attention layer from each model for comparison
    for model_name, results in all_results.items():
        for layer_name, stats in results.items():
            if 'attn' in layer_name or 'attention' in layer_name:
                if 'range' in stats and isinstance(stats['range'], dict):
                    # Sort by timestep
                    timesteps = sorted(int(t) for t in stats['range'].keys())
                    values = [stats['range'][str(t)] for t in timesteps]
                    
                    plt.plot(timesteps, values, label=f"{model_name} - {layer_name.split('.')[-1]}")
                    break  # Only use the first attention layer
    
    plt.title("Activation Range Progression Across Timesteps")
    plt.xlabel("Timestep")
    plt.ylabel("Activation Range (Max - Min)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "timestep_progression_comparison.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze activation ranges in diffusion models")
    parser.add_argument("--output_dir", type=str, default="activation_analysis_results", 
                        help="Directory to save results")
    parser.add_argument("--models", nargs='+', default=['DDIM', 'StableDiffusion', 'LDM-Bedroom', 'LDM-Church'],
                        help="Models to analyze")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run analysis on")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze each model
    all_results = {}
    for model_type in args.models:
        results = analyze_model(model_type, args.output_dir, device=args.device)
        all_results[model_type] = results
    
    # Generate cross-model comparison plots
    cross_model_comparison_plot(all_results, args.output_dir)
    
    print(f"All analyses completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()