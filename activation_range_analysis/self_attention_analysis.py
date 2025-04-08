import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from tqdm import tqdm
import argparse
import seaborn as sns

class SelfAttentionHook:
    """Special hook for analyzing self-attention modules."""
    
    def __init__(self, name):
        self.name = name
        self.qkv_activations = defaultdict(list)
        self.attention_activations = defaultdict(list)
        self.output_activations = defaultdict(list)
        
    def __call__(self, module, inputs, output):
        # Assuming timestep is set as an attribute
        timestep = getattr(module, 'current_timestep', 0)
        
        # For modules with separate Q, K, V projections
        if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                x = inputs[0]
                
                # Get QKV projections
                q = module.to_q(x)
                k = module.to_k(x)
                v = module.to_v(x)
                
                # Record QKV statistics
                self.qkv_activations[timestep].append({
                    'q_min': q.detach().float().min().item(),
                    'q_max': q.detach().float().max().item(),
                    'q_range': q.detach().float().max().item() - q.detach().float().min().item(),
                    'k_min': k.detach().float().min().item(),
                    'k_max': k.detach().float().max().item(),
                    'k_range': k.detach().float().max().item() - k.detach().float().min().item(),
                    'v_min': v.detach().float().min().item(),
                    'v_max': v.detach().float().max().item(),
                    'v_range': v.detach().float().max().item() - v.detach().float().min().item(),
                })
        
        # For modules with combined QKV projection
        elif hasattr(module, 'qkv') or hasattr(module, 'qkv_proj'):
            qkv_proj = getattr(module, 'qkv', None) or getattr(module, 'qkv_proj', None)
            
            if qkv_proj is not None and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                x = inputs[0]
                qkv = qkv_proj(x)
                
                # Record combined QKV statistics
                self.qkv_activations[timestep].append({
                    'qkv_min': qkv.detach().float().min().item(),
                    'qkv_max': qkv.detach().float().max().item(),
                    'qkv_range': qkv.detach().float().max().item() - qkv.detach().float().min().item()
                })
        
        # Record attention matrix statistics if we can find it
        # Different models might store the attention matrix differently
        attn = None
        if hasattr(module, 'attention_probs') and isinstance(module.attention_probs, torch.Tensor):
            attn = module.attention_probs
        elif hasattr(module, 'attn_probs') and isinstance(module.attn_probs, torch.Tensor):
            attn = module.attn_probs
        elif hasattr(module, 'attn') and isinstance(module.attn, torch.Tensor):
            attn = module.attn
            
        if attn is not None:
            self.attention_activations[timestep].append({
                'attn_min': attn.detach().float().min().item(),
                'attn_max': attn.detach().float().max().item(),
                'attn_mean': attn.detach().float().mean().item(),
                'attn_range': attn.detach().float().max().item() - attn.detach().float().min().item()
            })
        
        # Record output statistics
        if isinstance(output, torch.Tensor):
            self.output_activations[timestep].append({
                'out_min': output.detach().float().min().item(),
                'out_max': output.detach().float().max().item(),
                'out_mean': output.detach().float().mean().item(),
                'out_range': output.detach().float().max().item() - output.detach().float().min().item()
            })
    
    def clear(self):
        self.qkv_activations.clear()
        self.attention_activations.clear()
        self.output_activations.clear()

class SelfAttentionAnalyzer:
    """Analyzer specialized for self-attention modules."""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.hooks = {}
        self.results = {}
    
    def find_attention_modules(self):
        """Find all self-attention modules in the model."""
        attention_modules = []
        
        for name, module in self.model.named_modules():
            # Different models might name attention modules differently
            if any(attn_name in name.lower() for attn_name in [
                'attn', 'attention', 'self_attention', 'selfattention'
            ]):
                attention_modules.append((name, module))
        
        return attention_modules
    
    def register_hooks(self):
        """Register hooks on all self-attention modules."""
        attention_modules = self.find_attention_modules()
        
        for name, module in attention_modules:
            hook = SelfAttentionHook(name)
            handle = module.register_forward_hook(hook)
            self.hooks[name] = (hook, handle)
            print(f"Registered self-attention hook on: {name}")
    
    def set_timestep(self, t):
        """Set the current timestep for all modules."""
        for name, module in self.model.named_modules():
            setattr(module, 'current_timestep', t)
    
    def run_inference(self, sample_batch, timesteps):
        """Run inference across specified timesteps."""
        self.model.eval()
        
        with torch.no_grad():
            for t in tqdm(timesteps, desc="Processing timesteps"):
                self.set_timestep(t)
                
                # This needs to be adapted for your specific model interface
                if hasattr(self.model, 'denoise_step'):
                    self.model.denoise_step(sample_batch, t)
                else:
                    # Generic forward pass - adjust based on your model's interface
                    noise = torch.randn_like(sample_batch)
                    self.model(sample_batch, t, noise)
    
    def collect_results(self):
        """Aggregate results from all hooks."""
        results = {}
        
        for name, (hook, _) in self.hooks.items():
            layer_results = {
                'qkv': {},
                'attention': {},
                'output': {}
            }
            
            # Process QKV activations
            for timestep, stats_list in hook.qkv_activations.items():
                # Average stats across multiple occurrences
                avg_stats = {}
                for key in stats_list[0].keys():
                    avg_stats[key] = np.mean([stats[key] for stats in stats_list])
                
                layer_results['qkv'][timestep] = avg_stats
            
            # Process attention matrix activations
            for timestep, stats_list in hook.attention_activations.items():
                avg_stats = {}
                if stats_list:
                    for key in stats_list[0].keys():
                        avg_stats[key] = np.mean([stats[key] for stats in stats_list])
                    
                    layer_results['attention'][timestep] = avg_stats
            
            # Process output activations
            for timestep, stats_list in hook.output_activations.items():
                avg_stats = {}
                for key in stats_list[0].keys():
                    avg_stats[key] = np.mean([stats[key] for stats in stats_list])
                
                layer_results['output'][timestep] = avg_stats
            
            results[name] = layer_results
        
        self.results = results
        return results
    
    def save_results(self, output_dir):
        """Save results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw data
        with open(os.path.join(output_dir, f"{self.model_name}_attention_stats.json"), 'w') as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            json_results = {}
            for name, layer_results in self.results.items():
                json_results[name] = {}
                for part, timestep_data in layer_results.items():
                    json_results[name][part] = {}
                    for timestep, stats in timestep_data.items():
                        json_results[name][part][timestep] = stats
            
            json.dump(json_results, f, indent=2)
    
    def cleanup(self):
        """Remove all hooks."""
        for _, (_, handle) in self.hooks.items():
            handle.remove()
        self.hooks.clear()

def plot_attention_ranges(results, model_name, output_dir):
    """Generate visualizations for self-attention activations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all layers
    for layer_name, layer_data in results.items():
        # Skip if no data
        if not layer_data['qkv'] and not layer_data['attention'] and not layer_data['output']:
            continue
        
        # Create timestep progression plots for this layer
        plt.figure(figsize=(15, 10))
        
        # Plot QKV ranges
        plt.subplot(3, 1, 1)
        if layer_data['qkv']:
            timesteps = sorted([int(t) for t in layer_data['qkv'].keys()])
            
            # Check if we have separate Q, K, V or combined QKV
            first_timestep = str(timesteps[0])
            if 'q_range' in layer_data['qkv'][first_timestep]:
                # Separate Q, K, V
                q_ranges = [layer_data['qkv'][str(t)]['q_range'] for t in timesteps]
                k_ranges = [layer_data['qkv'][str(t)]['k_range'] for t in timesteps]
                v_ranges = [layer_data['qkv'][str(t)]['v_range'] for t in timesteps]
                
                plt.plot(timesteps, q_ranges, 'r-', label='Q Range')
                plt.plot(timesteps, k_ranges, 'g-', label='K Range')
                plt.plot(timesteps, v_ranges, 'b-', label='V Range')
            elif 'qkv_range' in layer_data['qkv'][first_timestep]:
                # Combined QKV
                qkv_ranges = [layer_data['qkv'][str(t)]['qkv_range'] for t in timesteps]
                plt.plot(timesteps, qkv_ranges, 'purple', label='QKV Range')
        
        plt.title(f"{model_name} - {layer_name} - QKV Activation Ranges")
        plt.xlabel("Timestep")
        plt.ylabel("Range (Max - Min)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot Attention matrix ranges
        plt.subplot(3, 1, 2)
        if layer_data['attention']:
            timesteps = sorted([int(t) for t in layer_data['attention'].keys()])
            attn_ranges = [layer_data['attention'][str(t)]['attn_range'] for t in timesteps]
            
            plt.plot(timesteps, attn_ranges, 'orange', label='Attention Range')
        
        plt.title(f"{model_name} - {layer_name} - Attention Matrix Ranges")
        plt.xlabel("Timestep")
        plt.ylabel("Range (Max - Min)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot Output ranges
        plt.subplot(3, 1, 3)
        if layer_data['output']:
            timesteps = sorted([int(t) for t in layer_data['output'].keys()])
            out_ranges = [layer_data['output'][str(t)]['out_range'] for t in timesteps]
            
            plt.plot(timesteps, out_ranges, 'cyan', label='Output Range')
        
        plt.title(f"{model_name} - {layer_name} - Output Ranges")
        plt.xlabel("Timestep")
        plt.ylabel("Range (Max - Min)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        short_name = layer_name.split('.')[-2:]
        short_name = '_'.join(short_name)
        plt.savefig(os.path.join(output_dir, f"{model_name}_{short_name}_ranges.png"), dpi=300)
        plt.close()
    
    # Create summary heatmap across all layers
    layer_names = list(results.keys())
    if not layer_names:
        return
    
    # Get all timesteps (assuming they're consistent across layers)
    first_layer = list(results.values())[0]
    if 'output' in first_layer and first_layer['output']:
        timesteps = sorted([int(t) for t in first_layer['output'].keys()])
        
        # Create output range heatmap
        heatmap_data = np.zeros((len(layer_names), len(timesteps)))
        
        for i, layer_name in enumerate(layer_names):
            layer_data = results[layer_name]
            if 'output' in layer_data and layer_data['output']:
                for j, t in enumerate(timesteps):
                    if str(t) in layer_data['output']:
                        heatmap_data[i, j] = layer_data['output'][str(t)]['out_range']
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, 
                   xticklabels=[str(t) for t in timesteps[::5]],  # Show every 5th timestep label
                   yticklabels=[l.split('.')[-2:] for l in layer_names],
                   cmap='viridis')
        
        plt.title(f"{model_name} - Attention Layer Output Ranges Across Timesteps")
        plt.xlabel("Timestep")
        plt.ylabel("Layer")
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{model_name}_attention_heatmap.png"), dpi=300)
        plt.close()
        
        # Create a plot showing average ranges by timestep
        avg_ranges_by_timestep = np.mean(heatmap_data, axis=0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps, avg_ranges_by_timestep)
        plt.title(f"{model_name} - Average Attention Output Range by Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Average Range (Max - Min)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(output_dir, f"{model_name}_avg_range_by_timestep.png"), dpi=300)
        plt.close()

def analyze_model(model_type, model, output_dir, batch_size=4, image_size=64, channels=3, device='cuda'):
    """Analyze self-attention activations for a specific model."""
    print(f"Analyzing self-attention in {model_type} model...")
    
    # Create dummy input
    sample_batch = torch.randn(batch_size, channels, image_size, image_size, device=device)
    
    # Generate timesteps to sample
    timesteps = list(range(0, 1000, 25))  # Sample every 25 timesteps
    
    # Create analyzer
    analyzer = SelfAttentionAnalyzer(model, model_type)
    
    # Register hooks
    analyzer.register_hooks()
    
    # Run inference
    analyzer.run_inference(sample_batch, timesteps)
    
    # Collect results
    results = analyzer.collect_results()
    
    # Save results
    model_output_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    analyzer.save_results(model_output_dir)
    
    # Generate plots
    plot_attention_ranges(results, model_type, model_output_dir)
    
    # Clean up
    analyzer.cleanup()
    
    print(f"Self-attention analysis completed for {model_type}. Results saved to {model_output_dir}")
    return results

def cross_model_comparison(all_results, output_dir):
    """Create comparison plots across all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for comparison
    models = list(all_results.keys())
    if not models:
        return
    
    # Compare average output ranges across models
    avg_output_ranges = {}
    
    for model_name, model_results in all_results.items():
        model_output_ranges = []
        
        for layer_name, layer_data in model_results.items():
            if 'output' in layer_data and layer_data['output']:
                # Calculate average range across all timesteps for this layer
                layer_ranges = []
                for timestep, stats in layer_data['output'].items():
                    layer_ranges.append(stats['out_range'])
                
                if layer_ranges:
                    model_output_ranges.append(np.mean(layer_ranges))
        
        if model_output_ranges:
            avg_output_ranges[model_name] = {
                'mean': np.mean(model_output_ranges),
                'std': np.std(model_output_ranges)
            }
    
    # Plot average output ranges
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(avg_output_ranges))
    models = list(avg_output_ranges.keys())
    means = [avg_output_ranges[m]['mean'] for m in models]
    stds = [avg_output_ranges[m]['std'] for m in models]
    
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, models)
    plt.xlabel('Model')
    plt.ylabel('Average Output Range (Max - Min)')
    plt.title('Self-Attention Output Ranges Across Models')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, "model_comparison_output_ranges.png"), dpi=300)
    plt.close()
    
    # Compare range patterns across timesteps
    plt.figure(figsize=(12, 8))
    
    for model_name, model_results in all_results.items():
        # Find first attention layer with output data
        timestep_data = None
        
        for layer_data in model_results.values():
            if 'output' in layer_data and layer_data['output']:
                timesteps = sorted([int(t) for t in layer_data['output'].keys()])
                ranges = [layer_data['output'][str(t)]['out_range'] for t in timesteps]
                
                timestep_data = (timesteps, ranges)
                break
        
        if timestep_data:
            timesteps, ranges = timestep_data
            plt.plot(timesteps, ranges, label=model_name)
    
    plt.xlabel('Timestep')
    plt.ylabel('Output Range (Max - Min)')
    plt.title('Self-Attention Output Range Patterns Across Timesteps')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "timestep_pattern_comparison.png"), dpi=300)
    plt.close()

def load_models(device):
    """Load all diffusion models for analysis."""
    # This function needs to be adapted to your specific model loading logic
    models = {}
    
    try:
        # Import your models - adjust these imports to match your repository structure
        from models.ddim import DDIM
        models['DDIM'] = DDIM().to(device)
        print("Loaded DDIM model")
    except:
        print("Failed to load DDIM model")
    
    try:
        from models.stable_diffusion import StableDiffusion
        models['StableDiffusion'] = StableDiffusion().to(device)
        print("Loaded Stable Diffusion model")
    except:
        print("Failed to load Stable Diffusion model")
    
    try:
        from models.ldm import LDM
        models['LDM-Bedroom'] = LDM(dataset='bedroom').to(device)
        print("Loaded LDM-Bedroom model")
    except:
        print("Failed to load LDM-Bedroom model")
    
    try:
        from models.ldm import LDM
        models['LDM-Church'] = LDM(dataset='church').to(device)
        print("Loaded LDM-Church model")
    except:
        print("Failed to load LDM-Church model")
    
    return models

def main():
    parser = argparse.ArgumentParser(description="Analyze self-attention modules in diffusion models")
    parser.add_argument("--output_dir", type=str, default="attention_analysis_results", 
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run analysis on")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--models", nargs='+', default=['DDIM', 'StableDiffusion', 'LDM-Bedroom', 'LDM-Church'],
                        help="Models to analyze")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    models = load_models(args.device)
    
    # Validate models
    valid_models = {name: model for name, model in models.items() if name in args.models}
    if not valid_models:
        print("No valid models found! Please check your model imports.")
        return
    
    # Model-specific settings
    model_settings = {
        'DDIM': {'image_size': 32, 'channels': 3},
        'StableDiffusion': {'image_size': 512, 'channels': 4},
        'LDM-Bedroom': {'image_size': 256, 'channels': 3},
        'LDM-Church': {'image_size': 256, 'channels': 3}
    }
    
    # Analyze each model
    all_results = {}
    for model_name, model in valid_models.items():
        settings = model_settings.get(model_name, {'image_size': 64, 'channels': 3})
        results = analyze_model(
            model_name, 
            model, 
            args.output_dir, 
            batch_size=args.batch_size,
            image_size=settings['image_size'],
            channels=settings['channels'],
            device=args.device
        )
        all_results[model_name] = results
    
    # Generate cross-model comparison plots
    cross_model_comparison(all_results, args.output_dir)
    
    print(f"All analyses completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()