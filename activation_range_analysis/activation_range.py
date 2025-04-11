import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from tqdm import tqdm

class ActivationHook:
    """Hook for recording activation statistics from model layers."""
    
    def __init__(self, name):
        self.name = name
        self.activations = defaultdict(list)
        
    def __call__(self, module, input, output):
        # Store statistics per timestep
        # Assuming timestep is available in a global context or passed somehow
        timestep = getattr(module, 'current_timestep', 0)
        
        if isinstance(output, torch.Tensor):
            # Calculate statistics
            min_val = output.detach().float().min().item()
            max_val = output.detach().float().max().item()
            mean = output.detach().float().mean().item()
            std = output.detach().float().std().item()
            
            self.activations[timestep].append({
                'min': min_val,
                'max': max_val,
                'mean': mean,
                'std': std,
                'range': max_val - min_val
            })
    
    def clear(self):
        self.activations.clear()

class ModelAnalyzer:
    """Analyzer for model activations across timesteps."""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.hooks = {}
        self.results = {}
        
    def register_hooks(self, target_modules=None):
        """Register hooks on target modules or all modules by default."""
        if target_modules is None:
            # Default: hook all potential layers of interest
            target_modules = [
                # For DDIM/LDM
                'qkv_proj', 'proj', 'conv', 'norm', 'linear',
                # For Stable Diffusion
                'to_q', 'to_k', 'to_v', 'to_out', 'ff'
            ]
        
        for name, module in self.model.named_modules():
            # Check if any target name appears in the module name
            if any(target in name.lower() for target in target_modules):
                hook = ActivationHook(name)
                handle = module.register_forward_hook(hook)
                self.hooks[name] = (hook, handle)
                print(f"Registered hook on: {name}")
    
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
                
                # This will need to be adapted for your specific model interface
                # Here's a generic example:
                if hasattr(self.model, 'denoise_step'):
                    self.model.denoise_step(sample_batch, t)
                else:
                    # Generic forward pass - adjust based on your model's interface
                    noise = torch.randn_like(sample_batch)
                    self.model(sample_batch, t, noise)
    
    def collect_results(self):
        """Aggregate results from all hooks."""
        layer_results = {}
        
        for name, (hook, _) in self.hooks.items():
            layer_data = defaultdict(lambda: defaultdict(list))
            
            # Aggregate across timesteps
            for timestep, stats_list in hook.activations.items():
                for stats in stats_list:
                    for stat_name, value in stats.items():
                        layer_data[stat_name][timestep].append(value)
            
            # Calculate averages per timestep
            layer_avg = {}
            for stat_name, timestep_data in layer_data.items():
                layer_avg[stat_name] = {
                    t: np.mean(values) for t, values in timestep_data.items()
                }
            
            layer_results[name] = layer_avg
        
        self.results = layer_results
        return layer_results
    
    def save_results(self, output_dir):
        """Save results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw data
        with open(os.path.join(output_dir, f"{self.model_name}_activation_stats.json"), 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def cleanup(self):
        """Remove all hooks."""
        for _, (_, handle) in self.hooks.items():
            handle.remove()
        self.hooks.clear()

def plot_activation_ranges(results, model_name, output_dir):
    """Plot activation ranges for different layers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group layers by type for more organized visualization
    layer_groups = defaultdict(list)
    
    for layer_name in results.keys():
        if 'attn' in layer_name or 'attention' in layer_name:
            group = 'attention'
        elif 'conv' in layer_name:
            group = 'conv'
        elif 'norm' in layer_name:
            group = 'norm'
        elif 'ff' in layer_name or 'feed_forward' in layer_name:
            group = 'feedforward'
        else:
            group = 'other'
        
        layer_groups[group].append(layer_name)
    
    # Plot ranges by group
    for group, layers in layer_groups.items():
        plt.figure(figsize=(12, 8))
        
        for i, layer_name in enumerate(sorted(layers)[:10]):  # Limit to 10 layers per plot for readability
            layer_data = results[layer_name]['range']
            
            # Convert to average per timestep if needed
            if isinstance(layer_data, dict):
                # Sort by timestep
                timesteps = sorted(int(t) for t in layer_data.keys())
                values = [layer_data[str(t)] for t in timesteps]
                
                plt.plot(timesteps, values, label=f"{layer_name.split('.')[-1]}")
            else:
                plt.bar(i, layer_data, label=layer_name.split('.')[-1])
        
        plt.title(f"{model_name} - {group.capitalize()} Layer Activation Ranges")
        plt.xlabel("Timestep" if isinstance(layer_data, dict) else "Layer")
        plt.ylabel("Activation Range (Max - Min)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"{model_name}_{group}_activation_ranges.png"), dpi=300)
        plt.close()
    
    # Create a summary plot showing average ranges across all timesteps for key layer types
    avg_ranges = defaultdict(list)
    
    for group, layers in layer_groups.items():
        for layer_name in layers:
            range_data = results[layer_name]['range']
            if isinstance(range_data, dict):
                avg_ranges[group].append(np.mean(list(range_data.values())))
            else:
                avg_ranges[group].append(range_data)
    
    plt.figure(figsize=(10, 6))
    for group, ranges in avg_ranges.items():
        plt.bar(group, np.mean(ranges), yerr=np.std(ranges), capsize=5)
    
    plt.title(f"{model_name} - Average Activation Ranges by Layer Type")
    plt.ylabel("Activation Range (Max - Min)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{model_name}_summary_activation_ranges.png"), dpi=300)
    plt.close()