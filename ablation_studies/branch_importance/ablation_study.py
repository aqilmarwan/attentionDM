import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import os

# Add the parent directory to sys.path so Python can find your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import Model
from utils.metrics import calculate_fid  

class DiffSearchAblation:
    def __init__(self, model, dataloader, lambda_values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                 eta_values=[0, 0.5, 1.0, 1.5, 2.5], device='cuda'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.lambda_values = lambda_values
        self.eta_values = eta_values
        self.importance_weights_history = {}
        self.fid_scores = {}
        self.device = device
        
        # Count model components for weighting
        self.num_res_blocks = len(model.down_blocks) + 2  # Include middle blocks
        self.num_attention_layers = 1  # middle_attn
        if hasattr(model, 'up_blocks'):
            self.num_res_blocks += len(model.up_blocks)
            # Count attention blocks in up_blocks if they have them
            for block in model.up_blocks:
                if hasattr(block, 'attn') and not isinstance(block.attn, torch.nn.Identity):
                    self.num_attention_layers += 1
        
    def initialize_architecture_weights(self):
        # Initialize weights for different components
        self.arch_weights = {
            'resblocks': torch.ones(self.num_res_blocks, requires_grad=True, device=self.device),
            'attention': torch.ones(self.num_attention_layers, requires_grad=True, device=self.device),
            'timestep_embed': torch.ones(1, requires_grad=True, device=self.device)
        }
        return self.arch_weights
    
    def importance_regularization(self, weights, eta):
        # L1 regularization for sparsity
        return eta * torch.sum(torch.abs(weights))
    
    def train_with_differentiable_search(self, lambda_val, eta_val, epochs=5):
        arch_weights = self.initialize_architecture_weights()
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 1e-4},
            {'params': list(arch_weights.values()), 'lr': 5e-3}
        ])
        
        weights_history = {k: [] for k in arch_weights.keys()}
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                # Extract data from batch
                if isinstance(batch, (tuple, list)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                # Apply architecture weights
                self.model.apply_arch_weights(arch_weights)
                
                # Calculate timesteps (random for training)
                batch_size = x.shape[0]
                t = torch.randint(0, 1000, (batch_size,), device=self.device)
                
                # Forward pass and loss calculation
                noise = torch.randn_like(x)
                x_noisy = self.add_noise(x, t, noise)
                predicted_noise = self.model(x_noisy, t)
                
                # MSE loss between predicted and actual noise
                loss = torch.mean((predicted_noise - noise) ** 2)
                
                # Add importance regularization
                reg_losses = [self.importance_regularization(w, eta_val) for w in arch_weights.values()]
                reg_loss = sum(reg_losses)
                total_loss = loss + lambda_val * reg_loss
                
                total_loss.backward(retain_graph=True)
                optimizer.step()
                
                # Normalize weights after update
                for k, w in arch_weights.items():
                    arch_weights[k] = torch.softmax(w, dim=0)
                
                epoch_loss += total_loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
            # Store weights history
            for k, w in arch_weights.items():
                weights_history[k].append(w.detach().cpu().numpy())
        
        # Calculate FID score
        print("Calculating FID score...")
        fid = calculate_fid(self.model, self.dataloader, device=self.device)
        print(f"FID Score: {fid:.2f}")
        
        return weights_history, fid
    
    def add_noise(self, x, t, noise):
        """Add noise to input according to diffusion schedule"""
        # Simple linear schedule from 0.0001 to 0.02
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, 1000, device=self.device)
        
        # Extract the corresponding beta values
        beta_t = betas[t]
        
        # Calculate alphas and related terms
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Add noise according to the schedule
        x_noisy = torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise
        
        return x_noisy
    
    def run_ablation(self):
        for lambda_val in self.lambda_values:
            self.fid_scores[lambda_val] = {}
            for eta_val in self.eta_values:
                print(f"Running with λ={lambda_val}, η={eta_val}")
                weights_history, fid = self.train_with_differentiable_search(lambda_val, eta_val)
                
                key = f"lambda_{lambda_val}_eta_{eta_val}"
                self.importance_weights_history[key] = weights_history
                self.fid_scores[lambda_val][eta_val] = fid
                
                # Save intermediate results
                torch.save({
                    'lambda': lambda_val,
                    'eta': eta_val,
                    'weights_history': weights_history,
                    'fid': fid
                }, f"ablation_results_{key}.pt")
    
    def visualize_weights_evolution(self):
        plt.figure(figsize=(15, 6))
        
        # Plot evolution of weights (similar to Figure 5.5a)
        plt.subplot(1, 2, 1)
        
        # Find the best performing configuration
        best_lambda = min(self.fid_scores.keys(), 
                          key=lambda l: min(self.fid_scores[l].values()))
        best_eta = min(self.fid_scores[best_lambda].keys(), 
                        key=lambda e: self.fid_scores[best_lambda][e])
        best_key = f"lambda_{best_lambda}_eta_{best_eta}"
        
        # Plot the weights from the best configuration
        for i, k in enumerate(['resblocks', 'attention', 'timestep_embed']):
            weights = self.importance_weights_history[best_key][k]
            
            # For multi-dimensional weights (like resblocks or attention)
            if len(weights[0].shape) > 0:
                for j in range(len(weights[0])):
                    plt.plot(range(len(weights)), [w[j] for w in weights], 
                             marker='o', label=f"{k}-{j}")
            else:
                plt.plot(range(len(weights)), weights, marker='o', label=k)
        
        plt.title('Evolution of Branch Importance Weights')
        plt.xlabel('Epoch')
        plt.ylabel('Importance Weights')
        plt.grid(True)
        plt.legend()
        
        # Plot FID scores (similar to Figure 5.5b)
        plt.subplot(1, 2, 2)
        for eta in self.eta_values:
            fids = [self.fid_scores[lambda_val][eta] for lambda_val in self.lambda_values]
            plt.plot(self.lambda_values, fids, marker='o', label=f"η={eta}")
        
        plt.title('Generation Quality')
        plt.xlabel('λ')
        plt.ylabel('FID')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('importance_weights_ablation.png', dpi=300)
        plt.show()
    
    def analyze_components_importance(self):
        # Find best configuration
        best_lambda = min(self.fid_scores.keys(), 
                          key=lambda l: min(self.fid_scores[l].values()))
        best_eta = min(self.fid_scores[best_lambda].keys(), 
                        key=lambda e: self.fid_scores[best_lambda][e])
        best_key = f"lambda_{best_lambda}_eta_{best_eta}"
        
        # Get final weights from best configuration
        final_weights = {k: w[-1] for k, w in self.importance_weights_history[best_key].items()}
        
        print("\n======= Component Importance Analysis =======")
        print(f"Best configuration: λ={best_lambda}, η={best_eta}, FID={self.fid_scores[best_lambda][best_eta]:.2f}")
        
        # Analyze each component type
        for k, weights in final_weights.items():
            if len(weights.shape) > 0:  # Vector of weights
                sorted_indices = np.argsort(weights)[::-1]  # Sort descending
                print(f"\n{k.capitalize()} importance ranking:")
                for i, idx in enumerate(sorted_indices):
                    print(f"  Rank {i+1}: {k}[{idx}] = {weights[idx]:.4f}")
            else:  # Single weight
                print(f"\n{k.capitalize()} importance: {float(weights):.4f}")
        
        # Overall analysis
        print("\nKey findings:")
        if 'resblocks' in final_weights and len(final_weights['resblocks']) > 0:
            most_important_resblock = np.argmax(final_weights['resblocks'])
            print(f"- Most important residual block: {most_important_resblock}")
            
        if 'attention' in final_weights and len(final_weights['attention']) > 0:
            if np.max(final_weights['attention']) > np.max(final_weights['resblocks']):
                print("- Attention mechanisms are more important than residual blocks")
            else:
                print("- Residual blocks are more important than attention mechanisms")
                
        if 'timestep_embed' in final_weights:
            if float(final_weights['timestep_embed']) > 0.5:
                print("- Timestep embedding is highly important for generation quality")
            else:
                print("- Timestep embedding has less impact on generation quality")