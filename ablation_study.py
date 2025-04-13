import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from models.diffusion import Model as DiffusionModel
from utils.metrics import calculate_fid

class DiffSearchAblation:
    def __init__(self, model, dataloader, lambda_values=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                 eta_values=[0, 0.5, 1.0, 1.5, 2.5]):
        self.model = model
        self.dataloader = dataloader
        self.lambda_values = lambda_values
        self.eta_values = eta_values
        self.importance_weights_history = {}
        self.fid_scores = {}
        
    def initialize_architecture_weights(self):
        # Initialize weights for different components
        self.arch_weights = {
            'resblocks': torch.ones(self.model.num_res_blocks, requires_grad=True),
            'attention': torch.ones(self.model.num_attention_layers, requires_grad=True),
            'timestep_embed': torch.ones(1, requires_grad=True)
        }
        return self.arch_weights
    
    def importance_regularization(self, weights, eta):
        # L1 regularization for sparsity
        return eta * torch.sum(torch.abs(weights))
    
    def train_with_differentiable_search(self, lambda_val, eta_val, epochs=10):
        arch_weights = self.initialize_architecture_weights()
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 1e-4},
            {'params': list(arch_weights.values()), 'lr': 5e-3}
        ])
        
        weights_history = {k: [] for k in arch_weights.keys()}
        
        for epoch in range(epochs):
            for batch in tqdm(self.dataloader):
                optimizer.zero_grad()
                
                # Apply architecture weights to corresponding components
                self.model.apply_arch_weights(arch_weights)
                
                # Regular model loss
                loss = self.model.compute_loss(batch)
                
                # Add importance regularization
                reg_loss = sum(self.importance_regularization(w, eta_val) for w in arch_weights.values())
                total_loss = loss + lambda_val * reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                # Normalize weights after update
                for k, w in arch_weights.items():
                    arch_weights[k] = torch.softmax(w, dim=0)
            
            # Store weights history
            for k, w in arch_weights.items():
                weights_history[k].append(w.detach().cpu().numpy())
                
        # Calculate FID score
        fid = calculate_fid(self.model, self.dataloader)
        
        return weights_history, fid
    
    def run_ablation(self):
        for lambda_val in self.lambda_values:
            self.fid_scores[lambda_val] = {}
            for eta_val in self.eta_values:
                print(f"Running with λ={lambda_val}, η={eta_val}")
                weights_history, fid = self.train_with_differentiable_search(lambda_val, eta_val)
                
                key = f"lambda_{lambda_val}_eta_{eta_val}"
                self.importance_weights_history[key] = weights_history
                self.fid_scores[lambda_val][eta_val] = fid
    
    def visualize_weights_evolution(self):
        plt.figure(figsize=(12, 5))
        
        # Plot evolution of weights (similar to Figure 5.5a)
        plt.subplot(1, 2, 1)
        for i, k in enumerate(['resblocks', 'attention', 'timestep_embed']):
            weights = self.importance_weights_history['lambda_0.6_eta_1.0'][k]
            if len(weights[0].shape) > 0:  # For multi-dimensional weights
                for j in range(len(weights[0])):
                    plt.plot(self.lambda_values, [w[j] for w in weights], 
                             marker='o', label=f"{k}-{j}")
            else:
                plt.plot(self.lambda_values, weights, marker='o', label=k)
        
        plt.title('Evolution of Branch Importance Weights')
        plt.xlabel('Iteration')
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
        plt.savefig('importance_weights_ablation.png')
        plt.show()
    
    def analyze_components_importance(self):
        # Analyze importance across denoising stages
        key = f"lambda_0.6_eta_1.0"  # Use the best hyperparameters
        final_weights = {k: w[-1] for k, w in self.importance_weights_history[key].items()}
        
        # This would depend on your model's specific architecture
        print("Component Importance Analysis:")
        for k, w in final_weights.items():
            if len(w.shape) > 0:
                print(f"{k}: {w}")
            else:
                print(f"{k}: {float(w)}") 