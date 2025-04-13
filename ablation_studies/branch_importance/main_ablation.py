import sys
import os
import torch
from ablation_study import DiffSearchAblation
from models.diffusion import Model

# Create a simple config function
def get_config():
    # Basic configuration for diffusion model
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
    config = AttrDict()
    
    # Model config
    config.model = AttrDict()
    config.model.ch = 64
    config.model.ch_mult = [1, 2, 2, 4]
    config.model.num_res_blocks = 2
    config.model.dropout = 0.1
    config.model.time_embed_dim = 256
    config.model.attention_resolutions = 1
    
    # Data config
    config.data = AttrDict()
    config.data.channels = 3
    config.data.image_size = 32
    
    return config

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and dataloader
    config = get_config()
    model = Model(config)
    model.to(device)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load dataloader - MODIFY THIS PART to use your actual dataset loading code
    # For example:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use CIFAR-10 as a placeholder - replace with your actual dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    print(f"Loaded dataloader with {len(dataloader)} batches")
    
    # Define hyperparameters for the ablation study
    lambda_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    eta_values = [0, 0.5, 1.0, 1.5, 2.5]
    
    # Run ablation study
    print("Initializing ablation study...")
    ablation = DiffSearchAblation(
        model=model,
        dataloader=dataloader,
        lambda_values=lambda_values,
        eta_values=eta_values,
        device=device
    )
    
    print("Running ablation study...")
    ablation.run_ablation()
    
    print("Visualizing results...")
    ablation.visualize_weights_evolution()
    
    print("Analyzing component importance...")
    ablation.analyze_components_importance()
    
    print("Ablation study completed. Results saved to 'importance_weights_ablation.png'")

if __name__ == "__main__":
    main()