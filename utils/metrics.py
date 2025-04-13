import torch
import numpy as np
from tqdm import tqdm
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

class InceptionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.eval()
        # Remove classification layer
        self.model.fc = torch.nn.Identity()
    
    def forward(self, x):
        # Adjust input format if needed
        if x.shape[1] == 1:  # If grayscale, repeat to make RGB
            x = x.repeat(1, 3, 1, 1)
        # Resize if needed
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Scale from [-1, 1] to [0, 1]
        if x.min() < 0:
            x = (x + 1) / 2
        # Forward pass
        with torch.no_grad():
            features = self.model(x)
        return features

def calculate_activation_statistics(images, model, batch_size=50, device='cuda'):
    model.to(device)
    act = []
    
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size].to(device)
            batch_acts = model(batch)
            act.append(batch_acts.cpu().numpy())
    
    act = np.concatenate(act, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(model, dataloader, sample_count=1000, batch_size=50, device='cuda'):
    """Calculate FID score between generated and real images."""
    inception = InceptionModel().to(device)
    
    # Generate samples
    generated_images = []
    with torch.no_grad():
        for _ in tqdm(range(0, sample_count, batch_size), desc="Generating samples"):
            batch_size = min(batch_size, sample_count - len(generated_images))
            noise = torch.randn(batch_size, 3, 32, 32).to(device)  # Adjust dimensions as needed
            timesteps = torch.zeros(batch_size, dtype=torch.long).to(device)
            samples = model(noise, timesteps)
            generated_images.append(samples.cpu())
    
    generated_images = torch.cat(generated_images, dim=0)
    
    # Get real images
    real_images = []
    for batch in tqdm(dataloader, desc="Processing real images"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        real_images.append(images)
        if len(real_images) * images.shape[0] >= sample_count:
            break
    
    real_images = torch.cat(real_images, dim=0)[:sample_count]
    
    # Calculate activation statistics
    m1, s1 = calculate_activation_statistics(real_images, inception, batch_size, device)
    m2, s2 = calculate_activation_statistics(generated_images, inception, batch_size, device)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value 