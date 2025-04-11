import torch
import os
import time
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

def find_attention_layers(model):
    """Find all attention layers in the model"""
    attention_layers = []
    
    for name, module in model.named_modules():
        # This pattern needs to be adapted based on the actual attention modules in tinyDM
        if any(attn_type in module.__class__.__name__.lower() for 
               attn_type in ['attention', 'transformer']):
            attention_layers.append((name, module))
            
    return attention_layers

def apply_quantization(model, bit_width=8, symmetric=True, per_channel=False):
    """Apply quantization to all attention layers in the model"""
    from .attention_quant import QuantizedAttention
    
    attention_layers = find_attention_layers(model)
    quantized_modules = []
    
    for name, module in attention_layers:
        print(f"Quantizing attention layer: {name}")
        quantized_module = QuantizedAttention(
            module, 
            bit_width=bit_width,
            symmetric=symmetric,
            per_channel=per_channel
        )
        quantized_modules.append(quantized_module)
        
    return quantized_modules

def calculate_fid(real_images, generated_images):
    """Calculate FID score between real and generated images"""
    fid = FrechetInceptionDistance(feature=64)
    
    # Add real images
    for img in real_images:
        fid.update(img.unsqueeze(0), real=True)
        
    # Add generated images
    for img in generated_images:
        fid.update(img.unsqueeze(0), real=False)
        
    return fid.compute().item()

def calculate_ssim(real_images, generated_images):
    """Calculate SSIM between real and generated images"""
    ssim_values = []
    
    for real, gen in zip(real_images, generated_images):
        # Convert to numpy arrays
        real_np = real.permute(1, 2, 0).cpu().numpy()
        gen_np = gen.permute(1, 2, 0).cpu().numpy()
        
        # Calculate SSIM
        ssim_val = ssim(real_np, gen_np, multichannel=True)
        ssim_values.append(ssim_val)
        
    return np.mean(ssim_values)

def measure_inference_speed(model, input_shape, num_runs=100):
    """Measure inference speed of the model"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        model(dummy_input)
        
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        model(dummy_input)
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time

def prepare_calibration_data(dataset_path, num_samples=100):
    """Prepare calibration data from a dataset without class folders"""
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    
    # Load images directly from the folder (no subfolders needed)
    image_files = [f for f in os.listdir(dataset_path) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    calibration_data = []
    for i in range(min(num_samples, len(image_files))):
        img_path = os.path.join(dataset_path, image_files[i])
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        calibration_data.append(img_tensor)
        
    return calibration_data