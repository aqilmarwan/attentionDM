#!/usr/bin/env python3
# Ablation Study: Impact of Attention Layer Quantization Precision
# 
# This script implements the ablation study to evaluate how quantization precision
# of attention layers impacts model performance compared to other components.
#
# Creates multiple model variants with different quantization strategies:
# - Variant A: Uniform quantization (baseline) at 4-bit precision
# - Variant B: 4-bit quantization for convolutional layers, 8-bit for attention layers
# - Variant C: 8-bit quantization for convolutional layers, 4-bit for attention layers
# - Variant D: Full 8-bit quantization

import os
import logging
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import torchvision.utils as tvu

from models.diffusion import Model
from models.ema import EMAHelper
from runners.diffusion import Diffusion, get_beta_schedule
from datasets import get_dataset, data_transform, inverse_data_transform
from functions import get_optimizer
from utils.quant_util import QConv2d
from models.self_attention import QSelfAttention, EnhancedQSelfAttention

# Attempt to import CLIP for CLIP score computation if available
try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available. CLIP score evaluation will be skipped.")

# Attempt to import FID calculation if available
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("pytorch-fid not available. FID evaluation will be skipped.")


class AttentionQuantizationAblation:
    def __init__(self, config_path, exp_dir="ablation_results", device=None):
        """
        Initialize the ablation study
        
        Args:
            config_path: Path to the configuration file
            exp_dir: Directory to save experiment results
            device: Device to run experiments on
        """
        self.exp_dir = exp_dir
        self.variant_dirs = {
            'A': os.path.join(exp_dir, 'variant_A_uniform_4bit'),
            'B': os.path.join(exp_dir, 'variant_B_conv_4bit_attn_8bit'),
            'C': os.path.join(exp_dir, 'variant_C_conv_8bit_attn_4bit'),
            'D': os.path.join(exp_dir, 'variant_D_uniform_8bit')
        }
        
        # Create experiment directories
        os.makedirs(exp_dir, exist_ok=True)
        for variant_dir in self.variant_dirs.values():
            os.makedirs(variant_dir, exist_ok=True)
            os.makedirs(os.path.join(variant_dir, 'samples'), exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.config = self._dict_to_namespace(self.config)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Set up logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(exp_dir, 'ablation.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AttentionQuantizationAblation')
        
        # Initialize CLIP model if available
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
    
    def _dict_to_namespace(self, config_dict):
        """Convert dictionary to namespace for compatibility with existing code"""
        namespace = argparse.Namespace()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                new_value = self._dict_to_namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    
    def prepare_model_variants(self):
        """
        Prepare the four model variants with different quantization strategies
        
        Returns:
            Dictionary of model variants
        """
        variants = {}
        
        # Create mock args for each variant
        args_a = argparse.Namespace(bitwidth=4, calibrate_attention=False)
        args_b = argparse.Namespace(bitwidth=4, calibrate_attention=True)  # Will override attention to 8-bit
        args_c = argparse.Namespace(bitwidth=8, calibrate_attention=True)  # Will override attention to 4-bit
        args_d = argparse.Namespace(bitwidth=8, calibrate_attention=False)
        
        # Sequence for progressive quantization
        sequence = list(range(self.config.diffusion.num_diffusion_timesteps))
        
        # Variant A: Uniform 4-bit quantization
        self.logger.info("Creating Variant A: Uniform 4-bit quantization")
        model_a = Model(self.config, quantization=True, sequence=sequence, args=args_a)
        
        # Override EnhancedQSelfAttention's bit_config in each model
        self._set_attention_precision(model_a, query_bits=4, key_bits=4, value_bits=4, output_bits=4)
        variants['A'] = model_a.to(self.device)
        
        # Variant B: 4-bit conv, 8-bit attention
        self.logger.info("Creating Variant B: 4-bit conv, 8-bit attention")
        model_b = Model(self.config, quantization=True, sequence=sequence, args=args_b)
        self._set_attention_precision(model_b, query_bits=8, key_bits=8, value_bits=8, output_bits=8)
        variants['B'] = model_b.to(self.device)
        
        # Variant C: 8-bit conv, 4-bit attention
        self.logger.info("Creating Variant C: 8-bit conv, 4-bit attention")
        model_c = Model(self.config, quantization=True, sequence=sequence, args=args_c)
        self._set_attention_precision(model_c, query_bits=4, key_bits=4, value_bits=4, output_bits=4)
        variants['C'] = model_c.to(self.device)
        
        # Variant D: Uniform 8-bit quantization
        self.logger.info("Creating Variant D: Uniform 8-bit quantization")
        model_d = Model(self.config, quantization=True, sequence=sequence, args=args_d)
        self._set_attention_precision(model_d, query_bits=8, key_bits=8, value_bits=8, output_bits=8)
        variants['D'] = model_d.to(self.device)
        
        return variants
    
    def _set_attention_precision(self, model, query_bits, key_bits, value_bits, output_bits):
        """
        Set bit precision for all attention modules in the model
        
        Args:
            model: The model to modify
            query_bits: Bit precision for query projection
            key_bits: Bit precision for key projection
            value_bits: Bit precision for value projection
            output_bits: Bit precision for output projection
        """
        # Configure bit precision for all attention modules
        for module in model.modules():
            if isinstance(module, EnhancedQSelfAttention):
                module.bit_config = {
                    "query": query_bits,
                    "key": key_bits,
                    "value": value_bits,
                    "output": output_bits
                }
                
                # Need to recreate the quantized convs with new bit widths if they've been initialized
                if module.quantization and hasattr(module, 'query_conv'):
                    module.query_conv.w_bit = query_bits
                    module.query_conv.a_bit = query_bits
                    
                    module.key_conv.w_bit = key_bits
                    module.key_conv.a_bit = key_bits
                    
                    module.value_conv.w_bit = value_bits
                    module.value_conv.a_bit = value_bits
                    
                    module.output_conv.w_bit = output_bits
                    module.output_conv.a_bit = output_bits
            
            # Also handle basic QSelfAttention if present
            elif isinstance(module, QSelfAttention):
                # Set the bit width for all convs in the attention module
                if hasattr(module, 'query_conv') and isinstance(module.query_conv, QConv2d):
                    module.query_conv.w_bit = query_bits
                    module.query_conv.a_bit = query_bits
                    
                    module.key_conv.w_bit = key_bits
                    module.key_conv.a_bit = key_bits
                    
                    module.value_conv.w_bit = value_bits
                    module.value_conv.a_bit = value_bits
                    
                    module.output_conv.w_bit = output_bits
                    module.output_conv.a_bit = output_bits
    
    def load_pretrained_weights(self, model_variants, checkpoint_path):
        """
        Load pretrained weights into all model variants
        
        Args:
            model_variants: Dictionary of model variants
            checkpoint_path: Path to the checkpoint to load
        """
        self.logger.info(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, list):
            # If checkpoint is a list (standard format from training)
            state_dict = checkpoint[0]
        else:
            # If checkpoint is just the state dict
            state_dict = checkpoint
        
        # Load weights into all variants
        for variant, model in model_variants.items():
            model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Loaded weights for Variant {variant}")
    
    def calibrate_models(self, model_variants, num_calibration_samples=32):
        """
        Calibrate all model variants using sample data
        
        Args:
            model_variants: Dictionary of model variants
            num_calibration_samples: Number of samples to use for calibration
        """
        self.logger.info(f"Calibrating models using {num_calibration_samples} samples")
        
        # Get dataset for calibration
        dataset, _ = get_dataset(argparse.Namespace(dataset="CIFAR10"), self.config)
        dataloader = data.DataLoader(
            dataset, 
            batch_size=num_calibration_samples,
            shuffle=True,
            num_workers=1
        )
        
        # Get a single batch for calibration
        x, _ = next(iter(dataloader))
        x = x.to(self.device)
        x = data_transform(self.config, x)
        
        # Define the timesteps for calibration (spread across diffusion process)
        timesteps = torch.linspace(
            0, self.config.diffusion.num_diffusion_timesteps - 1, 10, 
            dtype=torch.long, device=self.device
        )
        
        # Calibrate each model
        for variant, model in model_variants.items():
            self.logger.info(f"Calibrating Variant {variant}")
            model.eval()
            
            # Run calibration forward passes
            with torch.no_grad():
                for t in tqdm(timesteps, desc=f"Calibrating Variant {variant}"):
                    t_batch = t.repeat(x.shape[0])
                    _ = model(x, t_batch)
            
            self.logger.info(f"Finished calibrating Variant {variant}")
    
    def generate_samples(self, model_variants, num_samples=100, batch_size=10):
        """
        Generate samples from all model variants
        
        Args:
            model_variants: Dictionary of model variants
            num_samples: Total number of samples to generate
            batch_size: Batch size for generation
        
        Returns:
            Dictionary mapping variant names to paths containing generated samples
        """
        self.logger.info(f"Generating {num_samples} samples for each variant")
        
        # Set up diffusion parameters
        betas = get_beta_schedule(
            beta_schedule=self.config.diffusion.beta_schedule,
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float().to(self.device)
        
        # Variables for sample generation
        sample_shape = (batch_size, self.config.data.channels, 
                        self.config.data.image_size, self.config.data.image_size)
        sample_paths = {}
        
        # Generate samples for each variant
        for variant, model in model_variants.items():
            model.eval()
            
            # Path to save samples
            sample_dir = os.path.join(self.variant_dirs[variant], 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            sample_paths[variant] = sample_dir
            
            # Generate samples in batches
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                if current_batch_size != batch_size:
                    sample_shape = (current_batch_size, self.config.data.channels,
                                   self.config.data.image_size, self.config.data.image_size)
                
                self.logger.info(f"Generating samples {i+1}-{i+current_batch_size} for Variant {variant}")
                
                # Sample from standard normal distribution as starting point
                x = torch.randn(sample_shape, device=self.device)
                
                # Progressively denoise samples
                with torch.no_grad():
                    # Loop from T to 0
                    for t in tqdm(reversed(range(self.config.diffusion.num_diffusion_timesteps)), 
                                 desc=f"Sampling (Variant {variant}, batch {i//batch_size + 1})"):
                        # Create timestep batch (all same timestep)
                        timesteps = torch.full((current_batch_size,), t, 
                                              device=self.device, dtype=torch.long)
                        
                        # Get model prediction and update sample
                        noise_pred = model(x, timesteps)
                        
                        # Simple DDPM sampling (could be replaced with more advanced schemes)
                        alpha = 1 - betas[t]
                        alpha_bar = torch.prod(alpha.to(alpha.device))
                        sigma = betas[t].sqrt()
                        
                        # Compute the mean for x_{t-1}
                        mean = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * noise_pred)
                        
                        # Add noise if t > 0, otherwise use mean directly
                        if t > 0:
                            noise = torch.randn_like(x)
                            x = mean + sigma * noise
                        else:
                            x = mean
                
                # Save generated samples
                for j in range(current_batch_size):
                    sample_idx = i + j
                    sample = inverse_data_transform(self.config, x[j].unsqueeze(0))
                    tvu.save_image(
                        sample, 
                        os.path.join(sample_dir, f'sample_{sample_idx:05d}.png')
                    )
            
            # Save a grid of samples for easy visualization
            grid_samples = []
            for j in range(min(100, num_samples)):
                img_path = os.path.join(sample_dir, f'sample_{j:05d}.png')
                if os.path.exists(img_path):
                    grid_samples.append(torch.from_numpy(
                        np.array(Image.open(img_path)).transpose(2, 0, 1)
                    ) / 255.0)
            
            if grid_samples:
                grid = tvu.make_grid(grid_samples, nrow=10)
                tvu.save_image(grid, os.path.join(self.variant_dirs[variant], 'sample_grid.png'))
        
        return sample_paths
    
    def compute_fid(self, sample_paths, real_images_path):
        """
        Compute FID score for all variants
        
        Args:
            sample_paths: Dictionary mapping variant names to sample directories
            real_images_path: Path to directory containing real images
        
        Returns:
            Dictionary mapping variant names to FID scores
        """
        if not FID_AVAILABLE:
            self.logger.warning("FID computation skipped - pytorch-fid not available")
            return {variant: float('nan') for variant in sample_paths}
        
        self.logger.info("Computing FID scores")
        fid_scores = {}
        
        for variant, sample_dir in sample_paths.items():
            self.logger.info(f"Computing FID for Variant {variant}")
            try:
                fid_value = calculate_fid_given_paths(
                    [real_images_path, sample_dir],
                    batch_size=50,
                    device=self.device,
                    dims=2048
                )
                fid_scores[variant] = fid_value
                self.logger.info(f"Variant {variant} FID: {fid_value:.4f}")
            except Exception as e:
                self.logger.error(f"Error computing FID for Variant {variant}: {e}")
                fid_scores[variant] = float('nan')
        
        return fid_scores
    
    def compute_clip_score(self, sample_paths, prompts):
        """
        Compute CLIP score for text-image alignment
        
        Args:
            sample_paths: Dictionary mapping variant names to sample directories
            prompts: List of text prompts to evaluate against
        
        Returns:
            Dictionary mapping variant names to CLIP scores
        """
        if not CLIP_AVAILABLE:
            self.logger.warning("CLIP score computation skipped - CLIP not available")
            return {variant: float('nan') for variant in sample_paths}
        
        self.logger.info("Computing CLIP scores")
        clip_scores = {}
        
        # Encode text prompts
        with torch.no_grad():
            text_features = []
            for prompt in prompts:
                text = clip.tokenize([prompt]).to(self.device)
                text_feature = self.clip_model.encode_text(text)
                text_features.append(text_feature)
            
            # Average text features if multiple prompts
            if len(text_features) > 1:
                text_features = torch.cat(text_features, dim=0)
                text_features = text_features.mean(dim=0, keepdim=True)
            else:
                text_features = text_features[0]
            
            # Normalize text features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute CLIP score for each variant
        for variant, sample_dir in sample_paths.items():
            self.logger.info(f"Computing CLIP score for Variant {variant}")
            try:
                # Get list of generated images
                image_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                             if f.endswith('.png') and f.startswith('sample_')]
                
                if not image_files:
                    self.logger.warning(f"No images found in {sample_dir}")
                    clip_scores[variant] = float('nan')
                    continue
                
                # Process images in batches to avoid OOM
                batch_size = 32
                similarity_scores = []
                
                for i in range(0, len(image_files), batch_size):
                    batch_files = image_files[i:i+batch_size]
                    
                    # Load and preprocess images
                    images = []
                    for img_path in batch_files:
                        image = self.clip_preprocess(Image.open(img_path)).unsqueeze(0)
                        images.append(image)
                    
                    images = torch.cat(images, dim=0).to(self.device)
                    
                    # Encode images and compute similarity
                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(images)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        
                        # Compute cosine similarity between image and text features
                        similarity = (100.0 * image_features @ text_features.T).squeeze()
                        similarity_scores.append(similarity)
                
                # Compute average CLIP score
                all_similarities = torch.cat(similarity_scores, dim=0)
                clip_score = all_similarities.mean().item()
                
                clip_scores[variant] = clip_score
                self.logger.info(f"Variant {variant} CLIP Score: {clip_score:.4f}")
            
            except Exception as e:
                self.logger.error(f"Error computing CLIP score for Variant {variant}: {e}")
                clip_scores[variant] = float('nan')
        
        return clip_scores
    
    def run_ablation_study(self, checkpoint_path, real_images_path=None, 
                         text_prompts=None, num_samples=100):
        """
        Run the complete ablation study
        
        Args:
            checkpoint_path: Path to pretrained model checkpoint
            real_images_path: Path to real images for FID computation
            text_prompts: List of text prompts for CLIP score computation
            num_samples: Number of samples to generate for evaluation
        
        Returns:
            Dictionary with all evaluation results
        """
        # Prepare model variants
        model_variants = self.prepare_model_variants()
        
        # Load pretrained weights
        self.load_pretrained_weights(model_variants, checkpoint_path)
        
        # Calibrate models
        self.calibrate_models(model_variants)
        
        # Generate samples
        sample_paths = self.generate_samples(model_variants, num_samples=num_samples)
        
        # Compute FID if real images path is provided
        fid_scores = {}
        if real_images_path:
            fid_scores = self.compute_fid(sample_paths, real_images_path)
        
        # Compute CLIP score if text prompts are provided
        clip_scores = {}
        if text_prompts:
            clip_scores = self.compute_clip_score(sample_paths, text_prompts)
        
        # Compile and save results
        results = {
            'fid': fid_scores,
            'clip': clip_scores,
            'sample_paths': sample_paths
        }
        
        # Save results to file
        with open(os.path.join(self.exp_dir, 'ablation_results.yaml'), 'w') as f:
            yaml.dump(results, f)
        
        # Log results
        self.logger.info("=== Ablation Study Results ===")
        self.logger.info("FID Scores (lower is better):")
        for variant, fid in fid_scores.items():
            self.logger.info(f"  Variant {variant}: {fid:.4f}")
        
        self.logger.info("CLIP Scores (higher is better):")
        for variant, clip in clip_scores.items():
            self.logger.info(f"  Variant {variant}: {clip:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Attention Layer Quantization Ablation Study")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--exp_dir", type=str, default="ablation_results", help="Directory for experiment results")
    parser.add_argument("--real_images", type=str, help="Path to real images for FID computation")
    parser.add_argument("--prompts", type=str, nargs="+", help="Text prompts for CLIP score computation")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()
    
    # Run ablation study
    ablation = AttentionQuantizationAblation(args.config, exp_dir=args.exp_dir)
    results = ablation.run_ablation_study(
        args.ckpt,
        real_images_path=args.real_images,
        text_prompts=args.prompts,
        num_samples=args.num_samples
    )
    
    # Print final summary
    print("\n=== Ablation Study Complete ===")
    print("FID Scores (lower is better):")
    for variant, fid in results['fid'].items():
        print(f"  Variant {variant}: {fid:.4f}")
    
    print("\nCLIP Scores (higher is better):")
    for variant, clip in results['clip'].items():
        print(f"  Variant {variant}: {clip:.4f}")
    
    print("\nSamples generated in:", args.exp_dir)


if __name__ == "__main__":
    main() 