#!/bin/bash
# Script to run the attention layer quantization ablation study

# Check for necessary packages
pip install clip pytorch-fid pillow tqdm pyyaml --quiet

# Default checkpoint path - change as needed
CHECKPOINT="configs/default_checkpoint.pt"

# Create config file for the ablation study if it doesn't exist
CONFIG="configs/ablation_config.yml"

# Real images for FID calculation
REAL_IMAGES="path/to/real/images"  # Update this path to your real dataset

# Run the ablation study
python ablation_study_attention_quantization.py \
    --config $CONFIG \
    --ckpt $CHECKPOINT \
    --exp_dir "results/attention_quantization_ablation" \
    --real_images $REAL_IMAGES \
    --prompts "high quality image" "realistic image" \
    --num_samples 100

echo "Ablation study complete. Results available in results/attention_quantization_ablation/" 