# Attention Layer Quantization Precision Ablation Study

This ablation study examines the impact of quantization precision on different components of a diffusion model, with a focus on attention layers. The study tests the hypothesis that attention layers are more sensitive to quantization than other model components (e.g., convolutional layers).

## Study Design

We create four model variants with different quantization strategies:

1. **Variant A**: Uniform 4-bit quantization (baseline)
   - All components (conv layers and attention) quantized to 4-bit precision

2. **Variant B**: 4-bit for conv layers, 8-bit for attention layers
   - Convolutional layers: 4-bit precision
   - Attention layers (query, key, value, output projections): 8-bit precision

3. **Variant C**: 8-bit for conv layers, 4-bit for attention layers
   - Convolutional layers: 8-bit precision
   - Attention layers (query, key, value, output projections): 4-bit precision

4. **Variant D**: Uniform 8-bit quantization
   - All components quantized to 8-bit precision (high quality reference)

## Evaluation Metrics

Each variant is evaluated using:

1. **FID (Fréchet Inception Distance)**: Measures overall image quality and distribution similarity to real images
2. **CLIP Score**: Measures text-image alignment (if applicable)
3. **Visual inspection**: For qualitative assessment of sample quality

## Expected Outcomes

If the hypothesis is correct, we expect to observe:
- Variant B should perform significantly better than Variant A despite using only slightly more bits overall
- Variant C should perform worse than Variant D, with quality closer to Variant A
- This would demonstrate that prioritizing precision for attention layers is more important than for convolutional layers

## Setup and Requirements

The ablation study requires:

- PyTorch
- CLIP (for computing CLIP scores)
- pytorch-fid (for computing FID scores)
- A pretrained diffusion model checkpoint

Additional dependencies will be installed by the setup script.

## Running the Ablation Study

1. First, make sure you have a pretrained model checkpoint available.

2. Update the paths in the `run_attention_ablation.sh` script:
   - `CHECKPOINT`: Path to your pretrained model checkpoint
   - `REAL_IMAGES`: Path to a directory containing real images for FID calculation

3. Run the ablation study:
   ```bash
   chmod +x run_attention_ablation.sh
   ./run_attention_ablation.sh
   ```

4. The results will be saved in `results/attention_quantization_ablation/`:
   - Generated samples from each variant
   - FID scores
   - CLIP scores
   - Visual comparison grids

## Interpreting Results

The study provides empirical evidence for the importance of precision in attention mechanisms:

- If Variant B (4-bit conv, 8-bit attention) performs nearly as well as Variant D (full 8-bit), it confirms that attention requires higher precision.
- If Variant C (8-bit conv, 4-bit attention) performs poorly despite using more bits overall than Variant A, it suggests that lower precision in attention cannot be compensated by higher precision elsewhere.

These insights can guide more efficient model quantization strategies, focusing higher precision on the most sensitive components for better performance/size tradeoffs.

## Customization

You can customize the ablation study by:

- Modifying the config file (`configs/ablation_config.yml`) to use different model architectures
- Changing the bit-widths in the `prepare_model_variants` method in `ablation_study_attention_quantization.py`
- Adding additional evaluation metrics in the `run_ablation_study` method

## Troubleshooting

If you encounter issues:

1. Ensure you have enough GPU memory for all model variants
2. Try reducing batch sizes or number of samples if memory is constrained
3. Check that the model checkpoint is compatible with the configuration file
4. Verify that all dependencies are correctly installed

For more detailed analysis, check the log file in the results directory. 