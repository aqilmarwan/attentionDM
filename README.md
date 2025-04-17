# PTQ-AttnDM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/username/project/main)](https://github.com/username/project/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/username/project)](https://codecov.io/gh/username/project)
[![Version](https://img.shields.io/npm/v/project.svg)](https://www.npmjs.com/package/project)

## Overview

This is a pytorch implementation of the paper "PTQ-AttnDM: An Enhanced implementation of Post Training Quantisation with Self-attention on Diffusion Models".

## Prerequisites

- python>=3.8
- pytorch>=1.12.1
- torchvision>=0.13.0 
- other packages like numpy, tqdm and math

## Installation

Step-by-step installation instructions:


## Clone the repository
```bash
git clone https://github.com/aqilmarwan/attentionDM.git
cd attentionDM
```

## Configure environment variables from LDM

## Usage

### Basic Examples

### Training and Testing

The following experiments were performed in NVIDIA A500 with 24GB memory.

### Generate CIFAR-10 Images

You can run the following command to generate 50000 CIFAR-10 32*32 images in low bitwidths with differentiable group-wise quantization and active timestep selection.

```
sh sample_cifar.sh
```

### Calculate FID

After generation, you can run the following command to evaluate IS and FID.

```
python -m pytorch_fid <dataset path> <image path>
```

## Performance

Performance was conducted on NVIDIA A500 with 24GB memory with specified quantisation parameters settings.

## Class-conditional Image Generation
**Dataset:** ImageNet 256 × 256  

| Timesteps | Bit-width (W/A) | Method        | Size (MB) | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
|-----------|----------------|--------------|-----------|-------|--------|------|------------|
| 250       | 32/32          | `FP`         |           |       |        |      |            |
|           | 8/8            | `Q-Diffusion` |           |       |        |      |            |
|           |                | `PTQD`        |           |       |        |      |            |
|           |                | `PTQ4DM`      |           |       |        |      |            |
|           |                | **`TinyDM`**  |           |       |        |      |            |
|           | 4/8            | `Q-Diffusion` |           |       |        |      |            |
|           |                | `PTQD`        |           |       |        |      |            |
|           |                | `PTQ4DM`      |           |       |        |      |            |
|           |                | **`TinyDM`**  |           |       |        |      |            |
| 50        | 32/32          | `FP`         |           |       |        |      |            |
|           | 8/8            | `Q-Diffusion` |           |       |        |      |            |
|           |                | `PTQD`        |           |       |        |      |            |
|           |                | `PTQ4DM`      |           |       |        |      |            |
|           |                | **`TinyDM`**  |           |       |        |      |            |
|           | 4/8            | `Q-Diffusion` |           |       |        |      |            |
|           |                | `PTQD`        |           |       |        |      |            |
|           |                | `PTQ4DM`      |           |       |        |      |            |
|           |                | **`TinyDM`**  |           |       |        |      |            |

---

## Unconditional Image Generation
**Dataset:** LSUN-Bedrooms 256 × 256 LDM4  

| Timesteps | Bit-width (W/A) | Method        | Size (MB) | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ |
|-----------|----------------|--------------|-----------|-------|--------|------|------------|
| 250       | 32/32          | `FP`         |           |       |        |      |            |
|           | 8/8            | `Q-Diffusion` |           |       |        |      |            |
|           |                | `PTQD`        |           |       |        |      |            |
|           |                | `PTQ4DM`      |           |       |        |      |            |
|           |                | **`TinyDM`**  |           |       |        |      |            |
|           | 4/8            | `Q-Diffusion` |           |       |        |      |            |
|           |                | `PTQD`        |           |       |        |      |            |
|           |                | `PTQ4DM`      |           |       |        |      |            |
|           |                | **`TinyDM`**  |           |       |        |      |            |


## Image Synthesis Generation

## Roadmap
Current state of the project:

- [x] Results on state of the conditional. (current)
- [x] Results on un-conditional image benchmarking. (current)
- [x] Self attention mechanism in PTQ diffusion models
- [x] Dissertation writing of 3401 words.

Future possible/uncomplete plans for the project:

- [x] Explore Channel Wise Balancing Quantisation.
- [ ] Performance metrics
  - [ ] Performance w.r.t. the number of timestep groups
  - [ ] Performance w.r.t. different timestep sampling strategies for calibration set construction
  - [ ] Visualization of importance weight in differentiable search
  - [ ] Performance w.r.t. hyper-parameters λ and η:
  - [ ] Image synthesis/comparison on datasets. (LSUN Church, LSUN Bedroom, CIFAR-10 etc)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [DDIM](https://github.com/ermongroup/ddim)
- [DDPM](https://github.com/hojonathanho/diffusion)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [PTQ4DM](https://github.com/42Shawn/PTQ4DM)
- [Q-diffusion](https://github.com/Xiuyu-Li/q-diffusion)
- [guided-diffusion](https://github.com/openai/guided-diffusion)
- [Self-Attention-Guidance](https://github.com/SusungHong/Self-Attention-Guidance)
