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

## Acknowledgements

- [DDIM](https://github.com/ermongroup/ddim)
- [DDPM](https://github.com/hojonathanho/diffusion)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [PTQ4DM](https://github.com/42Shawn/PTQ4DM)
- [Q-diffusion](https://github.com/Xiuyu-Li/q-diffusion)
- [guided-diffusion](https://github.com/openai/guided-diffusion)
- [Self-Attention-Guidance](https://github.com/SusungHong/Self-Attention-Guidance)
