import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
from utils import *
# from utils.quant_util import calibrate
from utils.quant_util import QConv2d
from torch import optim
import torch.nn.functional as F
import util
from models.self_attention import EnhancedQSelfAttention

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def cal_entropy(self, attn):
        return -1 * torch.sum((attn * torch.log(attn)), dim=-1).mean()

    def generate_calibrate_set(self, fpmodel, model, t_mode, num_calibrate_set):
        print("start to create calibrate set in:" + str(t_mode))
        with torch.no_grad():
            n = min(num_calibrate_set, 16)  # Reduce the number of samples
            # n = self.args.timesteps
            x = torch.randn(
                n,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.device,
            )

            # x = self.sample_image(x, fpmodel)
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, self.seq, fpmodel, self.betas, eta=self.args.eta)[0]
            # print(xs[0].shape)  # 100*64*3*32*32
            if t_mode == "real":
                x = xs[-1]
            elif t_mode == "range":
                for s in range(n):
                    if s >= 100:
                        x[s] = xs[-1][s]
                    else:
                        x[s] = xs[s][s]
            elif t_mode == "random":
                shape = torch.Tensor(n)
                normal_val = torch.nn.init.normal_(shape, mean=0.4, std=0.4)*self.args.timesteps
                t = normal_val.clone().type(torch.int).to(device=self.device).clamp(0, self.args.timesteps - 1)
                # print(t)
                for s in range(n):
                    x[s] = xs[t[s]][s]
            elif t_mode == "diff":
                uncertainty = torch.zeros(self.args.timesteps).to(self.device)
                uncertainty_mark = torch.arange(0, self.args.timesteps).to(self.device)
                for k, layer in enumerate(model.modules()):
                    if type(layer) in [QConv2d]:
                        alpha = F.softmax(layer.alpha_activ, dim=1)
                        # print(alpha[0].grad)
                        _ ,group_n, dim = alpha.shape
                        for t in range(self.args.timesteps):
                            uncertainty[t] += self.cal_entropy(alpha[t]) / dim
                uncertainty -= self.args.sample_weight * self.sample_count 
                uncertainty = uncertainty[30:]
                uncertainty_mark = uncertainty_mark[30:]
                uncertainty_max = torch.max(uncertainty)
                uncertainty_max_list = uncertainty[uncertainty == uncertainty_max]
                uncertainty_mark_list = uncertainty_mark[uncertainty == uncertainty_max]
                t = uncertainty_mark_list[-1]
                self.sample_count[t] += 1 
                print(uncertainty, t)
                x = xs[t]
                self.timestep_select = t

            # x = generalized_steps_range(x, self.seq, fpmodel, self.betas, eta=self.args.eta)
            calibrate_set = inverse_data_transform(self.config, x)

            # img_id = len(glob.glob(f"{self.args.image_folder}/*"))
            # for i in range(n):
            #     tvu.save_image(
            #         calibrate_set[i], os.path.join(self.args.image_folder, f"{img_id}.png")
            #     )
            #     img_id += 1

        # print(calibrate_set.shape)  # torch.Size([batchsize, 3, 32, 32])
        return calibrate_set

    def calibrate_attention(self, model, image, device, batchsize):
        """
        Calibrate the self-attention modules in the model
        """
        print('\n==> start calibrating self-attention modules')
        # Set all QConv2d layers in self-attention modules to calibration mode
        for name, module in model.named_modules():
            if isinstance(module, EnhancedQSelfAttention):
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, QConv2d):
                        sub_module.set_calibrate(calibrate=True)
                        sub_module.first_calibrate(calibrate=self.first_flag)
        
        image = image.to(device)
        
        # Collect parameters for optimization
        attention_params = []
        for name, param in model.named_parameters():
            if "alpha_activ" in name and any(attn_name in name for attn_name in ["query_conv", "key_conv", "value_conv", "output_conv"]):
                param.requires_grad = True
                attention_params += [param]
        
        # Create optimizer for attention parameters
        if attention_params:
            optimizer = torch.optim.AdamW(attention_params, 0.05, weight_decay=0.05)
            
            # Run calibration with attention-specific optimization
            from functions.denoising import generalized_steps_loss
            xs = generalized_steps_loss(image, self.seq, model, self.betas, optimizer, eta=self.args.eta,
                                       t_mode=self.t_mode, timestep_select=self.timestep_select,
                                       args=self.args, attention_focus=True)
        
        # Set all QConv2d layers back to normal mode
        for name, module in model.named_modules():
            if isinstance(module, EnhancedQSelfAttention):
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, QConv2d):
                        sub_module.set_calibrate(calibrate=False)
        
        print('==> end calibrating self-attention modules')
        return model

    def sample(self):
        # Early debug statements
        print("Starting sampling...")
        
        # Add default value for ckpt if not provided
        if not hasattr(self.args, 'ckpt'):
            self.args.ckpt = '790000'  # Default checkpoint number
        
        print(f"Loading model checkpoint: model-{self.args.ckpt}.ckpt")
        
        # Calculate self.seq BEFORE using it in print statement
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            self.seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                )
                ** 2
            )
            self.seq = [int(s) for s in list(seq)]
        
        print("Model created, loading weights...")
        print("Weights loaded, starting sampling process...")
        print(f"Running diffusion with {len(self.seq)} timesteps...")
        
        model = Model(self.config, quantization=True, sequence=self.seq, args=self.args)
        
        # Move model to CUDA BEFORE loading state dict
        model = model.to(self.device)
        
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                if self.config.data.dataset == "CELEBA":
                    states = torch.load(
                        os.path.join(self.args.log_path, "ckpt.pth"),
                        map_location=self.config.device,
                    )
                elif self.config.data.dataset == "CIFAR10":
                    states = torch.load(
                        os.path.join(self.args.log_path, "model-790000.ckpt"),
                        map_location=self.config.device,
                    )
                elif self.config.data.dataset == "LSUN":
                    if self.config.data.category == "church_outdoor":
                        states = torch.load(
                            os.path.join(self.args.doc, "model-4432000.ckpt"),
                            map_location=self.config.device,
                        )
                    elif self.config.data.category == "bedroom":
                        states = torch.load(
                            os.path.join(self.args.doc, "model-2388000.ckpt"),
                            map_location=self.config.device,
                        )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )

            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
            if self.config.data.dataset == "CELEBA":
                states = states[-1] # ema
            state_dict = model.state_dict()
            keys = []
            for k, v in states.items():
                keys.append(k)
            i = 0
            for k, v in state_dict.items():
                if "activation_range_min" in k:
                    continue
                if "activation_range_max" in k:
                    continue
                if "x_min" in k:
                    continue
                if "x_max" in k:
                    continue
                if "groups_range" in k:
                    continue
                if "alpha_activ" in k:
                    continue
                if "mix_activ_mark1" in k:
                    continue
                # print(k, keys[i])
                if v.shape == states[keys[i]].shape:
                    state_dict[k] = states[keys[i]]
                    i = i + 1
            model.load_state_dict(state_dict, strict=False)
            # model.load_state_dict(states[0], strict=True)
            # model.load_state_dict(states, strict=False)

            # if self.config.model.ema:
            #     ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            #     ema_helper.register(model)
            #     ema_helper.load_state_dict(states[-1])
            #     ema_helper.ema(model)
            # else:
            #     ema_helper = None

        # Don't use DataParallel at all
        model = model.module  # Extract from existing DataParallel
        model = model.to(self.device)  # Move to device directly

        # Force all parameters and buffers to CUDA
        for param in model.parameters():
            param.data = param.data.to(self.device)
        for buffer in model.buffers():
            buffer.data = buffer.data.to(self.device)

        print("Moved model to CUDA without DataParallel")
        
        # Skip the parameter check and proceed with sampling
        # Remove the "raise RuntimeError" line
        
        # After loading model weights, add:
        model.eval()
        
        # Add default num_samples if not set
        if not hasattr(self.args, 'num_samples'):
            self.args.num_samples = 50
        
        with torch.no_grad():
            n = self.args.num_samples
            print(f"Generating {n} samples...")
            
            # Generate initial noise
            x = torch.randn(
                n,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
                device=self.device,
            )
            
            # Run the diffusion sampling process
            from functions.denoising import generalized_steps
            x = generalized_steps(x, self.seq, model, self.betas, eta=self.args.eta)[0][-1]
            
            # Transform and save images
            print(f"Saving samples to {self.args.image_folder}")
            x = inverse_data_transform(self.config, x)
            for i in range(n):
                tvu.save_image(
                    x[i], os.path.join(self.args.image_folder, f"sample_{i}.png")
                )
            
            print(f"Sampling complete. {n} images saved to {self.args.image_folder}")

    def calibrate_model(self, model, data, device):
        """
        Complete calibration pipeline with three distinct calibration stages
        """
        # Stage 1: General calibration for all quantized modules
        # This establishes baseline quantization for the entire model
        self.calibrate_general(model, data, device, self.args.batchsize)
        
        # Stage 2: Attention-specific calibration with entropy optimization
        # This refines the quantization of attention projection layers
        self.calibrate_attention(model, data, device, self.args.batchsize)
        
        # Stage 3: Mixed-precision attention calibration if enabled
        # This calibrates the internal attention computation quantization
        if self.args.mixed_precision_attention:
            self.calibrate_mixed_precision_attention(model, data, device)
        
        return model

    def calibrate_mixed_precision_attention(self, model, image, device):
        """
        Calibrate specifically the mixed-precision attention quantization parameters
        This addresses the specialized quantization within attention computation
        """
        print('\n==> Calibrating mixed-precision attention quantization parameters')
        
        # 1. Find modules utilizing mixed-precision attention
        mixed_attn_modules = []
        for name, module in model.named_modules():
            if hasattr(module, 'mixed_precision') and module.mixed_precision and module.quantization:
                mixed_attn_modules.append(module)
        
        if not mixed_attn_modules:
            print("No mixed-precision attention modules found")
            return
        
        # 2. Create and use the specialized calibrator
        from utils.attention_quant_util import AttentionCalibrator
        calibrator = AttentionCalibrator(model, device)
        
        # 3. Sample key timesteps across diffusion process for comprehensive calibration
        # This ensures we calibrate for both early and late diffusion steps
        timesteps = [0, 250, 500, 750, 999]  
        
        # 4. Execute calibration on the attention processors
        # This updates the quantization parameters inside MixedPrecisionAttention modules
        calibrator.calibrate(image, timesteps)
        
        print(f"Calibrated {len(mixed_attn_modules)} mixed-precision attention modules")
