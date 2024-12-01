import torch
import einops
import numpy as np
import torch.nn as nn
from torch import Tensor
from functools import partial
from torchdiffeq import odeint

import os
from pathlib import Path
from huggingface_hub.utils import EntryNotFoundError
from huggingface_hub.file_download import hf_hub_download
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, get_args

from .unet.openaimodel import UNetModel
from diffusers import AutoencoderKL
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE, PYTORCH_WEIGHTS_NAME
from huggingface_hub.utils import is_safetensors_available

from torch.nn.parameter import Parameter
from src.models.depthfm.unet.util import conv_nd

if is_safetensors_available():
    from safetensors.torch import load_model as load_model_as_safetensor
    from safetensors.torch import save_model as save_model_as_safetensor


def exists(val):
    return val is not None


class DepthFMAmodal(nn.Module, PyTorchModelHubMixin):
    def __init__(self, ckpt_path: str, guide_type):
        super().__init__()
        vae_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
        self.scale_factor = 0.18215

        # set with checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # print('noising_step: {}; ldm_hp: {}'.format(ckpt['noising_step'], ckpt['ldm_hparams'])) 
        # noising_step: 400; ldm_hp: {'image_size': 32, 'in_channels': 8, 'out_channels': 4, 'model_channels': 320, 'attention_resolutions': [4, 2, 1], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4], 'num_heads': 8, 'use_spatial_transformer': True, 'use_linear_in_transformer': True, 'transformer_depth': 1, 'context_dim': 1024, 'legacy': False}
        self.noising_step = ckpt['noising_step']
        self.empty_text_embed = ckpt['empty_text_embedding'] # it would be a tensor
        self.model = UNetModel(**ckpt['ldm_hparams'])
        self.model.load_state_dict(ckpt['state_dict'])
        
        self.guide_type = guide_type
        if guide_type == 'image+mask+observation':
            additional_dim = 4 + 1 + 1
        elif guide_type == 'image+mask':
            additional_dim = 4 + 1
        elif guide_type == 'image+observation':
            additional_dim = 4 + 1
        elif guide_type == 'mask+observation':
            additional_dim = 1 + 1
        elif guide_type == 'mask':
            additional_dim = 1
        elif guide_type == 'observation':
            additional_dim = 1
        elif guide_type == 'image':
            additional_dim = 4
        elif guide_type == 'none':
            additional_dim = 0
        else:
            raise NotImplementedError
        
        # input channel from 8 to 9
        _weight = self.model.input_blocks[0][0].weight.clone()  # [320, 8, 3, 3]
        _bias = self.model.input_blocks[0][0].bias.clone()  # [320]
        
        __weight = torch.zeros((320, 8 + additional_dim, 3, 3))
        __weight[:, :8, :, :] = _weight
        
        _n_convin_out_channel = self.model.input_blocks[0][0].out_channels
        _new_conv_in = conv_nd(2, 8 + additional_dim, _n_convin_out_channel, 3, padding=1)
        _new_conv_in.weight = Parameter(__weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.input_blocks[0][0] = _new_conv_in
        self.model.in_channels = 8 + additional_dim
        # self.model.in_channels = 16
    
    def _save_pretrained(self, save_directory) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        model_to_save = model_to_save.model
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
    
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            model.model = cls._load_as_safetensor(model.model, model_file, map_location, strict)
            return model
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                model.model = cls._load_as_safetensor(model.model, model_file, map_location, strict)
                return model
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                model.model = cls._load_as_pickle(model.model, model_file, map_location, strict)
                return model
            
    def ode_fn(self, t: Tensor, x: Tensor, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        return self.model(x=x, t=t, **kwargs)
    
    def generate(self, z: Tensor, num_steps: int = 4, n_intermediates: int = 0, **kwargs):
        """
        ODE solving from z0 (ims) to z1 (depth).
        """
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0 / num_steps))
        
        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        t = torch.linspace(0, 1, n_intermediates + 2, device=z.device, dtype=z.dtype)
        # t = torch.tensor([0., 1.], device=z.device, dtype=z.dtype)

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, **kwargs)
        
        ode_results = odeint(ode_fn, z, t, **ode_kwargs)
        
        if n_intermediates > 0:
            return ode_results
        return ode_results[-1]
    
    def forward(self, ims, num_steps=4, ensemble_size=1, mode='train', depth=None, guide_mask=None, rand_num_generator=None, guide_rgb=None, observation=None):
        """
        Args:
            ims: Tensor of shape (b, 3, h, w) in range [-1, 1]
        Returns:
            depth: Tensor of shape (b, 1, h, w) in range [0, 1]
        """
        
        with torch.no_grad():
            # Encode image
            rgb_latent = self.encode(ims, sample_posterior=False)  # [B, 4, h, w]
            # empty text guide
            conditioning = torch.tensor(self.empty_text_embed).to(rgb_latent.device).repeat(rgb_latent.shape[0], 1, 1)
            # NOTE: add one more channel
            
            if self.guide_type == 'image+mask+observation':
                guide_rgb_latent = self.encode(guide_rgb, sample_posterior=False)
                # guide_mask_latent = self.encode((guide_mask*2-1).repeat(1, 3, 1, 1), sample_posterior=False)
                # guide_observation_latent = self.encode((observation*2-1).repeat(1, 3, 1, 1), sample_posterior=False)
                guide_mask_latent = F.interpolate(guide_mask.float(), size=(guide_rgb_latent.shape[2], guide_rgb_latent.shape[3]))
                guide_observation_latent = F.interpolate(observation.float(), size=(guide_rgb_latent.shape[2], guide_rgb_latent.shape[3]))
                guide = torch.cat([guide_rgb_latent, guide_mask_latent, guide_observation_latent], dim=1)
                rgb_mask_latent = torch.cat([rgb_latent, guide], dim=1)
            elif self.guide_type == 'image+mask':
                guide_rgb_latent = self.encode(guide_rgb, sample_posterior=False)
                guide_mask_latent = F.interpolate(guide_mask.float(), size=(guide_rgb_latent.shape[2], guide_rgb_latent.shape[3]))
                guide = torch.cat([guide_rgb_latent, guide_mask_latent], dim=1)
                rgb_mask_latent = torch.cat([rgb_latent, guide], dim=1)
            elif self.guide_type == 'image+observation':
                guide_rgb_latent = self.encode(guide_rgb, sample_posterior=False)
                guide_observation_latent = F.interpolate(observation.float(), size=(guide_rgb_latent.shape[2], guide_rgb_latent.shape[3]))
                guide = torch.cat([guide_rgb_latent, guide_observation_latent], dim=1)
                rgb_mask_latent = torch.cat([rgb_latent, guide], dim=1)
            elif self.guide_type == 'mask+observation':
                guide_mask_latent = F.interpolate(guide_mask.float(), size=(rgb_latent.shape[2], rgb_latent.shape[3]))
                guide_observation_latent = F.interpolate(observation.float(), size=(rgb_latent.shape[2], rgb_latent.shape[3]))
                guide = torch.cat([guide_mask_latent, guide_observation_latent], dim=1)
                rgb_mask_latent = torch.cat([rgb_latent, guide], dim=1)
            elif self.guide_type == 'mask':
                guide_mask_latent = F.interpolate(guide_mask.float(), size=(rgb_latent.shape[2], rgb_latent.shape[3]))
                guide = guide_mask_latent
                rgb_mask_latent = torch.cat([rgb_latent, guide], dim=1)
            elif self.guide_type == 'observation':
                guide_observation_latent = F.interpolate(observation.float(), size=(rgb_latent.shape[2], rgb_latent.shape[3]))
                guide =  guide_observation_latent
                rgb_mask_latent = torch.cat([rgb_latent, guide], dim=1)
            elif self.guide_type == 'none':
                rgb_mask_latent = rgb_latent
            else:
                raise NotImplementedError
            
            # rgb_mask_latent = torch.cat([rgb_latent, guide_mask_down], dim=1)
            # rgb_mask_latent = torch.cat([rgb_latent, guide_mask_latent, guide_rgb_latent], dim=1)
            # rgb_mask_latent = torch.cat([rgb_latent, guide], dim=1)
            
        if mode == 'train':
            with torch.no_grad():
                # set other parts as -1
                # depth[torch.logical_not(guide_mask)] = -1
                depth = (1 - depth) * 2 - 1
                gt_depth_latent = self.encode_depth(depth, sample_posterior=False)
                
                x_1 = gt_depth_latent
                timesteps = torch.ones(rgb_latent.shape[0], 1, 1, 1, device=rgb_latent.device)
                x_0 = q_sample(rgb_latent, t = (timesteps*self.noising_step).cpu().numpy())
                timesteps = torch.randint(
                    0, self.noising_step,
                    (rgb_latent.shape[0], 1, 1, 1),
                    device=rgb_latent.device,
                    generator=rand_num_generator) / self.noising_step
                x_t = (1 - timesteps) * x_0 + timesteps * x_1
                target = x_1 - x_0
            
            model_pred = self.model(x_t, timesteps.squeeze(), context=rgb_mask_latent, context_ca=conditioning)
            return model_pred, target

        else:
            
            with torch.no_grad():
                x_source = rgb_latent
                x_source = q_sample(x_source, self.noising_step)    

                # solve ODE
                depth_z = self.generate(x_source, num_steps=num_steps, context=rgb_mask_latent, context_ca=conditioning)
                depth = self.decode(depth_z)
                depth = depth.mean(dim=1, keepdim=True)

                if ensemble_size > 1:
                    depth = depth.mean(dim=0, keepdim=True)

                
                depth = torch.clamp((depth + 1) / 2, min=0, max=1) # from [-1, 1] to [0, 1]
                depth = 1 - depth # reverse the depth
                # depth = per_sample_min_max_normalization(depth.exp())

            return depth
    
    @torch.no_grad()
    def predict_depth(self, ims: Tensor, num_steps: int = 4, ensemble_size: int = 1):
        """ Inference method for DepthFM. """
        return self.forward(ims, num_steps, ensemble_size)
    
    @torch.no_grad()
    def encode(self, x: Tensor, sample_posterior: bool = True):
        posterior = self.vae.encode(x)
        if sample_posterior:
            z = posterior.latent_dist.sample()
        else:
            z = posterior.latent_dist.mode()
        # normalize latent code
        z = z * self.scale_factor
        return z
    
    @torch.no_grad()
    def decode(self, z: Tensor):
        z = 1.0 / self.scale_factor * z
        return self.vae.decode(z).sample

    def encode_depth(self, depth_in, sample_posterior: bool = True):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.encode(stacked, sample_posterior)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def cosine_alpha_bar(t):
    return sigmoid(cosine_log_snr(t))


def q_sample(x_start: torch.Tensor, t: int, noise: torch.Tensor = None, n_diffusion_timesteps: int = 1000):
    """
    Diffuse the data for a given number of diffusion steps. In other
    words sample from q(x_t | x_0).
    """
    dev = x_start.device
    dtype = x_start.dtype

    if noise is None:
        noise = torch.randn_like(x_start)
    
    alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
    alpha_bar_t = torch.tensor(alpha_bar_t).to(dev).to(dtype)
    
    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)
