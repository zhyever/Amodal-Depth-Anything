import torch
import numpy as np
import torch.nn as nn

import warnings
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.utils import is_safetensors_available

from torch.nn.parameter import Parameter

if is_safetensors_available():
    from safetensors.torch import load_model as load_model_as_safetensor
    from safetensors.torch import save_model as save_model_as_safetensor

import os
import glob
from src.models.amodalsynthdrive.zoedepth.utils.config import get_config
from src.models.amodalsynthdrive.zoedepth.models.builder import build_model
from src.models.amodalsynthdrive.zoedepth.models.model_io import load_wts
from huggingface_hub import hf_hub_download

def load_ckpt(config, model, checkpoint_dir: str = "./checkpoints", ckpt_type: str = "best"):
    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]

    else:
        return model
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

def get_zoe_dc_model(vanilla: bool = False, ckpt_path: str = None, **kwargs):
    def ZoeD_N(midas_model_type="DPT_BEiT_L_384", vanilla=False, **kwargs):
        if midas_model_type != "DPT_BEiT_L_384":
            raise ValueError(f"Only DPT_BEiT_L_384 MiDaS model is supported for pretrained Zoe_N model, got: {midas_model_type}")

        zoedepth_config = get_config("zoedepth", "train", **kwargs)
        model = build_model(zoedepth_config)

        if vanilla:
            model.__setattr__("vanilla", True)
            return model
        else:
            model.__setattr__("vanilla", False)

        if zoedepth_config.add_depth_channel and not vanilla:
            model.core.core.pretrained.model.patch_embed.proj = torch.nn.Conv2d(
                model.core.core.pretrained.model.patch_embed.proj.in_channels+2,
                model.core.core.pretrained.model.patch_embed.proj.out_channels,
                kernel_size=model.core.core.pretrained.model.patch_embed.proj.kernel_size,
                stride=model.core.core.pretrained.model.patch_embed.proj.stride,
                padding=model.core.core.pretrained.model.patch_embed.proj.padding,
                bias=True)

        if ckpt_path is not None:
            assert os.path.exists(ckpt_path)
            zoedepth_config.__setattr__("checkpoint", ckpt_path)
        else:
            assert vanilla, "ckpt_path must be provided for non-vanilla model"

        model = load_ckpt(zoedepth_config, model)

        return model

    return ZoeD_N(vanilla=vanilla, ckpt_path=ckpt_path, **kwargs)

class InvisibleStitch(nn.Module, PyTorchModelHubMixin):
    def __init__(self, loss_stategy):
        super().__init__()
        
        self.loss_stategy = loss_stategy
        self.zoe_dc_model = get_zoe_dc_model(ckpt_path=hf_hub_download(repo_id="paulengstler/invisible-stitch", filename="invisible-stitch.pt"))
        
    def forward(self, x, invisible_mask=None, observation=None):
        depth_mask = (1 - invisible_mask.float())
        depth_mask = depth_mask > 0
        observation = observation * (1 - invisible_mask.float())


        # print(x.shape, observation.shape, depth_mask.shape)
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('Agg')
        # plt.subplot(2, 2, 1)
        # plt.imshow(x[1].squeeze().cpu().permute(1, 2, 0).numpy())
        # plt.subplot(2, 2, 2)
        # plt.imshow(observation[1].squeeze().cpu().numpy())
        # plt.subplot(2, 2, 3)
        # plt.imshow(depth_mask[1].squeeze().cpu().numpy())
        # plt.savefig("./work_dir/test.png")
        
        x = torch.cat([x, observation, depth_mask], dim=1)
        depth = self.zoe_dc_model(x)["metric_depth"]

        return depth
        
    def _save_pretrained(self, save_directory) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
    