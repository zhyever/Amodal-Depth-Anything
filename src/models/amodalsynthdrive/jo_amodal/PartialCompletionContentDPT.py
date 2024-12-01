import multiprocessing as mp
import argparse
import yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# from deocclusion import utils
import numpy as np
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from src.models.amodalsynthdrive.jo_amodal import amodal_utils
from src.models.amodalsynthdrive.jo_amodal.dpt import DPTDepthModel
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.utils import is_safetensors_available

if is_safetensors_available():
    from safetensors.torch import load_model as load_model_as_safetensor
    from safetensors.torch import save_model as save_model_as_safetensor
    
class PartialCompletionContentDPT(nn.Module, PyTorchModelHubMixin):

    def __init__(self, loss_stategy):
        super(PartialCompletionContentDPT, self).__init__()
        self.loss_stategy = loss_stategy
        # self.model = DPTDepthModel(
        #     backbone="vitl16_384",
        #     non_negative=True,
        #     enable_attention_hooks=False)
        self.model = DPTDepthModel(
            backbone="vitl16_384",
            non_negative=False,
            enable_attention_hooks=False,
            invert=False)
        
        self.sigmoid = nn.Sigmoid()
        
        ckp = torch.load('/ibex/ai/home/liz0l/codes/depth-fm/work_dir/weights/amodal_depth.pth.tar', map_location=torch.device("cpu"))
        print(self.model.load_state_dict(ckp['state_dict']))

    def forward(self, x, guide_rgb=None, guide_mask=None, observation=None):
        
        output = self.model(torch.cat([x, guide_mask], dim=1), guide_mask)
        output = self.sigmoid(output)
        return output
    
    def _save_pretrained(self, save_directory) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

        
