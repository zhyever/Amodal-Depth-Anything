import torch
import numpy as np
import torch.nn as nn

import warnings
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.utils import is_safetensors_available

import timm
from torch.nn.parameter import Parameter

from src.models.amodalsynthdrive.deeplab import resize, Conv2DModule, ASPPModule, ASPPHead, UpSample, DepthPredictionHead, mViT
from .depth_anything_v2.dpt import DepthAnythingV2

if is_safetensors_available():
    from safetensors.torch import load_model as load_model_as_safetensor
    from safetensors.torch import save_model as save_model_as_safetensor
        
class AmodalDAv2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, guide_type='image+mask', loss_stategy='invisible_part', encoder='vitg', pretrained=True,):
        super().__init__()
        
        # self.encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)
        # self.register_buffer("pixel_mean", torch.Tensor(self.encoder.default_cfg['mean']).view(-1, 1, 1), False)
        # self.register_buffer("pixel_std", torch.Tensor(self.encoder.default_cfg['std']).view(-1, 1, 1), False)
        self.guide_type = guide_type
        self.encoder = encoder
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'pretrain': './work_dir/ckp/depth_anything_v2_vits.pth'},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'pretrain': './work_dir/ckp/depth_anything_v2_vitb.pth'},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'pretrain': './work_dir/ckp/depth_anything_v2_vitl.pth'}}
                
        self.encoder = DepthAnythingV2(
            encoder=self.encoder, 
            features=model_configs[encoder]['features'], 
            out_channels=model_configs[encoder]['out_channels'],
            guide_type=guide_type, 
            loss_stategy=loss_stategy)
        
        self.pretrained = pretrained
        # if self.pretrained:
        #     print(self.encoder.load_state_dict(torch.load(model_configs[encoder]['pretrain'], map_location='cpu'), strict=False))
        # else:
        #     print(self.encoder.pretrained.load_state_dict(torch.load('/home/liz0l/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth', map_location='cpu'), strict=False))
        #     pass
            
        self.register_buffer("pixel_mean", torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), False)
        
        
        # add one more embedding layer for guidance mask
        if self.guide_type != 'none':
            _weight = self.encoder.pretrained.patch_embed_guidance.proj.weight.clone()  # [320, 4, 3, 3]
            _bias = self.encoder.pretrained.patch_embed_guidance.proj.bias.clone()  # [320]
            _weight = torch.zeros_like(_weight)  # Keep selected channel(s)
            _bias = torch.zeros_like(_bias)  # Keep selected channel(s)
            self.encoder.pretrained.patch_embed_guidance.proj.weight = Parameter(_weight)
            self.encoder.pretrained.patch_embed_guidance.proj.bias = Parameter(_bias)
        
        
    def forward(self, x, guide_rgb=None, guide_mask=None, observation=None):
        x = (x - self.pixel_mean) / self.pixel_std
        
        if self.guide_type == 'image+mask+observation':
            guide = torch.cat([guide_rgb, guide_mask, observation], dim=1)
        elif self.guide_type == 'image+mask':
            guide = torch.cat([guide_rgb, guide_mask], dim=1)
        elif self.guide_type == 'image+observation':
            guide = torch.cat([guide_rgb, observation], dim=1)
        elif self.guide_type == 'mask+observation':
            guide = torch.cat([guide_mask, observation], dim=1)
        elif self.guide_type == 'observation':
            guide = observation
        elif self.guide_type == 'mask':
            guide = guide_mask
        elif self.guide_type == 'none':
            guide = None
        else:
            raise NotImplementedError
        
        pred_sigmoid = self.encoder(x, guide)
        return pred_sigmoid
        
    def _save_pretrained(self, save_directory) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
    
# if __name__ == '__main__':
#     model = ADDeepLabDAv2().cuda()
#     a = torch.rand((1, 3, 518, 518)).cuda()
#     b = torch.rand((1, 1, 518, 518)).cuda()
#     pred1, pred2 = model(a, b)
#     print(pred1.shape, pred2.shape)
    