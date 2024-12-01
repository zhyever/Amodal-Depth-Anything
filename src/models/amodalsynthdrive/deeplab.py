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

if is_safetensors_available():
    from safetensors.torch import load_model as load_model_as_safetensor
    from safetensors.torch import save_model as save_model_as_safetensor

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Conv2DModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=bias)
        
        if self.with_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        if self.with_activation:
            if act_cfg['type'] == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_cfg['type'] == 'ELU':
                self.act = nn.ELU(inplace=True)
            else:
                raise NotImplementedError(f"Activation type {act_cfg['type']} is not implemented.")
            # self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.act(x)
        return x
    
class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, dilations, in_channels, channels):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        for dilation in dilations:
            self.append(
                Conv2DModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    norm_cfg=None))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


class ASPPHead(nn.Module):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, in_channels, channels, out_channel, dilations=(1, 6, 12, 18), **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2DModule(in_channels, channels, 1))
        self.aspp_modules = ASPPModule(
            dilations,
            in_channels,
            channels)
        self.bottleneck = Conv2DModule(
            (len(dilations) + 1) * channels,
            out_channel,
            3,
            padding=1,)

    def _forward_feature(self, x):
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=True)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        return output

class UpSample(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = Conv2DModule(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.convB = Conv2DModule(output_features, output_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))
    

class DepthPredictionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = Conv2DModule(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.conv_2 = Conv2DModule(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.pred = Conv2DModule(in_channels, 1, kernel_size=3, stride=1, padding=1, act_cfg=None)
        
    def forward(self, x):
        ym = self.conv_1(x)
        x = F.interpolate(ym, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_2(x)
        x = self.pred(x)
        return ym, x

class mViT(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=128, num_heads=4,):
        super(mViT, self).__init__()
        self.patch_size = patch_size
        self.patch_x = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.patch_ym = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        decoder_layers = nn.TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=4)
        
    def forward(self, x, ym):
        n, c, h, w = x.size()
        x_pf = self.patch_x(x.clone()).flatten(2).permute(2, 0, 1)
        ym_pf = self.patch_ym(ym.clone()).flatten(2).permute(2, 0, 1)
        updated_feat = self.transformer_decoder(x_pf, ym_pf)
        updated_feat = updated_feat.permute(1, 2, 0).view(n, -1, h // self.patch_size, w // self.patch_size)
        return updated_feat
        
class ADDeepLab(nn.Module, PyTorchModelHubMixin):
    def __init__(self, encoder_name='convnext_xlarge', channels=[256, 512, 1024, 2048], up_sample_channels=[128, 256, 512, 1024]):
        super().__init__()
        
        self.encoder = timm.create_model(encoder_name, pretrained=True, features_only=True)
        self.register_buffer("pixel_mean", torch.Tensor(self.encoder.default_cfg['mean']).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.encoder.default_cfg['std']).view(-1, 1, 1), False)
        self.aspp_head = ASPPHead(channels[-1], channels=512, out_channel=channels[-1], dilations=(1, 12, 24, 36))
        
        # decoder part
        self.conv_list = nn.ModuleList()
        for index, (in_channel, up_channel) in enumerate(
                zip(channels[::-1], up_sample_channels[::-1])):
            if index == 0:
                self.conv_list.append(
                    Conv2DModule(
                        in_channels=in_channel,
                        out_channels=up_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ))
            else:
                self.conv_list.append(
                    UpSample(skip_input=in_channel + up_channel_temp,
                                output_features=up_channel))

            # save earlier fusion target
            up_channel_temp = up_channel
        
        # head part
        self.patch_size = 8
        self.visible_depth_head = DepthPredictionHead(up_sample_channels[0])
        self.cross_att_layer = mViT(up_sample_channels[0], patch_size=self.patch_size, embedding_dim=up_sample_channels[0], num_heads=8)
        self.in_visible_depth_head = DepthPredictionHead(up_sample_channels[0])
        
        # replace the first layer to accept 8 in_channels
        _weight = self.encoder.conv1.weight.clone()
        # __weight = torch.zeros((64, 4, 7, 7))
        __weight = torch.zeros((64, 4, 7, 7))
        __weight[:, :3, :, :] = _weight
        
        _n_convin_out_channel = self.encoder.conv1.out_channels
        # _new_conv_in = nn.Conv2d(4, _n_convin_out_channel, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        _new_conv_in = nn.Conv2d(4, _n_convin_out_channel, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        _new_conv_in.weight = Parameter(__weight)
        self.encoder.conv1 = _new_conv_in
            
    def forward(self, x, guide_rgb=None, guide_mask=None, observation=None):
        x = (x - self.pixel_mean) / self.pixel_std
        x = torch.cat([x, guide_mask], dim=1)
        enc_feats = self.encoder(x)
        aspp_feat = self.aspp_head(enc_feats[-1])
        
        # replace the last feat with aspp feat
        enc_feats[-1] = aspp_feat
        enc_pp_feats = enc_feats[::-1] # bottom-up
        
        temp_feat_list = []
        for index, feat in enumerate(enc_pp_feats):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
                temp_feat_list.append(temp_feat)
            else:
                skip_feat = feat
                up_feat = temp_feat_list[index-1]
                temp_feat = self.conv_list[index](up_feat, skip_feat)
                temp_feat_list.append(temp_feat)

        ym, visible_pred = self.visible_depth_head(temp_feat_list[-1])
        cross_att_output = self.cross_att_layer(temp_feat_list[-1], ym)
        updated_feat = temp_feat_list[-1] + F.interpolate(cross_att_output, scale_factor=self.patch_size, mode='bilinear', align_corners=True)
        _, in_visible_pred = self.in_visible_depth_head(updated_feat)
        
        visible_pred_depth = F.sigmoid(visible_pred)
        in_visible_pred_depth = F.sigmoid(in_visible_pred)
        return visible_pred_depth, in_visible_pred_depth
        
    def _save_pretrained(self, save_directory) -> None:
        """Save weights from a Pytorch model to a local directory."""
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
    
# if __name__ == '__main__':
#     model = ADDeepLab()
#     a = torch.rand((1, 3, 512, 512))
#     pred1, pred2 = model(a)
#     print(pred1.shape, pred2.shape)
    