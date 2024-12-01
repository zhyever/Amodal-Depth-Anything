import torch
import torch.nn as nn
import torch.nn.functional as F
from .SPADE import SPADE
from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    FeatureFusionBlock_custom2,
    Interpolate,
    _make_encoder,
    forward_vit,
    _make_scratch,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


def _make_fusion_block2(features, use_bn):
    return FeatureFusionBlock_custom2(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
            self,
            head,
            features=256,
            backbone="vitb_rn50_384",
            readout="project",
            channels_last=False,
            use_bn=False,
            enable_attention_hooks=False,
    ):
        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "beitl16_384": [5, 11, 17, 23],
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        label = 1
        self.spade_fusion4 = SPADE(features, label)
        self.spade_fusion3 = SPADE(features, label)
        self.spade_fusion2 = SPADE(features, label)
        self.spade_fusion1 = SPADE(features, label)

        # self.spade_fusion4 = _make_fusion_block2(features+2, use_bn)
        # self.spade_fusion3 = _make_fusion_block2(features+2, use_bn)
        # self.spade_fusion2 = _make_fusion_block2(features+2, use_bn)
        # self.spade_fusion1 = _make_fusion_block2(features+2, use_bn)

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head
        self.d_feat = nn.Conv2d(features, 80, kernel_size=1)
    def forward(self, x,seg):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)


        seg_4= F.interpolate(seg,size=(layer_4_rn.shape[-2],layer_4_rn.shape[-1]))
        layer_4_rn = self.spade_fusion4(layer_4_rn,seg_4)
        #layer_4_rn = self.spade_fusion4(torch.cat([layer_4_rn,seg_4],dim=1))
        path_4 = self.scratch.refinenet4(layer_4_rn)

        seg_3 = F.interpolate(seg, size=(layer_3_rn.shape[-2], layer_3_rn.shape[-1]))
        path_4 = self.spade_fusion3(path_4,seg_3)
        #path_4 = self.spade_fusion3(torch.cat([path_4, seg_3],dim=1))
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)

        seg_2 = F.interpolate(seg, size=(layer_2_rn.shape[-2], layer_2_rn.shape[-1]))
        path_3 = self.spade_fusion2(path_3,seg_2)
        #path_3 = self.spade_fusion2(torch.cat([path_3, seg_2], dim=1))
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)

        seg_1 = F.interpolate(seg, size=(layer_1_rn.shape[-2], layer_1_rn.shape[-1]))
        path_2 = self.spade_fusion1(path_2,seg_1)
        #path_2 = self.spade_fusion1(torch.cat([path_2, seg_1], dim=1))
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        path_1 = self.d_feat(path_1)
        return out ,path_1


class DPT_2(BaseModel):
    def __init__(
            self,
            head,
            head2,
            features=256,
            backbone="vitb_rn50_384",
            readout="project",
            channels_last=False,
            use_bn=False,
            enable_attention_hooks=False,
    ):
        super(DPT_2, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        self.m_scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=1, expand=False
        )

        self.m_scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.m_scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.m_scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.m_scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.m_fusionblock1 = _make_fusion_block2(features, use_bn)
        self.m_fusionblock2 = _make_fusion_block2(features, use_bn)
        self.m_fusionblock3 = _make_fusion_block2(features, use_bn)
        self.m_fusionblock4 = _make_fusion_block2(features, use_bn)

        self.scratch.output_conv = head
        self.m_scratch.output_conv = head2

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        m_layer_1_rn = self.m_scratch.layer1_rn(layer_1)
        m_layer_2_rn = self.m_scratch.layer2_rn(layer_2)
        m_layer_3_rn = self.m_scratch.layer3_rn(layer_3)
        m_layer_4_rn = self.m_scratch.layer4_rn(layer_4)

        m_path_4 = self.m_scratch.refinenet4(m_layer_4_rn)
        m_path_3 = self.m_scratch.refinenet3(m_path_4, m_layer_3_rn)
        m_path_2 = self.m_scratch.refinenet2(m_path_3, m_layer_2_rn)
        m_path_1 = self.m_scratch.refinenet1(m_path_2, m_layer_1_rn)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_4 = self.m_fusionblock4(path_4, m_path_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_3 = self.m_fusionblock3(path_3, m_path_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_2 = self.m_fusionblock3(path_2, m_path_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        path_1 = self.m_fusionblock3(path_1, m_path_1)

        out = self.scratch.output_conv(path_1)
        mask = self.m_scratch.output_conv(m_path_1)

        return out, mask


class DPTDepthModel(DPT):
    def __init__(
            self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=True, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        # head = nn.Sequential(
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        # )

        super().__init__(head, **kwargs)

        if path is not None:
            self.if_mask = True
            self.load(path)

    def forward(self, x,seg):
        inv_depth,path_1 = super().forward(x,seg)
        #path_1 = F.interpolate(path_1, size=(120, 160))
        # mask = super().forward(y)
        # return inv_depth, mask

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            depth = F.interpolate(depth, size=(120, 160))
            path_1 = F.interpolate(path_1, size=(120, 160))
            depth = depth  * 10.0
            depth[depth>6]=6
            return depth,path_1
        else:
            return inv_depth


class DPTRGBModel(DPT):
    def __init__(
            self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=True, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.if_rgb = True
            self.load(path)

    def forward(self, x):
        RGB = super().forward(x)

        return RGB





class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)
