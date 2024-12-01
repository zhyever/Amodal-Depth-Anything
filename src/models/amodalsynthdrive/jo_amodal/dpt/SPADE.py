"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm

class SPADE(nn.Module):

    def __init__(self,  norm_nc, label_nc): # norm_nc 256  , label_nc 2
        super().__init__()

        #assert config_text.startswith('spade')
        #parsed = re.search('spade(\D+)(\d)x\d', config_text)
        #param_free_norm_type = str(parsed.group(1))
        #ks = int(parsed.group(2))
        ks = 3
        # if param_free_norm_type == 'instance':
        #     self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # # elif param_free_norm_type == 'syncbatch':
        # #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'batch':
        #     self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # else:
        #     raise ValueError('%s is not a recognized param-free norm type in SPADE'
        #                      % param_free_norm_type)
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out