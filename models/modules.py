"""
Replications of modules from Wortsman et al. SupSup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import models.module_util as module_util

from args import args as pargs
import wandb

StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d
StandLinear=nn.Linear


class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)

class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )
class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(module_util.mask_init(self))

        self.weight.requires_grad = False
            
        # default sparsity
        self.sparsity = wandb.config.k
        
    def forward(self, x):
        subnet = module_util.GetSubnet.apply(self.scores, wandb.config.k)
        w = self.weight * subnet
       
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class MaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the scores
        self.scores = nn.Parameter(module_util.mask_init(self))

        # Freeze the original weights
        self.weight.requires_grad = False

        # Default sparsity
        self.sparsity = wandb.config.k  # Sparsity level (fraction of weights to keep)

    # def mask_init(self):
    #     """
    #     Initialize the scores for the mask.
    #     """
    #     return torch.Tensor(self.weight.size()).uniform_(-0.1, 0.1)  # Example initialization

    def forward(self, x):
        # Apply the subnet mask to the weights
        subnet = module_util.GetSubnet.apply(self.scores, self.sparsity)
        w = self.weight * subnet

        # Perform the linear transformation
        x = F.linear(x, w, self.bias)
        return x

    

