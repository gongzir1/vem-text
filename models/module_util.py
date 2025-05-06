"""
Replications of module utils from Wortsman et al. SupSup
"""
import torch
import torch.nn as nn
import torch.autograd as autograd

import math


def mask_init(module):      # current
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(2))
    return scores
def mask_init_normal(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_normal_(scores, mode='fan_in', nonlinearity='relu')
    return scores

def pspinit(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def bn_mask_init(module):
    return torch.ones(module.num_features)


def bn_mask_initv2(module):
    return torch.zeros(module.num_features)


def rank_one_init(module):
    scores = torch.Tensor(module.weight.size(0))
    nn.init.uniform_(scores, a=-1, b=1)
    scores = scores.sign().float()
    return scores


def rank_one_initv2(module):
    scores = torch.Tensor(module.weight.size(1))
    nn.init.uniform_(scores, a=-1, b=1)
    scores = scores.sign().float()
    return scores


def mask_initv2(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores[0]

def kaiming_normal_init(module):

    # Create a tensor for the weights with the same shape as the module's weight
    scores =torch.Tensor(module.weight.size())
    # Apply Kaiming Normal initialization
    nn.init.kaiming_normal_(scores, mode='fan_in', nonlinearity='relu')
    
    # Return the initialized tensor (e.g., first row for specific usage)
    return scores[0]  # Adjust as needed for your specific use case

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class GetSignedSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        ctx.save_for_backward(scores)
        out = scores.clone()
        # _, idx = scores.abs().flatten().sort()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out * scores.sign()

    @staticmethod
    def backward(ctx, g):
        scores, = ctx.saved_tensors

        # send the gradient g straight-through on the backward pass.
        return g , None


def get_subnet(scores, k):
    out = scores.clone()
    _, idx = scores.flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return out

def get_subnet_signed(scores, k):
    out = scores.clone()
    # _, idx = scores.abs().flatten().sort()
    _, idx = scores.flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return out * scores.sign()


class GetSubnetFast(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float()

    @staticmethod
    def backward(ctx, g):
        return g, None


def get_subnet_fast(scores, a=0):
    return (scores >= a).float()


def kaiming_normal(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_normal_(scores, nonlinearity="relu")
    return scores