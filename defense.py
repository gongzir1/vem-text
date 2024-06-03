from args import args
from eval import *
from misc import *
import torch
import pickle
from AGRs import *
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import wandb

def cosine(FLmodel, user_updates,n_attackers):
    similarities = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    x1 = concatenated_updates_tensor
    x2 =concatenated_updates_tensor
    eps=1e-8
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    similarities = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    k=len(concatenated_updates_tensor)-n_attackers


    k_nearest = torch.topk(similarities, k, dim=1)      # return the largest for each client 
    neighbour_dist = torch.zeros(concatenated_updates_tensor.size(0))
    for i in range(concatenated_updates_tensor.size(0)):
        idx = k_nearest.indices[i]
        neighbour = similarities[idx][:,idx]
        neighbour_dist[i] = neighbour.sum()

    all_indices = torch.arange(neighbour_dist.size(0)) 
    cos_selected = torch.topk(neighbour_dist, k).indices

    selected_dict = defaultdict(lambda: torch.empty(len(cos_selected), user_updates[next(iter(user_updates))].size(1)))

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n]=user_updates[str(n)][cos_selected]
    return selected_dict



def Euclidean(FLmodel, user_updates,n_attackers):
    distance = []
    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    dist_matrix = torch.cdist(concatenated_updates_tensor, concatenated_updates_tensor)
    
    k=len(concatenated_updates_tensor)-n_attackers

    k_nearest = torch.topk(dist_matrix, k, largest=False,dim=1)      # return the largest for each client 
    neighbour_dist = torch.zeros(concatenated_updates_tensor.size(0))
    for i in range(concatenated_updates_tensor.size(0)):
        idx = k_nearest.indices[i]
        neighbour = dist_matrix[idx][:,idx]
        neighbour_dist[i] = neighbour.sum()

    all_indices = torch.arange(neighbour_dist.size(0)) 
    eud_selected = torch.topk(neighbour_dist, k,largest=False).indices

    selected_dict = defaultdict(lambda: torch.empty(len(eud_selected), user_updates[next(iter(user_updates))].size(1)))

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n]=user_updates[str(n)][eud_selected]
    return selected_dict



def FRL_fang(FLmodel, user_updates,round_users,n_attackers,val_loader,criterion,device,initial_scores,mode):
    losses_list = []
    acc_list = []
    k=round_users-n_attackers
    for i in range(round_users):
        local_rank=user_updates[i]

    with torch.no_grad():
        for batch_ind, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device, torch.float)
            targets = targets.to(device, torch.long)
            #   get local model
            local_models=utils.Get_local_models(FLmodel, user_updates, initial_scores,round_users)
            for i in range(round_users):
                outputs = local_models[i](inputs)
                if len(outputs.shape) == 1:
                    outputs = outputs.unsqueeze(0)

                loss = criterion(outputs, targets)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

                # Append individual losses and top-1 accuracies to lists
                losses_list.append(loss.data)
                acc_list.append(prec1 / 100.0)
        # Convert lists to tensors
        losses = torch.tensor(losses_list)
        acc = torch.tensor(acc_list)

    selected_dict = defaultdict(lambda: torch.empty(k, user_updates[next(iter(user_updates))].size(1)))
    ERR = torch.topk(acc, k).indices
    LFR = torch.topk(losses, k, largest=False).indices
    
    if mode == "ERR":  # acc
        fang_selected=ERR
    elif mode =='LFR':  # loss
        fang_selected = LFR
    elif mode =='combined':
        union = torch.cat([ERR, LFR])
        uniques, counts = union.unique(return_counts=True)
        fang_selected = uniques[counts > 1]

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            selected_dict[n]=user_updates[str(n)][fang_selected]
    return selected_dict