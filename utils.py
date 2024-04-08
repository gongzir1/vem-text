from args import args
from eval import *
from misc import *
import torch
import pickle
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def Find_rank(scores):
    _, idx = scores.detach().flatten().sort()
    return idx.detach()


def FRL_Vote(FLmodel, user_updates, initial_scores):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(user_updates[str(n)])[1]
            # print(user_updates[str(n)].shape)
            # print('user_updates[str(n)]',user_updates[str(n)])
            # print('user_updates[str(n)] shape',user_updates[str(n)].shape)
            # print(len(args_sorts))
            sum_args_sorts=torch.sum(args_sorts, 0)         # get the score for each edge
            idxx=torch.sort(sum_args_sorts)[1]          # get the index of sorted edge
            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)]
            m.scores=torch.nn.Parameter(temp1)                    
            del idxx, temp1


            
def train_label_flip(trainloader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

         # Change the labels from l to L - l - 1 label flip attack
        targets_flipped = 10 - targets - 1  # untarget label flip
    
        # Change the labels to a specific value (e.g., 1)
        # target label flip
        # targets_flipped = torch.full_like(targets, 1, device=device, dtype=torch.long)


        # targets_flipped = targets_flipped.to(device, torch.long)

        # Print original and flipped labels
        # print('Original Labels:', targets)
        # print('Flipped Labels:', targets_flipped)

        outputs = model(inputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, targets_flipped)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets_flipped.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top5.update(prec5.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (losses.avg, top1.avg)
            
def train(trainloader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top5.update(prec5.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for batch_ind, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device, torch.float)
            targets = targets.to(device, torch.long)
            outputs = model(inputs)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size()[0])
            top1.update(prec1/100.0, inputs.size()[0])
            top5.update(prec5/100.0, inputs.size()[0])
    return (losses.avg, top1.avg)



def compare_user_updates(FLmodel, user_updates):
    similarities = []
    concatenated_updates_tensor = []


    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    num_rows = concatenated_updates_tensor.shape[0]
    for i in range(num_rows):
        for j in range(i+1, num_rows):
            row_i = concatenated_updates_tensor[i]
            row_j = concatenated_updates_tensor[j]
            similarity = F.cosine_similarity(row_i, row_j, dim=0)
            similarities.append(similarity.item())

    similarities = np.array(similarities)
    similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities))

    # Plot similarities
    # plt.figure(figsize=(8, 6))
    # plt.scatter(range(len(similarities)), similarities, marker='.')
    # plt.title('Similarities')
    # plt.xlabel('Pair Index')
    # plt.ylabel('Similarity')
    # plt.grid(True)
    # plt.show()

    return similarities


def compare_user_updates_distance(FLmodel, user_updates):
    distances = []

    concatenated_updates_tensor = []

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"): 
            if len(concatenated_updates_tensor) == 0: # Check if concatenated_updates_tensor is empty
                concatenated_updates_tensor = user_updates[str(n)].float()
            else:
                concatenated_updates_tensor = torch.stack([torch.cat((row_a, row_b)) for row_a, row_b in zip(concatenated_updates_tensor, user_updates[str(n)])])

    new_tensor = torch.unsqueeze(concatenated_updates_tensor, dim=0).cpu()
    new_tensor_2 = torch.unsqueeze(concatenated_updates_tensor, dim=1).cpu()

    distance = torch.abs(new_tensor - new_tensor_2).float()

    mean_along_third_dim = torch.mean(distance, dim=2)
    non_zero_tensor = mean_along_third_dim[mean_along_third_dim != 0]
    mean_along_third_dim_normalized = (non_zero_tensor - non_zero_tensor.mean()) / non_zero_tensor.std()
    

    distances=mean_along_third_dim_normalized.flatten()
    distances_numpy = distances.detach().numpy()


    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(distances_numpy)), distances_numpy, marker='o', color='b')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Normalized Tensor')
    plt.grid(True)
    plt.show()

    return distances


# Example tensor with shape (3,)
# tensor = torch.tensor([[1, 2, 3],[2,3,4]])

# # Add an extra dimension at position 0
# new_tensor = torch.unsqueeze(tensor, dim=0)
# new_tensor_2 = torch.unsqueeze(tensor, dim=1)

# distance = torch.abs(new_tensor - new_tensor_2).float()

# mean_along_third_dim = torch.mean(distance, dim=2)

# # Create a mask to identify non-zero elements
# non_zero_mask = mean_along_third_dim != 0

# # Filter the tensor using the mask
# filtered_tensor = mean_along_third_dim[non_zero_mask]

# # Calculate the mean of non-zero elements
# mean_of_non_zero = torch.mean(filtered_tensor)

# print("Distance (flattened and unique):", distance)
# print(mean_along_third_dim)

# print("Mean of non-zero elements:", mean_of_non_zero)



