from args import args
import torch
import torch.nn as nn
import models
from utils import *

from AGRs import *
from Attacks import *

import copy
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
import collections
from loss_new import MalRankOptimizerMid

def My_attack(tr_loaders,te_loader):

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            mal_rank={}
            for kk in np.random.choice(n_attackers, min(len(round_malicious), args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()

                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        rank=rank+1
                        # my new code
                        optimizer = MalRankOptimizerMid(rank)

                        final,p = optimizer.optimize_p()

                        rounded_mask_tensor = optimizer.convert_to_binary(final)
                        masked_input = optimizer.rank * rounded_mask_tensor
                        masked_reverse=torch.flip(masked_input, [0])

                        # Flip all 1s to 0s and 0s to 1s
                        flipped_mask_tensor = 1 - rounded_mask_tensor
                        rest=optimizer.rank*flipped_mask_tensor

                        # filled_rest_tensor = rest.masked_fill(rest == 0, masked_reverse[masked_reverse != 0])
                        # Find non-zero values in the masked input tensor
                        non_zero_indices = torch.nonzero(masked_reverse).squeeze()

                        # # Fill the zero values in the rest tensor with non-zero values from the masked input tensor
                        filled_rest_tensor = rest.clone()
                        

                        filled_rest_tensor[rest == 0] = masked_reverse[non_zero_indices]
                        # print(p)

                        # end

                        mal_rank = filled_rest_tensor.clone()
                        mal_rank=mal_rank-1
                        # print(mal_rank)
                        # for kk in round_malicious:
                        user_updates[str(n)]=mal_rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], mal_rank[None,:]), 0)
                        # print(user_updates[str(n)][1])
                        
                        del rank,mal_rank
            del optimizer, mp, scheduler
        ########################################Server AGR#########################################
        # similarities = compare_user_updates(FLmodel, user_updates)      
        # for i, sim in enumerate(similarities, 1):
        #     print(f"Pair {i}: Cosine Similarity = {sim}")
 
        # print("Pairwise Cosine Similarities:")
        # for key, value in similarities.items():
        #     print(f"Pair {key}: Similarity {value}")

       
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc
        

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1

def AS(tr_loaders,te_loader):

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)

        # Initialize a dictionary to store losses for the current round
        round_losses = {}

        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            train_loss_sum = 0.0
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

                train_loss_sum += train_loss

            # Store the loss for the current client
            round_losses[f'client_{kk}'] = train_loss_sum / args.local_epochs

           
            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal={}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)

                train_loss_sum = 0.0

                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train_label_flip(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()
                    train_loss_sum += train_loss

                # Store the loss for the current client
                round_losses[f'client_{kk}'] = train_loss_sum / args.local_epochs

                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler


        #######################################Pre AGR Based on Loss#########################################
        # Update the overall losses dictionary
        average_curr_round_loss=0
        average_prev_round_loss=0

        if e > 0:
            # Calculate average loss for the previous round
            average_prev_round_loss = sum(prev_round_losses.values()) / len(prev_round_losses)
            print(f"Average loss for previous round: {average_prev_round_loss}")

        # Calculate average loss for the current round
        average_curr_round_loss = sum(round_losses.values()) / len(round_losses)
        print(f"Average loss for current round: {average_curr_round_loss}")

        # Find the maximum value in round_losses
        max_L = max(round_losses.values())
        # print('MAX',max_L)

        # Update the losses for the previous round
        prev_round_losses = round_losses.copy()

        # Calculate the difference between current and previous round losses
        dL = average_curr_round_loss - average_prev_round_loss

        # Calculate e^(dL)
        e_to_dL = math.exp(dL)

        # group selection

        # Initialize selected_clients list
        selected_clients = []

        # Iterate through round_losses to select clients
        for client_id, loss in round_losses.items():
            if loss > max_L * e_to_dL:
                selected_clients.append(client_id)

        
    
        print('selectde clients',selected_clients)
        print('length of selected clients',len(selected_clients))
        
        
        
        group_client=3


        for kk in np.random.choice(args.round_nclients,group_client, replace=False):
        # for kk in selected_clients:
            sum_agrs_rank={}
            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    rank=Find_rank(m.scores.detach().clone())
                    rank_arg=torch.sort(rank)[1]
                    if str(n) in sum_agrs_rank:
                        sum_agrs_rank[str(n)]+=rank_arg
                    else:
                        sum_agrs_rank[str(n)]=rank_arg
                    del rank, rank_arg
            # del optimizer, mp, scheduler

        for n, m in FLmodel.named_modules():
            if hasattr(m, "scores"):
                sum_agrs_agr=torch.sort(sum_agrs_rank[str(n)])[1]
                    
                user_updates[str(n)]=sum_agrs_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], sum_agrs_agr[None,:]), 0)
        del sum_agrs_rank

                    
                
        ########################################Server AGR#########################################
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1


def FRL_train_label_flip_attack_agr_malicious(tr_loaders,te_loader):

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal={}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train_label_flip(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()
                for n, m in mp.named_modules():    
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())      # get the rank of current score
                        rank_arg=torch.sort(rank)[1]
                        if str(n) in sum_args_sorts_mal:
                            sum_args_sorts_mal[str(n)]+=rank_arg       # aggreate the ranking of malicious
                        else:
                            sum_args_sorts_mal[str(n)]=rank_arg
                        del rank, rank_arg
                del optimizer, mp, scheduler

            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    rank_mal_agr=torch.sort(sum_args_sorts_mal[str(n)])[1]   
                    for kk in round_malicious:
                        user_updates[str(n)]=rank_mal_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank_mal_agr[None,:]), 0)
            del sum_args_sorts_mal
                    
                
        ########################################Server AGR#########################################
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1


def FRL_SAGR_train_label_flip(tr_loaders,te_loader):

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal={}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train_label_flip(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()

                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        #######################################Pre AGR#########################################
        # get the sum of every client score
            
        group_client=3

        for kk in np.random.choice(args.round_nclients,group_client, replace=False):
            sum_agrs_rank={}
            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    rank=Find_rank(m.scores.detach().clone())
                    rank_arg=torch.sort(rank)[1]
                    if str(n) in sum_agrs_rank:
                        sum_agrs_rank[str(n)]+=rank_arg
                    else:
                        sum_agrs_rank[str(n)]=rank_arg
                    del rank, rank_arg
            # del optimizer, mp, scheduler

        for n, m in FLmodel.named_modules():
            if hasattr(m, "scores"):
                sum_agrs_agr=torch.sort(sum_agrs_rank[str(n)])[1]
                    
                user_updates[str(n)]=sum_agrs_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], sum_agrs_agr[None,:]), 0)
        del sum_agrs_rank

        
        ########################################Server AGR#########################################
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1


def Test_random(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal={}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()

                for n, m in mp.named_modules():    
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())      # get the rank of current score
                        rank_arg=torch.sort(rank)[1]

                            # aggreate the ranking of malicious 
                        if str(n) in sum_args_sorts_mal:
                            sum_args_sorts_mal[str(n)]+=rank_arg       
                        else:
                            sum_args_sorts_mal[str(n)]=rank_arg
                        del rank, rank_arg

                        #  get the ramdom rank
                    
                        
                        # random_indices = torch.randperm(len(rank))

                        # # Use the random indices to reorder the tensor
                        # random_rank = rank[random_indices]

                        # user_updates[str(n)]=random_rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], random_rank[None,:]), 0)

                # print(user_updates[str(n)])

                del optimizer, mp, scheduler

            # reverse aggreation ranking
            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    # rank_mal_agr=torch.sort(sum_args_sorts_mal[str(n)], descending=True)[1]    # simply sort in descending order
                    # print('descending')
                    # print(rank_mal_agr)

                    # shuffle the ranking
                    n = len(sum_args_sorts_mal[str(n)])

                    # Create a list of indices from 0 to n-1
                    indices = list(range(n))

                    # Shuffle the list of indices randomly
                    random.shuffle(indices)

                    # Convert the shuffled list of indices to a tensor
                    rank_mal_agr = torch.tensor(indices)
                    # print('random')
                    # print(rank_mal_agr)

                    


                    for kk in round_malicious:
                        user_updates[str(n)]=rank_mal_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank_mal_agr[None,:]), 0)
            del sum_args_sorts_mal
        ########################################Server AGR#########################################
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1


def FRL_train_label_flip_attack(tr_loaders,te_loader):

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train_label_flip(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()

                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        for kk in round_malicious:
                            user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################Server AGR#########################################
        # similarities = compare_user_updates(FLmodel, user_updates)      
        # for i, sim in enumerate(similarities, 1):
        #     print(f"Pair {i}: Cosine Similarity = {sim}")
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1

#####################################FRL#########################################
def FRL_train(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            sum_args_sorts_mal={}
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                torch.cuda.empty_cache()  
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()

                for n, m in mp.named_modules():    
                    if hasattr(m, "scores"):
                        rank=Find_rank(m.scores.detach().clone())      # get the rank of current score
                        rank_arg=torch.sort(rank)[1]
                        if str(n) in sum_args_sorts_mal:
                            sum_args_sorts_mal[str(n)]+=rank_arg       # aggreate the ranking of malicious
                        else:
                            sum_args_sorts_mal[str(n)]=rank_arg
                        del rank, rank_arg
                del optimizer, mp, scheduler

            for n, m in FLmodel.named_modules():
                if hasattr(m, "scores"):
                    rank_mal_agr=torch.sort(sum_args_sorts_mal[str(n)], descending=True)[1]    # simply sort in descending order
                    for kk in round_malicious:
                        user_updates[str(n)]=rank_mal_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank_mal_agr[None,:]), 0)
            del sum_args_sorts_mal
        ########################################Server AGR#########################################
        similarities = compare_user_updates(FLmodel, user_updates) 
        # for i, sim in enumerate(similarities, 1):
        #     print(f"Pair {i}: Cosine Similarity = {sim}")
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1
        
        
#####################################FedAVG#########################################
def FedAVG(tr_loaders, te_loader):
    print ("#########Federated Learning using FedAVG############")
    args.conv_type = 'StandardConv'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    model_received = []
    for i, (name, param) in enumerate(FLmodel.state_dict().items()):
        model_received = param.view(-1).data.type(torch.cuda.FloatTensor) if len(model_received) == 0 else torch.cat((model_received, param.view(-1).data.type(torch.cuda.FloatTensor)))
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates = []
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()
                
                
            params = []
            for i, (name, param) in enumerate(mp.state_dict().items()):
                params = param.view(-1).data.type(torch.cuda.FloatTensor) if len(params) == 0 else torch.cat((params, param.view(-1).data.type(torch.cuda.FloatTensor)))

            update =  (params - model_received)

            user_updates = update[None,:] if len(user_updates) == 0 else torch.cat((user_updates, update[None,:]), 0)

            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        for kk in round_malicious:
            scale=100000
            mal_update = scale * model_received
            user_updates = mal_update[None,:] if len(user_updates) == 0 else torch.cat((user_updates, mal_update[None,:]), 0)

        ########################################Server AGR#########################################
        agg_update = torch.mean(user_updates, dim=0)
        del user_updates
        model_received = model_received + agg_update
        FLmodel = getattr(models, args.model)().to(args.device)
        start_idx=0
        state_dict = {}
        previous_name = 'none'
        for i, (name, param) in enumerate(FLmodel.state_dict().items()):
            start_idx = 0 if i == 0 else start_idx + len(FLmodel.state_dict()[previous_name].data.view(-1))
            start_end = start_idx + len(FLmodel.state_dict()[name].data.view(-1))
            params = model_received[start_idx:start_end].reshape(FLmodel.state_dict()[name].data.shape)
            state_dict[name] = params
            previous_name = name

        FLmodel.load_state_dict(state_dict)
        
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
                
            if math.isnan(t_loss) or t_loss > 10000:
                print('val loss %f... exit: The global model is totally destroyed by the adversary' % t_loss)
                break
        e+=1
        
#######################################Trimmed-Mean######################################
def Tr_Mean(tr_loaders, te_loader):
    print ("#########Federated Learning using Trimmed Mean############")
    args.conv_type = 'StandardConv'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    model_received = []
    for i, (name, param) in enumerate(FLmodel.state_dict().items()):
        model_received = param.view(-1).data.type(torch.cuda.FloatTensor) if len(model_received) == 0 else torch.cat((model_received, param.view(-1).data.type(torch.cuda.FloatTensor)))
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates = []
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()
                
                
            params = []
            for i, (name, param) in enumerate(mp.state_dict().items()):
                params = param.view(-1).data.type(torch.cuda.FloatTensor) if len(params) == 0 else torch.cat((params, param.view(-1).data.type(torch.cuda.FloatTensor)))

            update =  (params - model_received)

            user_updates = update[None,:] if len(user_updates) == 0 else torch.cat((user_updates, update[None,:]), 0)

            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):  # check if there are any malicious clients in the current round.
            mal_updates = []
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)

                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()


                params = []
                for i, (name, param) in enumerate(mp.state_dict().items()):
                    params = param.view(-1).data.type(torch.cuda.FloatTensor) if len(params) == 0 else torch.cat((params, param.view(-1).data.type(torch.cuda.FloatTensor)))

                update =  (params - model_received)

                mal_updates = update[None,:] if len(mal_updates) == 0 else torch.cat((mal_updates, update[None,:]), 0)

                del optimizer, mp, scheduler
                
            mal_update = our_attack_trmean(mal_updates, len(round_malicious), dev_type='std', threshold=5.0)
            del mal_updates

            for kk in round_malicious:
                user_updates = mal_update[None,:] if len(user_updates) == 0 else torch.cat((user_updates, mal_update[None,:]), 0)
        ########################################Server AGR#########################################
        agg_update = tr_mean(user_updates, len(round_malicious))
        del user_updates
        model_received = model_received + agg_update
        FLmodel = getattr(models, args.model)().to(args.device)
        start_idx=0
        state_dict = {}
        previous_name = 'none'
        for i, (name, param) in enumerate(FLmodel.state_dict().items()):
            start_idx = 0 if i == 0 else start_idx + len(FLmodel.state_dict()[previous_name].data.view(-1))
            start_end = start_idx + len(FLmodel.state_dict()[name].data.view(-1))
            params = model_received[start_idx:start_end].reshape(FLmodel.state_dict()[name].data.shape)
            state_dict[name] = params
            previous_name = name

        FLmodel.load_state_dict(state_dict)
        
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if math.isnan(t_loss) or t_loss > 10000:
                print('val loss %f... exit: The global model is totally destroyed by the adversary' % val_loss)
                break
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1
        
        
        
############################Multi-Krum#########################################
def Mkrum(tr_loaders, te_loader):
    print ("#########Federated Learning using Multi-Krum############")
    args.conv_type = 'StandardConv'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    FLmodel = getattr(models, args.model)().to(args.device)
    
    model_received = []
    for i, (name, param) in enumerate(FLmodel.state_dict().items()):
        model_received = param.view(-1).data.type(torch.cuda.FloatTensor) if len(model_received) == 0 else torch.cat((model_received, param.view(-1).data.type(torch.cuda.FloatTensor)))
    
    e=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates = []
        ########################################benign Client Learning#########################################
        for kk in round_benign:
            mp = copy.deepcopy(FLmodel)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()
                
                
            params = []
            for i, (name, param) in enumerate(mp.state_dict().items()):
                params = param.view(-1).data.type(torch.cuda.FloatTensor) if len(params) == 0 else torch.cat((params, param.view(-1).data.type(torch.cuda.FloatTensor)))

            update =  (params - model_received)

            user_updates = update[None,:] if len(user_updates) == 0 else torch.cat((user_updates, update[None,:]), 0)

            del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            mal_updates = []
            for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
                mp = copy.deepcopy(FLmodel)
                optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)

                scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
                for epoch in range(args.local_epochs):
                    train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                    scheduler.step()


                params = []
                for i, (name, param) in enumerate(mp.state_dict().items()):
                    params = param.view(-1).data.type(torch.cuda.FloatTensor) if len(params) == 0 else torch.cat((params, param.view(-1).data.type(torch.cuda.FloatTensor)))

                update =  (params - model_received)

                mal_updates = update[None,:] if len(mal_updates) == 0 else torch.cat((mal_updates, update[None,:]), 0)

                del optimizer, mp, scheduler
                
            mal_agg_update = torch.mean(mal_updates, 0)
            mal_update = our_attack_mkrum(mal_updates, mal_agg_update, len(round_malicious), dev_type='std', threshold=5.0, threshold_diff=1e-5)
            del mal_updates

            for kk in round_malicious:
                user_updates = mal_update[None,:] if len(user_updates) == 0 else torch.cat((user_updates, mal_update[None,:]), 0)
        ########################################Server AGR#########################################
        agg_update, krum_candidate = multi_krum(user_updates, len(round_malicious), multi_k=True)
        del user_updates
        model_received = model_received + agg_update
        FLmodel = getattr(models, args.model)().to(args.device)
        start_idx=0
        state_dict = {}
        previous_name = 'none'
        for i, (name, param) in enumerate(FLmodel.state_dict().items()):
            start_idx = 0 if i == 0 else start_idx + len(FLmodel.state_dict()[previous_name].data.view(-1))
            start_end = start_idx + len(FLmodel.state_dict()[name].data.view(-1))
            params = model_received[start_idx:start_end].reshape(FLmodel.state_dict()[name].data.shape)
            state_dict[name] = params
            previous_name = name

        FLmodel.load_state_dict(state_dict)
        
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device)
            if math.isnan(t_loss) or t_loss > 10000:
                print('val loss %f... exit: The global model is totally destroyed by the adversary' % val_loss)
                break
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
        e+=1