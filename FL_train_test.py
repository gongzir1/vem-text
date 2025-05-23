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
from loss_m import MalOptimizer
import wandb
from loss_new_noise import MalOptimizerNoise
# from my_attack import My_ATTACK_OPTIMISE
import my_attack_new
    # track hyperparameters and run metadata
config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    # "α":{"max": 0.6, "min": 0.4},
    # "δ":{"max": 0.6, "min": 0.4},
    # "β":{"max": 0.6, "min": 0.4},
    # "γ":{"max": 0.6, "min": 0.5},
    # "lr1":{"values": [0.001]},
    # "nep":{"values": [25,20]},
    # "k":{"max": 0.6, "min": 0.4},
    # # "l":{"values": [1,0.5]},
    # "n_max":{"max": 0.06, "min": 0.04},
    
    'm_r':{"values": [0.2,0.1,0.3,0.05]},

    },
}
# sweep_id = wandb.sweep(sweep=config, project="matrix-mnist_new")

def FRL_matrix_attack(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings############")
    # run = wandb.init()
    # m_r=wandb.config.m_r
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
        # round_malicious = round_users[:0]
        # round_benign = round_users[0:]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        rs=collections.defaultdict(list)
        mylist=['convs.0','convs.2']
        ########################################benign Client Learning#########################################
        m_c=collections.defaultdict(list)
        for n, m in FLmodel.named_modules():
            if hasattr(m, "scores"):
                m_c[str(n)]=0
            
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
                    ########### pass m benign rankings to attacker#############
                    if m_c[str(n)]<len(round_malicious):
                        # rank=rank.unsqueeze(0)
                        rs[str(n)]=rank[None,:] if len(rs[str(n)])==0 else torch.cat((rs[str(n)],rank[None,:]),0)
                        m_c[str(n)]=m_c[str(n)]+1
                    # del permutation_matrix
                    ######################################################################
                    
                    user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                    del rank
                            
        del optimizer, mp, scheduler
        ########################################malicious Client Learning######################################
        if len(round_malicious):
            mal_rank={}
            # for kk in np.random.choice(n_attackers, min(len(round_malicious), args.rand_mal_clients), replace=False):
            #     torch.cuda.empty_cache()  
            #     mp = copy.deepcopy(FLmodel)
            #     optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            #     scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            #     for epoch in range(args.local_epochs):
            #         train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
            #         scheduler.step()
            torch.cuda.empty_cache()  
            mp = copy.deepcopy(FLmodel)
            for n, m in mp.named_modules():
                if hasattr(m, "scores"):

                    ######### my attack###########
                    # optimizer = My_ATTACK_OPTIMISE(args.round_nclients,pms[str(n)],agr_matrix_m[str(n)],args.sparsity,len(round_malicious),mylist,args.device)
                    mal_rank=my_attack_new.optimize(args.round_nclients,rs[str(n)],args.sparsity,len(round_malicious),args.device)
                    user_updates[str(n)]=mal_rank if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], mal_rank), 0)
                    del mal_rank
                        
            del mp      

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
            # wandb.log(
            # {
            #     "val_acc": t_acc,
            #     "best_acc":t_best_acc,
            #     "loss":t_loss,
            #     # "location":args.run_base_dir / "output.txt"

            # })
        e+=1
 
def FRL_matrix_attack_old(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings############")
    # run = wandb.init()
    # m_r=wandb.config.m_r
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
        # round_malicious = round_users[:0]
        # round_benign = round_users[0:]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        agr_matrix=collections.defaultdict(list)
        agr_matrix_m=collections.defaultdict(list)
        pms=collections.defaultdict(list)
        mylist=['convs.0','convs.2']
        ########################################benign Client Learning#########################################
        m_c=collections.defaultdict(list)
        for x in mylist:
            m_c[x]=0

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
                    ########################### my permutation matrix #########################
                    if str(n) in mylist:
                        
                        # Create a binary 2D tensor
                        size = rank.size(0)

                        # Initialize an empty permutation matrix
                        permutation_matrix = torch.zeros(size, size,dtype=torch.float32).to(args.device)

                        # Iterate over the indices and fill the permutation matrix
                        for i, index in enumerate(rank):
                            permutation_matrix[i, index] = 1                               
                        agr_matrix[str(n)] = permutation_matrix if len(agr_matrix[str(n)])==0 else permutation_matrix+agr_matrix[str(n)]
                        ########### pass m benign permutation matrix to attacker#############
                        if m_c[str(n)]<len(round_malicious):
                            agr_matrix_m[str(n)] = permutation_matrix if len(agr_matrix[str(n)])==0 else permutation_matrix+agr_matrix[str(n)]
                            permutation_matrix=permutation_matrix.unsqueeze(0)
                            pms[str(n)]=permutation_matrix if len(pms[str(n)])==0 else torch.cat((pms[str(n)],permutation_matrix),0)
                            m_c[str(n)]=m_c[str(n)]+1
                        del permutation_matrix
                    ######################################################################
                    else:
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
                        if str(n) in mylist:
                            #########my attack###########
                            # optimizer = My_ATTACK_OPTIMISE(args.round_nclients,pms[str(n)],agr_matrix_m[str(n)],args.sparsity,len(round_malicious),mylist,args.device)
                            mal_rank=my_attack.optimize(args.round_nclients,pms[str(n)],agr_matrix_m[str(n)],args.sparsity,len(round_malicious),args.device)
                            user_updates[str(n)]=mal_rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], mal_rank[None,:]), 0)
                            del mal_rank
                        else:
                            rank=Find_rank(m.scores.detach().clone())
                            user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                            del rank    
                        
            del optimizer, mp, scheduler      

        ########################################Server AGR#########################################

        FRL_Vote(FLmodel, user_updates, initial_scores,mylist,agr_matrix)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
            # wandb.log(
            # {
            #     "val_acc": t_acc,
            #     "best_acc":t_best_acc,
            #     "loss":t_loss,
            #     # "location":args.run_base_dir / "output.txt"

            # })
        e+=1
  

def My_attack_Noise_Mask_Backdoor(tr_loaders,te_loader):
    lr1 = 0.01   
    α=0.5
    γ=0.5
    β = 0.5
    δ=0.5
    nep=50

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"  
    # n_attackers = int(args.nClients * args.at_fractions)  
    args.round_nclients=5
    args.at_fractions=0.1
    n_attackers = int(args.nClients * args.at_fractions) 
    
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
        round_malicious = round_users[:1]
        round_benign = round_users[1:]
        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]
        # while len(round_malicious)>=args.round_nclients/2:
        #     round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        #     round_malicious = round_users[round_users < n_attackers]
        #     round_benign = round_users[round_users >= n_attackers]
            
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
                ################ attack on m directly ###############
                # lr=0.01
                k=0.5
                l=0.5
                n_max=0.05
                # my_list=['convs.0','convs.2','linear.2','linear.4']
                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                    
                        #######################################
                        optimizer = MalOptimizerNoise(m.scores.flatten(),args.device,γ,β,δ,lr1,nep,k,n_max)
                        mal_score = optimizer.optimize_noise_mask()
                        
                
                        rank=Find_rank(mal_score.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
                
            del optimizer, mp, scheduler
        ########################################Server AGR#########################################
        similarities = compare_user_updates_ranked_cluster_layer_wise(FLmodel, user_updates)      
     

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
        # wandb.log(
        #     {
        #         "val_acc": t_acc,
        #         "best_acc":t_best_acc,
        #         "loss":t_loss,
        #         # "pick out":cs_not_selected,
        #         # "location":args.run_base_dir / "output.txt"

        #     }
        # )
        e+=1

def My_attack_Noise_Mask(tr_loaders,te_loader):
    lr1 = 0.01   
    α=0.5
    γ=0.5
    β = 0.5
    δ=0.5
    nep=50

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"  
    # n_attackers = int(args.nClients * args.at_fractions)  
    args.round_nclients=5
    args.at_fractions=0.1
    n_attackers = int(args.nClients * args.at_fractions) 
    
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
        round_malicious = round_users[:1]
        round_benign = round_users[1:]
        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]
        # while len(round_malicious)>=args.round_nclients/2:
        #     round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        #     round_malicious = round_users[round_users < n_attackers]
        #     round_benign = round_users[round_users >= n_attackers]
            
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
                        # Plot the tensor
                        
                        #############Plot the tensor###########
                        # scores_cpu = m.scores.detach().flatten().cpu()

                        # plt.scatter(range(len(scores_cpu.flatten())), scores_cpu.flatten(),marker='.')
                        # # Show the plot
                        # plt.show()

                        #########Plotting the histogram##################
                        # scores_cpu = m.scores.detach().cpu().numpy()
                        # plt.hist(scores_cpu.flatten(), bins=50)  # Adjust the number of bins as needed
                        # plt.xlabel('Value')
                        # plt.ylabel('Frequency')
                        # plt.title('Distribution of Tensor')
                        # plt.grid(True)
                        # plt.show()
                        ########################################

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
                ################ attack on m directly ###############
                # lr=0.01
                k=0.5
                l=0.5
                n_max=0.05
                # my_list=['convs.0','convs.2','linear.2','linear.4']
                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        ########Plotting the histogram##################
                        # scores_cpu = m.scores.detach().cpu().numpy()
                        # plt.figure(figsize=(8, 6))
                        # plt.hist(scores_cpu.flatten(), bins=50)  # Adjust the number of bins as needed
                        # plt.xlabel('Value')
                        # plt.ylabel('Frequency')
                        # plt.title('Distribution of score')
                        # plt.grid(True)
                        # plt.show()
                        #######################################
                        optimizer = MalOptimizerNoise(m.scores.flatten(),args.device,γ,β,δ,lr1,nep,k,n_max)
                        mal_score = optimizer.optimize_noise_mask()
                        # rank=Find_rank(m.scores.detach().clone())
                        ########Plotting the histogram##################
                        # scores_cpu = mal_score.detach().cpu().numpy()
                        # plt.figure(figsize=(8, 6))
                        # plt.hist(scores_cpu.flatten(), bins=50)  # Adjust the number of bins as needed
                        # plt.xlabel('Value')
                        # plt.ylabel('Frequency')
                        # plt.title('Distribution of mal_score')
                        # plt.grid(True)
                        # plt.show()
                        #######################################
                
                        rank=Find_rank(mal_score.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
                
            del optimizer, mp, scheduler
        ########################################Server AGR#########################################
        similarities = compare_user_updates_ranked_cluster_layer_wise(FLmodel, user_updates)      
        # for i, sim in enumerate(similarities, 1):
        #     print(f"Pair {i}: Cosine Similarity = {sim}")
 
        # print("Pairwise Cosine Similarities:")
        # for key, value in similarities.items():
        #     print(f"Pair {key}: Similarity {value}")
        # cs_selected,cs_not_selected =cosine_distance_PICK_OUT(user_updates,FLmodel,len(round_malicious))  


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
        # wandb.log(
        #     {
        #         "val_acc": t_acc,
        #         "best_acc":t_best_acc,
        #         "loss":t_loss,
        #         # "pick out":cs_not_selected,
        #         # "location":args.run_base_dir / "output.txt"

        #     }
        # )
        e+=1
 
def My_attack_new(tr_loaders,te_loader):
    # run = wandb.init()

    lr1 = 0.1
    lr2 = 0.01

    α=0.5
    γ=0.5
    β = 0.5
    δ=0.5
    

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"  
    args.round_nclients=5
    args.at_fractions=0.1
    n_attackers = int(args.nClients * args.at_fractions)  
    
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
        round_malicious = round_users[:1]
        round_benign = round_users[1:]
        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]
        # while len(round_malicious)>=args.round_nclients/2:
        #     round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        #     round_malicious = round_users[round_users < n_attackers]
        #     round_benign = round_users[round_users >= n_attackers]
            
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
                ################ attack on m directly ###############
                lr=0.01
                v=0.3
                nep=1
                scale1=10
                scale2=100
                nep=5
                l_c=0.001
                for n, m in mp.named_modules():
                    if hasattr(m, "scores"):
                        optimizer = MalOptimizerNoise(m.scores.flatten(),args.device,α,γ,β,δ,lr1,lr2,nep,v,scale1,scale2,nep,l_c)
                        mask_tensor_b,d = optimizer.optimize_mask_on_m()
                          
                        ####################### get binary mask############################
                        scalared=100000*(2*mask_tensor_b-1)
                        final_mask=torch.sigmoid(scalared)
                        mean=torch.mean(final_mask)

                        #####################get mal score#####################
                        masked_input = m.scores.flatten() * final_mask                           
                        masked_mal=-(masked_input*abs(d.item()))
                        mal_score=m.scores.flatten()+masked_mal
                        ####################################################
                        rank=Find_rank(mal_score.detach().clone())
                        user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                        del rank
            del optimizer, mp, scheduler
        ########################################Server AGR#########################################
        # arg_selected =cosine_distance(user_updates,FLmodel,len(round_malicious))  
        compare_user_updates_ranked_cluster_layer_wise(FLmodel,user_updates) 
       
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
        # wandb.log(
        #     {
        #         "val_acc": t_acc,
        #         "best_acc":t_best_acc,
        #         "loss":t_loss,
        #         # "location":args.run_base_dir / "output.txt"

        #     }
        # )
        e+=1
 
def My_attack(tr_loaders,te_loader):

    print ("#########Federated Learning using Rankings############")
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    args.round_nclients=5
    args.at_fractions=0.1
    n_attackers = int(args.nClients * args.at_fractions)
    
    # n_attackers = int(args.nClients * args.at_fractions)
    # n_attackers=1
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
    i=0
    t_best_acc=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[:1]
        round_benign = round_users[1:]
        # round_malicious = []
        # round_benign = round_users

        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]
        # while len(round_malicious)>=args.round_nclients/2:
        #     round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        #     round_malicious = round_users[round_users < n_attackers]
        #     round_benign = round_users[round_users >= n_attackers]
            
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
                        lr=0.01
                        # k_s=200
                        # n_s=40
                        α=0.5
                        # optimizer = MalRankOptimizerMid(rank,lr,k_s,n_s,α)
                        optimizer = MalRankOptimizerMid(rank,lr,args.device)

                        final,p = optimizer.optimize_p()
                        

                        # optimize_p(self, num_epochs=100, lr=0.001,k=10,α):

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
   
        arg_selected =my_cosine_distance(user_updates,FLmodel,len(round_malicious))  

        if args.agr=='cosine':
            FRL_Vote(FLmodel, arg_selected, initial_scores)
        else:
            FRL_Vote(FLmodel, user_updates, initial_scores)

        
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc
                
    
            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            # with (args.run_base_dir / "output.txt").open("a") as f:
            #     f.write("\n"+str(sss))
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
    run = wandb.init()
    m_r=wandb.config.m_r
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    # args.round_nclients=25
    # args.at_fractions=0.2
    
    n_attackers = int(args.nClients * m_r)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(m_r,
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
        # round_malicious = round_users[:0]
        # round_benign = round_users[0:]
        while len(round_malicious)>=args.round_nclients/2:
            round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
            round_malicious = round_users[round_users < n_attackers]
            round_benign = round_users[round_users >= n_attackers]
            
        user_updates=collections.defaultdict(list)
        agr_matrix=collections.defaultdict(list)
        mylist=['convs.0','convs.2']
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
                        ########################### my permutation matrix #########################
                        if str(n) in mylist:
                            # Create a binary 2D tensor
                            size = rank.size(0)

                            # Initialize an empty permutation matrix
                            permutation_matrix = torch.zeros(size, size,dtype=torch.int32)

                            # Iterate over the indices and fill the permutation matrix
                            for i, index in enumerate(rank):
                                permutation_matrix[i, index] = 1                               
                            agr_matrix[str(n)] = permutation_matrix if len(agr_matrix[str(n)])==0 else permutation_matrix+agr_matrix[str(n)]
                            user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                            del permutation_matrix
                        ######################################################################
                        else:
                            user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                            del rank

        # for n, m in mp.named_modules():
        #         if hasattr(m, "scores"):
        #             if str(n) in mylist:

        #                 # ############## Plot histograms for each row##########
        #                 # rows_to_plot = [[142],[141],[140]]  # Indices for the first row, last row, and row 864
        #                 # for i, row_indices in enumerate(rows_to_plot):
        #                 #         # Select the rows from agr_matrix[str]
        #                 #     selected_rows = agr_matrix[str(n)][row_indices, :]
        #                 #     row_sum = torch.sum(selected_rows, dim=1).cpu().numpy()
        #                 #     print(row_sum)

        #                 #         # Flatten the selected rows to get all values
        #                 #     values = selected_rows.flatten().cpu().numpy()  # Assuming agr_matrix[str] is on GPU, so we move it to CPU and convert to NumPy array

        #                 #         # Plot the histogram
        #                 #     plt.figure(figsize=(8, 6))
        #                 #     plt.hist(values, bins=10)  # You can adjust the number of bins as needed
        #                 #     plt.title(f'Histogram for row {row_indices[0]}')
        #                 #     plt.xlabel('Values')
        #                 #     plt.ylabel('Frequency')
        #                 #     plt.show()
        #                 ###################################################################
        #                 ##########PLOT the largest and second largest#######################
        #                 # Find the indices of the largest and second-largest values in each row
        #                 # 1. Sort each row in descending order and get the largest values of each row
        #                 sorted_matrix = np.sort(agr_matrix[str(n)], axis=1)[:, ::-1]

        #                 # Extract the largest values in each row
        #                 largest_values = sorted_matrix[:, 0]
        #                 # 2. Extract the second largest starting from selected rows
        #                 # for not selected rows 
        #                 start_row = agr_matrix[str(n)].size(0)//2
                        
        #                 indices_largest_first = torch.argmax(agr_matrix[str(n)][:start_row], dim=1)

        #                 # Set all coloums with indices_largest to 0
        #                 agr_matrix[str(n)][:start_row, indices_largest_first] = 0
             
        #                 # for selected rows
                        
        #                 # Find the indices of the largest values in each row starting from the specified row
        #                 indices_largest = torch.argmax(agr_matrix[str(n)][start_row:], dim=1)
                        
        #                 # Set all coloums with indices_largest to 0
        #                 agr_matrix[str(n)][start_row:, indices_largest] = 0

        #                 # 3. get the second largest which are not in the selected edges
        #                 sorted_matrix_2 = np.sort(agr_matrix[str(n)], axis=1)[:, ::-1]
                        
        #                 second_largest_values = sorted_matrix_2[:, 0]

        #                 difference = largest_values - second_largest_values

        #                 # Plotting rhw difference
        #                 plt.figure(figsize=(8, 6))
        #                 plt.scatter(range(len(difference)), difference, label='Difference', marker='o',s=10)
        #                 plt.xlabel('Row Index')
        #                 plt.ylabel('Difference')
        #                 plt.title('Difference between Largest and Second Largest Values in Each Row')
        #                 plt.legend()
        #                 plt.show()

        #                 # Plotting the largest and seocnd largest 
        #                 plt.figure(figsize=(8, 6))
        #                 plt.plot(largest_values, label='Largest')
        #                 plt.plot(second_largest_values, label='Second Largest')
        #                 plt.xlabel('Row Index')
        #                 plt.ylabel('Value')
        #                 plt.title('Largest and Second Largest Values in Each Row')
        #                 plt.legend()
        #                 plt.show()

                        
        #                 ###########################################
        #                 # Define the bins for the histogram                    

        

        #                 # Divide the indices into four equal parts
        #                 num_parts = 4
        #                 indices_per_part = len(difference) // num_parts

        #                 # Initialize counters for each category within each part
        #                 count_0 = np.zeros(num_parts, dtype=int)
        #                 count_1 = np.zeros(num_parts, dtype=int)user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
        #                 count_gt1 = np.zeros(num_parts, dtype=int)

        #                 # Iterate over the parts and count occurrences of each category
        #                 for i in range(num_parts):
        #                     start_idx = i * indices_per_part
        #                     end_idx = (i + 1) * indices_per_part
        #                     part_difference = difference[start_idx:end_idx]

        #                     count_0[i] = np.sum(part_difference == 0)
        #                     count_1[i] = np.sum(part_difference == 1)
        #                     count_gt1[i] = np.sum(part_difference > 1)

        #                 # Plotting
        #                 plt.figure(figsize=(10, 6))
        #                 bar_width = 0.2
        #                 bar_positions = np.arange(num_parts)

        #                 plt.bar(bar_positions - bar_width, count_0, width=bar_width, label='0')
        #                 plt.bar(bar_positions, count_1, width=bar_width, label='1')
        #                 plt.bar(bar_positions + bar_width, count_gt1, width=bar_width, label='>1')

        #                 plt.xlabel('Ranges')
        #                 plt.ylabel('Count')
        #                 plt.title('Counts of Difference Values in Three Ranges for Each Part')
        #                 # plt.xticks(bar_positions, labels=[f'Part {i+1}' for i in range(num_parts)])
        #                 # Customize x-axis labels to represent fractions of the total range
        #                 total_range = len(difference)
        #                 part_ranges = [f'{i*indices_per_part/total_range:.2f}-{(i+1)*indices_per_part/total_range:.2f}' for i in range(num_parts)]
        #                 plt.xticks(bar_positions, labels=part_ranges)

        #                 plt.legend()
        #                 plt.grid(axis='y', linestyle='--', alpha=0.7)
        #                 plt.show()

        #                 # ###########################################
                            
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
                    ########################### my permutation matrix #########################
                    if str(n) in mylist:
                        # Create a binary 2D tensor
                        size = rank_mal_agr.size(0)

                        # Initialize an empty permutation matrix
                        permutation_matrix = torch.zeros(size, size,dtype=torch.int32)

                        # Iterate over the indices and fill the permutation matrix
                        for i, index in enumerate(rank_mal_agr):
                            permutation_matrix[i, index] = 1
                        for kk in round_malicious:                           
                            agr_matrix[str(n)] = permutation_matrix if len(agr_matrix[str(n)])==0 else permutation_matrix+agr_matrix[str(n)]
                            user_updates[str(n)]=rank_mal_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank_mal_agr[None,:]), 0)
                        del permutation_matrix
                     
                    ######################################################################
                    else:
                        for kk in round_malicious:
                            user_updates[str(n)]=rank_mal_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank_mal_agr[None,:]), 0)
            del sum_args_sorts_mal
        ########################################Server AGR#########################################

        FRL_Vote(FLmodel, user_updates, initial_scores,mylist,agr_matrix)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
            wandb.log(
            {
                "val_acc": t_acc,
                "best_acc":t_best_acc,
                "loss":t_loss,
                # "location":args.run_base_dir / "output.txt"

            })
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