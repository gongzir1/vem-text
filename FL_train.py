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
# from loss_new import MalRankOptimizerMid
# from loss_m import MalOptimizer
# from loss_new_noise import MalOptimizerNoise
import wandb
# import my_attack_new
# import other_attacks
import defense
# import forecast
# from scipy.interpolate import make_interp_spline
# from matplotlib.ticker import MaxNLocator
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_curve, roc_auc_score
# import matplotlib.pyplot as plt



 
def FRL_matrix_attack_defense(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings############")
    # run = wandb.init()
    k=wandb.config.k

    m_r=wandb.config.m_r

    lr=wandb.config.lr
    nep=wandb.config.nep
    # max_t=wandb.config.max_t
    temp=wandb.config.temp
    iteration=wandb.config.iteration
    noise=wandb.config.noise

    # args.conv_type = 'MaskConv'
    args.conv_type='MaskLinear'  # change here to all liner for text dataset
    
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    

    
    n_attackers = int(args.nClients * m_r)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(m_r,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    # FLmodel = getattr(models, args.model)().to(args.device)
    if args.model=='FCN':
        input_size = 6169  # Number of features in the Purchase dataset
        hidden_size = 1024*2  # Number of neurons in the hidden layer
        num_classes = 100  # Number of output classes
        FLmodel = getattr(models, args.model)(input_size, hidden_size,num_classes).to(args.device)   
    elif args.model=='TextCNN':
        input_size = 446  # Number of features in the Purchase dataset
        # hidden_size = 1024  # Number of neurons in the hidden layer
        num_classes = 30  # Number of output classes
        sequence_length=1
        FLmodel = getattr(models, args.model)(446,30).to(args.device)
    else:
        FLmodel = getattr(models, args.model)().to(args.device)
    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    total_sum = 0
    total_fpr=0
    total_tpr=0
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        # random select
        # round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        # round_malicious = round_users[round_users < n_attackers]
        # round_benign = round_users[round_users >= n_attackers]

        all_clients = np.arange(args.nClients)
        malicious_clients = np.random.choice(all_clients, n_attackers, replace=False)
        
        # Select clients for the round
        round_users = np.random.choice(all_clients, args.round_nclients, replace=False)

        num_round_malicious = int(args.round_nclients * m_r)
        # Ensure exactly num_round_malicious malicious clients
        round_malicious = np.random.choice(round_users, num_round_malicious, replace=False)
        round_benign = np.setdiff1d(round_users, round_malicious) 
            
        user_updates=collections.defaultdict(list)
        rs=collections.defaultdict(list)

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
            torch.cuda.empty_cache()  
            mp = copy.deepcopy(FLmodel)
            for n, m in mp.named_modules():
                if hasattr(m, "scores"):
                    ######### my attack###########
                    # optimizer = My_ATTACK_OPTIMISE(args.round_nclients,pms[str(n)],agr_matrix_m[str(n)],args.sparsity,len(round_malicious),mylist,args.device)
                    mal_rank=my_attack_new.optimize(args.round_nclients,rs[str(n)],k,len(round_malicious),args.device,lr,nep,wandb.config.max_t,temp,iteration,noise)
                    
                    # mal_rank=my_attack_new.reverse(args.round_nclients,rs[str(n)],k,len(round_malicious))
                    user_updates[str(n)]=mal_rank if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], mal_rank), 0)
                    del mal_rank                  
            del mp      

        ########################################Server AGR#########################################
        if wandb.config.defense=='foolsgold':
            selected_user_updates=defense.foolsgold(FLmodel, user_updates,args.device,initial_scores)
        elif wandb.config.defense=='my_defense_adaptive':
            selected_user_updates=user_updates
            FRL_Vote_adaptive(FLmodel, selected_user_updates, initial_scores,wandb.config.k_a)
        else:
            if wandb.config.defense=='cosine':
                selected_user_updates=defense.cosine(FLmodel, user_updates,int(0.2*len(round_users)))

            elif wandb.config.defense=='Eud':
                selected_user_updates=defense.Euclidean(FLmodel, user_updates,int(0.2*len(round_users)))
            elif wandb.config.defense=='Krum':
                selected_user_updates=defense.Krum(FLmodel, user_updates,int(0.2*len(round_users)))
            elif wandb.config.defense=='FABA':
                selected_user_updates=defense.FABA(FLmodel, user_updates,int(0.2*len(round_users)))
            elif wandb.config.defense=='DnC':
                selected_user_updates=defense.DnC(FLmodel, user_updates,int(0.2*len(round_users)),wandb.config.sub_dim, wandb.config.num_iters,wandb.config.filter_frac)
            elif wandb.config.defense=='my_DnC':
                #max_val,min_val=defense.test(FLmodel, user_updates)
                #selected_user_updates=defense.My_Dnc_defense_old(FLmodel, user_updates,int(0.2*len(round_users)))
                selected_user_updates=defense.My_Dnc_defense(FLmodel, user_updates,wandb.config.maxt,wandb.config.mint,sub_dim=wandb.config.sub_dim)
            elif wandb.config.defense=='My_Dnc_defense_topk':
                # selected_user_updates,detection_acc,fpr,tpr=defense.My_Dnc_defense_topk_threshold(FLmodel, user_updates,wandb.config.k,wandb.config.threshold)
                selected_user_updates,detection_acc,fpr,tpr=defense.My_Dnc_defense_topk_cluster(FLmodel, user_updates,wandb.config.k,int(0.2*len(round_users)))
            elif wandb.config.defense=='FRL':
                selected_user_updates=user_updates
            
            FRL_Vote(FLmodel, selected_user_updates, initial_scores)

        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc)

            print (sss)
            # defense
            # total_sum+=detection_acc
            # total_fpr+=fpr
            # total_tpr+=tpr
            # sss='e %d | detection_acc: %.4F | fpr %.4f| tpr %.4f |' % (e,detection_acc, fpr, tpr)

            # print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
            wandb.log(
            {
                "val_acc": t_acc,
                "best_acc":t_best_acc,
                # "loss":t_loss,
                # 'detection_acc':detection_acc,
                # 'fpr':fpr,
                # 'tpr':tpr,
            })
        e+=1
    # print('detection accuracy')
    # print(total_sum/(e+1))
    # print('fpr:')
    # print(total_fpr/(e+1))
    # print('tpr')
    # print(total_tpr/(e+1))



#####################################FRL#########################################
def FRL_train(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings############")
    run = wandb.init()
    m_r=wandb.config.m_r
    # args.conv_type = 'MaskConv'
    args.conv_type = 'MaskLinear'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * m_r)
    sss = "fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(m_r,
                                                                                        n_attackers)
    print (sss)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(sss))
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    # print(args.model)
    if args.model=='FCN' or args.model=='FCN_Four':
        
        if args.set=='Location':
            input_size = 446  # Number of features in the Purchase dataset
            hidden_size = 1024*2  # Number of neurons in the hidden layer
            num_classes = 30  # Number of output classes
        elif args.set=='Texas':
            input_size = 6169  # Number of features in the Purchase dataset
            hidden_size = 1024*2  # Number of neurons in the hidden layer
            num_classes = 100  # Number of output classes
        else:
            input_size = 600  # Number of features in the Purchase dataset
            hidden_size = 1024*2  # Number of neurons in the hidden layer
            num_classes = 100  # Number of output classes
        FLmodel = getattr(models, args.model)(input_size, hidden_size,num_classes).to(args.device)
    elif args.model=='TextCNN':
        input_size = 446  # Number of features in the Purchase dataset
        # hidden_size = 1024  # Number of neurons in the hidden layer
        num_classes = 30  # Number of output classes
        sequence_length=1
        FLmodel = getattr(models, args.model)(446,30).to(args.device)
    else:
        FLmodel = getattr(models, args.model)().to(args.device)

    initial_scores={}
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            initial_scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]
    
    e=0
    t_best_acc=0
    e_values = []
    acc_values = []
    robust_values = []
    
    print(args.lr,args.lrdc, args.momentum, args.wd, args.local_epochs)
    while e <= args.FL_global_epochs:
        torch.cuda.empty_cache() 
        all_clients = np.arange(args.nClients)
        malicious_clients = np.random.choice(all_clients, n_attackers, replace=False)
        
        # Select clients for the round
        round_users = np.random.choice(all_clients, args.round_nclients, replace=False)

        num_round_malicious = int(args.round_nclients * m_r)
        # Ensure exactly num_round_malicious malicious clients
        round_malicious = np.random.choice(round_users, num_round_malicious, replace=False)
        round_benign = np.setdiff1d(round_users, round_malicious) 

        # round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
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
            # for n, m in mp.named_modules():
            #     if hasattr(m, "scores"):
            #         print(m.scores)

            # print(mp.named_modules)
            # print(n,m)
            optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**e), momentum=args.momentum, weight_decay=args.wd)
            # optimizer = optim.Adam([p for p in mp.parameters() if p.requires_grad], lr=args.lr)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
            for epoch in range(args.local_epochs):
                train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
                scheduler.step()

            for n, m in mp.named_modules():
                # print(m)
                if hasattr(m, "scores"):
                    # print('yes')
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
        # succ_rate=FRL_Vote_both(FLmodel, user_updates, initial_scores)
        FRL_Vote(FLmodel, user_updates, initial_scores)
        del user_updates
        if (e+1)%1==0:
            t_loss, t_acc = test(te_loader, FLmodel, criterion, args.device) 
            if t_acc>t_best_acc:
                t_best_acc=t_acc

            sss='e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f  train loss %.4f' % (e, len(round_malicious), t_acc, t_loss, t_best_acc,train_loss)
            print (sss)
            with (args.run_base_dir / "output.txt").open("a") as f:
                f.write("\n"+str(sss))
            
            wandb.log(
            {
                "val_acc": t_acc,
                "best_acc":t_best_acc,
                "loss":t_loss,

            })
        e+=1
        