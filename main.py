import os
from args import args
import random
import numpy as np
import pathlib
import torch

import data
from FL_train import *
from args import *


# config={
#     "method": "grid",
#     "metric":{
#         "goal": "minimize", 
#         "name": "t_acc"
#         },
#     "parameters":{
#     # "lr":{"values": [0.01]},
#     # "nep":{"values": [50]},
#     # "max_t":{"values": [2000]},
#     # "iteration":{"values": [50]},
#     # "temp":{"values": [0.001]},
#     # "noise":{"values": [1]},
#     'k':{"values": [0.5]},
#     'm_r':{"values": [0.2,0]},
#     'non_iid':{"values": [1]},
#     # 'non_iid':{"values": [0.1,0.5,0.9,'iid']},
#     # 'mode':{"values": ['ERR','LFR','combined']},
#     # 'defense':{"values": ['FRL','FABA','DnC','cosine','Eud',]},
#     # 'attacks':{"values": ['grad_ascent','min_max','min_sum','noise']},
#     # 'defense':{"values": ['FABA','DnC']},
#     'defense':{"values": ['foolsgold']},
#     'attacks':{"values": ['lable']},
#     # 'attacks':{"values": ['rank-reverse']},
#     # 'attacks':{"values": ['my_attack']},
#     # 'defense':{"values": ['my_DnC']},
#     #  'poison_p':{"values": [0.2,0.4,0.6,0.8]}
#     # 'poison_layer':{"values":[['linear.2'],['linear.0','linear.2'],['convs.3','linear.0','linear.2'],['convs.0','convs.3','linear.0','linear.2']]},
#     },
#     }

if args.FL_type =='other_attacks':
    config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.5]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['grad_ascent','min_max','min_sum','noise']},
    'defense':{"values": ['foolsgold']},
    'attacks':{"values": ['lable']},
      },
    }
elif args.FL_type =='other_attacks_Fang':
    config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.5]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['grad_ascent','min_max','min_sum','noise']},
    'defense':{"values": ['fl_trust']},
    },
    }
elif args.FL_type =='FRL_label_flip_fang':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.5]},
    'm_r':{"values": [0.2,0]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['label']},
    'defense':{"values": ['fl_trust']},
    },
    }
elif args.FL_type =='FRP_defense':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.5]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['rank-reverse']},
    'defense':{"values": ['foolsgold']},
    },
    }
elif args.FL_type =='FRL_defense_Fang':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    'k':{"values": [0.5]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['rank-reverse']},
    'defense':{"values": ['fl_trust']},
    },
    }
elif args.FL_type =='FRL_fang':
       config={
    "method": "grid",
    "metric":{
        "goal": "minimize", 
        "name": "t_acc"
        },
    "parameters":{
    "lr":{"values": [0.01]},
    "nep":{"values": [50,30]},
    "max_t":{"values": [2000]},
    "iteration":{"values": [50,30]},
    "temp":{"values": [0.0010]},
    "noise":{"values": [1]},
    'k':{"values": [0.5]},
    'm_r':{"values": [0.2]},
    'non_iid':{"values": [1]},
    'attacks':{"values": ['my_attack']},
    'defense':{"values": ['fl_trust']},
    },
    }




def main():
    wandb.init()
    
    # config = wandb.config

    start_time = time.time() 

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
    # Make the a directory corresponding to this run for sval_loaderaving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/"+args.set+args.FL_type+f"~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"~try={i}"
            break
        i += 1

    (run_base_dir / "output.txt").write_text(str(args))
    args.run_base_dir = run_base_dir
    # args.FL_type=wandb.config.FL_type

    print(f"=> Saving data in {run_base_dir}")
    
    
    #distribute the dataset
    print ("dataset to use is: ", args.set)
    print ("number of FL clients: ", args.nClients)
    # print ("non-iid degree data distribution: ", args.non_iid_degree)
    print ("non-iid degree data distribution: ", wandb.config.non_iid)
    print ("batch size is : ", args.batch_size)
    print ("test batch size is: ", args.test_batch_size)
    
    data_distributer = getattr(data, args.set)()
    if args.FL_type == "FRL_defense_Fang" or args.FL_type =="FRL_fang" or args.FL_type =='other_attacks_Fang'or args.FL_type == 'FRL_label_flip_fang' or args.FL_type =='Reverse_mid_val'or args.FL_type =='other_attacks_agnostic_val' or args.FL_type =='FRL_train_agnostic_val':  
        tr_loaders = data_distributer.get_tr_loaders()    # len=10000 list
        te_loader = data_distributer.get_te_loader()
        val_loader = data_distributer.get_val_loader()
    else:
        tr_loaders = data_distributer.get_tr_loaders()
        te_loader = data_distributer.get_te_loader()
    
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print ("use_cuda: ", use_cuda) 
    
    #Federated Learning
    print ("type of FL: ", args.FL_type)
    if args.FL_type == "FRL":
        FRL_train(tr_loaders, te_loader)
    elif args.FL_type == "FedAVG":
        FedAVG(tr_loaders, te_loader)
    elif args.FL_type == "trimmedMean":
        Tr_Mean(tr_loaders, te_loader)
    elif args.FL_type == "Mkrum":
        Mkrum(tr_loaders, te_loader)
    elif args.FL_type =='other_attacks':
    # elif args.FL_type == "FRL_min_max" or args.FL_type == "FRL_min_sum" or args.FL_type =="FRL_noise" or args.FL_type =="FRL_grad_ascent":
        FRL_attacks(tr_loaders, te_loader)
    elif args.FL_type =="FRL_label_flip" or args.FL_type =="FRL_grad_ascent_label":
        FRL_train_label_flip_attack(tr_loaders, te_loader)
    elif args.FL_type =="FRL_label_flip_fang": 
        FRL_train_label_flip_attack_fang(tr_loaders, te_loader,val_loader)
    elif args.FL_type =="FRL_matrix_attack":
        FRL_matrix_attack(tr_loaders, te_loader)
    elif args.FL_type =="my_attack_defense":
        FRL_matrix_attack_defense(tr_loaders, te_loader)
    elif args.FL_type =="FRL_fang":
        FRL_matrix_attack_defense_val(tr_loaders, te_loader,val_loader)
    elif args.FL_type =="FRP_defense":
        FRL_train_defense(tr_loaders, te_loader)
    elif args.FL_type =="FRL_defense_Fang" :
        FRL_train_defense_val(tr_loaders, te_loader,val_loader)
    elif args.FL_type =="other_attacks_Fang" :
        other_attacks_val(tr_loaders, te_loader,val_loader)
    elif args.FL_type =="other_attacks_agnostic":
        FRL_attacks_agnostic(tr_loaders, te_loader)
    elif args.FL_type =="other_attacks_agnostic_val":
        FRL_attacks_agnostic_val(tr_loaders, te_loader,val_loader)
    elif args.FL_type =="my_attacks_agnostic":
        matrix_attack_defense_agnostic(tr_loaders, te_loader)
    elif args.FL_type =="FRL_train_agnostic":
        FRL_train_agnostic(tr_loaders, te_loader)
    elif args.FL_type =="FRL_train_agnostic_val":
        FRL_train_agnostic_val(tr_loaders, te_loader,val_loader)
    else:
        FedAVG(tr_loaders, te_loader)


    end_time = time.time()  # Record the end time
    total_time = end_time - start_time   
    print(total_time)
    
if __name__ == "__main__":
    # main()
    # Start sweep job.
    
    sweep_id = wandb.sweep(sweep=config, project="cifar-args")
    wandb.agent(sweep_id, function=main, count=400)