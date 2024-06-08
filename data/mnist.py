from args import args
from torchvision import datasets, transforms
import torchvision
from data.Dirichlet_noniid import *
# import numpy as np
# import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader, random_split

class MNIST:
    def __init__(self):
        
        args.output_size = 10
        
        Mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_dataset = datasets.MNIST(root=args.data_loc, train=True, download=True, transform=Mytransform)

        test_dataset = datasets.MNIST(root=args.data_loc, train=False, download=True, transform=Mytransform)

       


        # tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=args.non_iid_degree, force=False)
        tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=wandb.config.non_iid, force=False)
        self.tr_loaders = []
        tr_count = 0
        for pos, indices in tr_per_participant_list.items():
            if len(indices)==1 or len(indices)==0:
                print (pos)
            tr_count += len(indices)
            batch_size = args.batch_size
            self.tr_loaders.append(get_train(train_dataset, indices, args.batch_size))
#         print ("number of total training points:" ,tr_count)
        # self.te_loader= torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        if args.FL_type =="FRL_fang" or args.FL_type =="FRL_defense_Fang"or args.FL_type =='other_attacks_Fang':
             # validation_size = int(0.2 * len(test_dataset)) 
            validation_size = 100 
            test_size = len(test_dataset) - validation_size

            validation_dataset, test_dataset = random_split(test_dataset, [validation_size, test_size])
            self.validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.test_batch_size, shuffle=False)
            self.te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        else:
            self.te_loader= torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)


    def get_tr_loaders(self):
        return self.tr_loaders
    
    def get_te_loader(self):
        return self.te_loader
    
    def get_val_loader(self):
        return self.validation_loader