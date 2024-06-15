from args import args
from torchvision import datasets, transforms
import torchvision
from data.Dirichlet_noniid import *
# import numpy as np
# import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import h5py
import random

class femnist_base(Dataset):
    def __init__(self, img_dir='./femnist/femnist_digits.hdf5', transform=None, target_transform=None, train=True, client_id=None):
        self.dataset = h5py.File(img_dir, 'r')
        self.writers = sorted(self.dataset.keys())
        self.data_idx=[]
        random.seed(123)
        random.shuffle(self.data_idx)
        if train:
            # for writer in self.writers[client_id]:
            for idx in range(len(self.dataset[self.writers[client_id]]['images'])):
                self.data_idx.append((self.writers[client_id], idx))
            len_writer=len(self.data_idx)
            self.data_idx=self.data_idx[0:int(len_writer*0.8)]
        else:
            for writer in self.writers:
                len_data=len(self.dataset[writer]['images'])
                for idx in range(int(len_data*0.8), len_data, 1):
                    self.data_idx.append((writer, idx))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        (ida, idb)=self.data_idx[idx]
        image = self.dataset[ida]['images'][idb]
        label = self.dataset[ida]['labels'][idb]
        # image = torch.tensor(image)
        # label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class FEMNIST:
    def __init__(self):
        
        args.output_size = 10
        
        Mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # train_dataset = femnist_base(train=True, transform=Mytransform)

        test_dataset = femnist_base(train=False, transform=Mytransform)

        # tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=args.non_iid_degree, force=False)
        # tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=wandb.config.non_iid, force=False)
        self.tr_loaders = []
        # tr_count = 0
        for indices in range(args.nClients):
            # if len(indices)==1 or len(indices)==0:
            #     print (pos)
            # tr_count += len(indices)
            batch_size = args.batch_size
            # self.tr_loaders.append(get_train(train_dataset, indices, args.batch_size))
            self.tr_loaders.append(
                torch.utils.data.DataLoader(
                    femnist_base(train=True, client_id=indices, transform=Mytransform), 
                    batch_size=args.batch_size, shuffle=False
                )
            )
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