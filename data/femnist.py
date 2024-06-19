from args import args
<<<<<<< HEAD
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
=======
import json
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import wandb
from PIL import Image
import numpy as np

class FEMNISTDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            image = np.array(image)  # Convert the list to a numpy array
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label

class FEMNIST:
    def __init__(self):
        self.data_loc = 'leaf/data/femnist/data'
        args.output_size = 62  # FEMNIST has 62 classes

        Mytransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Adjust normalization if necessary
        ])

        # Load client data for training
        client_data = self.load_femnist(self.data_loc)

        self.tr_loaders = []
        for client_id, (client_images, client_labels) in client_data.items():
            train_dataset = FEMNISTDataset(client_images, client_labels, transform=Mytransform)
            batch_size = args.batch_size
            self.tr_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))

        # Load test data
        test_data, test_targets = self.load_femnist(os.path.join(self.data_loc, 'test'), test=True)
        test_dataset = FEMNISTDataset(test_data, test_targets, transform=Mytransform)
        self.te_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

        # Setup validation loader if needed
        if args.FL_type in ["FRL_fang", "FRL_defense_Fang", "other_attacks_Fang", "FRL_label_flip_fang", "Reverse_mid_val", "other_attacks_agnostic_val", "FRL_train_agnostic_val"]:
            validation_size = 100
            test_size = len(test_dataset) - validation_size
            validation_dataset, test_dataset = random_split(test_dataset, [validation_size, test_size])
            self.validation_loader = DataLoader(validation_dataset, batch_size=args.test_batch_size, shuffle=False)
        else:
            self.validation_loader = None

    def load_femnist(self, data_dir, test=False):
        data = []
        targets = []
        client_data = {}
        
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(data_dir, file_name), 'r') as f:
                    file_data = json.load(f)
                    for user in file_data['users']:
                        user_images = file_data['user_data'][user]['x']
                        user_labels = file_data['user_data'][user]['y']
                        if test:
                            data.extend(user_images)
                            targets.extend(user_labels)
                        else:
                            client_data[user] = (user_images, user_labels)
                            
        if test:
            return data, targets
        return client_data

    def get_tr_loaders(self):
        return self.tr_loaders

    def get_te_loader(self):
        return self.te_loader

    def get_val_loader(self):
        return self.validation_loader if hasattr(self, 'validation_loader') else None
>>>>>>> 536798a (update agrs)
