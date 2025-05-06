import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import wandb
from args import args
from torchvision import datasets, transforms
import torchvision
from data.Dirichlet_noniid import *
from data.iid import *
# import numpy as np
# import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader, random_split

class PurchaseDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label

class Purchase:
    def __init__(self):
        # Load dataset
        data = np.load("/home/test/FRL/purchase/purchase.npz")
        features = data['arr_0']  # Features (e.g., purchase history)
        labels = data['arr_1']    # Labels (e.g., customer segments)
        labels -= 1  # Adjust labels if necessary (e.g., zero-indexing)

        # Split dataset
        train_features = features[:180000]
        train_labels = labels[:180000]
        test_features = features[180000:]
        test_labels = labels[180000:]

        # Convert to tensors
        train_features = torch.tensor(train_features, dtype=torch.float)
        test_features = torch.tensor(test_features, dtype=torch.float)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        # Transformations (if any)
        Mytransform = transforms.Compose([
            # Add your transformations here if necessary
        ])

        # Create datasets
        train_dataset = PurchaseDataset(features=train_features, labels=train_labels, transform=Mytransform)
        test_dataset = PurchaseDataset(features=test_features, labels=test_labels, transform=Mytransform)

        # Sampling logic (modify according to dataset specifics)
        if wandb.config.non_iid == 'iid':
            tr_per_participant_list = sample_iid_train_data(train_dataset, args.nClients)
        else:
            tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(
                train_dataset, args.nClients, alpha=wandb.config.non_iid, force=False
            )
        
        # Create DataLoaders for training data
        self.tr_loaders = []
        tr_count = 0
        for pos, indices in tr_per_participant_list.items():
            if len(indices) == 1 or len(indices) == 0:
                print(pos)
            tr_count += len(indices)
            batch_size = args.batch_size
            self.tr_loaders.append(get_train(train_dataset, indices, args.batch_size))
        
        # Split and create DataLoaders for validation and test data if needed
        if args.FL_type in ["FRL_fang", "FRL_defense_Fang", 'other_attacks_Fang', 'FRL_label_flip_fang', 
                            'Reverse_mid_val', 'other_attacks_agnostic_val', 'FRL_train_agnostic_val']:
            validation_size = 100  # Adjust if necessary
            test_size = len(test_dataset) - validation_size

            validation_dataset, test_dataset = random_split(test_dataset, [validation_size, test_size])
            self.validation_loader = DataLoader(validation_dataset, batch_size=args.test_batch_size, shuffle=False)
            self.te_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        else:
            self.te_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        ###test#########
        # print(train_features.shape)  # Should be (num_samples, num_features)
        # print(train_labels.shape)    # Should be (num_samples,)

        # print(np.unique(train_labels, return_counts=True))
        # for loader in self.tr_loaders:  # Iterate over each DataLoader in the list
        #     for batch in loader:  # Iterate over batches in the current DataLoader
        #         features, labels = batch
        #         print(features.shape, labels.shape)  # Print batch shapes
        #         break  # Only check the first batch
        #     break  # Only check the first DataLoader

        ################################
    def get_tr_loaders(self):
        return self.tr_loaders
    
    def get_te_loader(self):
        return self.te_loader
    
    def get_val_loader(self):
        return self.validation_loader if hasattr(self, 'validation_loader') else None