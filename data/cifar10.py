from args import args
from torchvision import datasets, transforms
from badnet import CIFAR10Poison
import torchvision
from data.Dirichlet_noniid import *
import wandb
from torch.utils.data import DataLoader, random_split

class CIFAR10:
    def __init__(self):

        
        args.output_size = 10
        
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_loc,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_loc,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        # tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=args.non_iid_degree, force=False)
        tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=wandb.config.noniid, force=False)

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
        if args.FL_type =="FRL_fang" or args.FL_type =="FRL_defense_Fang":
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
 
  
def build_poisoned_training_set(is_train, args):
    # transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    if args.set == 'CIFAR10':
        trainset = CIFAR10Poison(args, args.data_backdoor, train=is_train, download=True, transform=transform)
        nb_classes = 10
    # elif args.dataset == 'MNIST':
    #     trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    #     nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes


def build_testset(is_train, args):
    # transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    if args.set == 'CIFAR10':
        testset_clean = datasets.CIFAR10(args.data_backdoor, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_backdoor, train=is_train, download=True, transform=transform)
        nb_classes = 10
    # elif args.dataset == 'MNIST':
    #     testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
    #     testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    #     nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned

    



