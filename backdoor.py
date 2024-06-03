# Import the necessary functions and classes
from data.cifar10 import build_poisoned_training_set
from args import args
# Set the necessary parameters
is_train = True  # or False depending on whether it's a training set or test set
# args = {
#     'dataset': 'CIFAR10',  # or 'MNIST' depending on the dataset you want to use
#     'data_path': args.data_backdoor, # Path to your dataset
#     'nb_classes': 10  # Number of classes in your dataset
# }

# Call the function
trainset, nb_classes = build_poisoned_training_set(is_train, args)
print('finish')