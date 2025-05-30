# python main.py --set CIFAR10 --model "Conv8" --batch_size 32 --FL_global_epochs 500 --lr 0.4   --FL_type "Reverse_mid_certified" 
# python main.py --set Fashion --model "LeNet" --batch_size 32 --local_epochs 2 --FL_global_epochs 500 --lr 0.4  --FL_type "Reverse_mid_certified" 
# python main.py --set MNIST --model "Conv2" --batch_size 32 --local_epochs 2 --FL_global_epochs 500 --lr 0.5   --FL_type "Reverse_mid_certified" 
python main.py --set Texas --model "FCN" --batch_size 8 --round_nclients 20 --rand_mal_clients 20 --FL_global_epochs 1000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "Reverse_mid_certified" 
python main.py --set Location --model "FCN" --batch_size 8 --round_nclients 20 --rand_mal_clients 20 --FL_global_epochs 1000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "Reverse_mid_certified"
python main.py --set Purchase --model "FCN_Four" --batch_size --round_nclients 20 --rand_mal_clients 20 8 --FL_global_epochs 1000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "Reverse_mid_certified" 
