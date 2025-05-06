python main.py --set MNIST --model "Conv2" --batch_size 32 --FL_global_epochs 1000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "other_attacks" 
python main.py --set MNIST --model "Conv2" --batch_size 32 --FL_global_epochs 1000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "FRP_defense" 
python main.py --set MNIST --model "Conv2" --batch_size 32 --FL_global_epochs 1000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "FRL_label_flip"  
python main.py --set MNIST --model "Conv2" --batch_size 32 --FL_globali_epochs 200 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "my_attack_defense" 
