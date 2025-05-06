python main.py --set Purchase --model "FCN_Four" --batch_size 32 --FL_global_epochs 1000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "Reverse_mid" 
python main.py --set Purchase --model "FCN_Four" --batch_size 32 --FL_global_epochs 3000 --lr 0.4  --data_loc "./MNIST/data/" --data_backdoor "./MNIST/data/backdoor" --FL_type "FRL" 
