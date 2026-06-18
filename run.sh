
# CIFAR-10
python pretrain.py -d cifar10 

python PIDD.py -d cifar10 --type PIDD --ipc 1 --num_premodel 5 --lambdac 0.5 --seed 0 --eps 0.1 --gpu 0 
python PIDD.py -d cifar10 --type PIDD --ipc 10 --num_premodel 5 --lambdac 0.01 --seed 0 --eps 1e-7 --gpu 0 
python PIDD.py -d cifar10 --type PIDD --ipc 50 --num_premodel 5 --lambdac 0.0005 --seed 0 --eps 1e-8 --gpu 0 
# python PIDD.py -d cifar10 --type PIDD --ipc 50 

# ImageNette
python pretrain.py -d ImageNette

python PIDD.py -d ImageNette --type PIDD --ipc 1 --num_premodel 5 --lambdac 1.0 --seed 0 --eps 0.1 --gpu 0 
python PIDD.py -d ImageNette --type PIDD --ipc 10 --num_premodel 5 --lambdac 0.01 --seed 0 --eps 1e-7 --gpu 0 


# Visualization
cd visualization
python visualize.py