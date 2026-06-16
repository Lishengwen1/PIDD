Code for "Physics-Informed Dataset Distillation"

## Getting Started
1. Change the data paths and results paths in arguments/reproduce_xxxx.py
2. Perform the pre-training process
```
python pretrain.py -d cifar10 
```

3. Perform the condensation process using PIDD
```
python PIDD.py -d cifar10 --ipc 50 --factor 2 
```












