Code for "Physics-Informed Dataset Distillation"

## Dataset Distillation
1. Change the data paths and results paths in arguments/reproduce_xxxx.py
2. Perform the pre-training process
```
python pretrain.py -d cifar10 
```

3. Perform the condensation process using PIDD
```
python PIDD.py -d cifar10 --ipc 50 --factor 2 
```

## Visualization
4. Feature Evolution on Energy Landscape 
```
cd visualization
python visualize.py
```

## Acknowledgement
Our code is built upon [IDC](https://github.com/snu-mllab/efficient-dataset-condensation) and [DANCE](https://github.com/Hansong-Zhang/DANCE)











