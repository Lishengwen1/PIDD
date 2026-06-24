Code for "Physics-Informed Dataset Distillation"

## Dataset Distillation
1. Change the data paths paths in arguments/reproduce_pretrain.py and reproduce_distillation.py
2. Perform the pre-training process
```
python pretrain.py -d cifar10 
```

3. Perform the distillation process using PIDD
```
python PIDD.py -d cifar10 --type PIDD --ipc 50 
```

## Visualization
4. Feature Evolution on Energy Landscape 
```
cd visualization
python visualize.py
```

## Example Results

The folder `./results/cifar10/PIDD/` contains example results from running PIDD on CIFAR-10. 

The folder `./visualization/results/` contains example results from visualization. 

## Acknowledgement
Our code is built upon [IDC](https://github.com/snu-mllab/efficient-dataset-condensation) and [DANCE](https://github.com/Hansong-Zhang/DANCE)











