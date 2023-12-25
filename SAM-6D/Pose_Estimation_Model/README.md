# Pose Estimation Model (PEM) for SAM-6D 



![image](https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_pem.png)

## Requirements
The code has been tested with
- python 3.7.6
- pytorch 1.9.0
- CUDA 11.3

Other dependencies:

```
sh dependencies.sh
```

## Data Preparation

Please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for more details.


## Training on MegaPose Training Set

To train the Pose Estimation Model of SAM-6D, please prepare the training data and run the folowing command:
```
python train.py --gpus 0,1 --model pose_estimation_model --config config/base.yaml
```
By default, we use two GPUs of 3090ti to train the model.


## Evaluation on BOP Datasets

To evaluate the model on BOP datasets, please run the following command:
```
python test_bop.py --gpus 0 --model pose_estimation_model --config config/base.yaml --iter 600000 --dataset $DATASET
```
or 
```
python test_bop.py --gpus 0 --model pose_estimation_model --config config/base.yaml --checkpoint_path $CHECK_POINT --dataset $DATASET
```
The string "DATASET" could be set as `lmo`, `icbin`, `itodd`, `hb`, `tless`, `tudl`, `ycbv`, or `all`.

## Acknowledgement
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [GDRNPP](https://github.com/shanice-l/gdrnpp_bop2022)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [Flatten Transformer](https://github.com/LeapLabTHU/FLatten-Transformer)

