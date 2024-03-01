# Pose Estimation Model (PEM) for SAM-6D 



![image](https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_pem.png)

## Requirements
The code has been tested with
- python 3.9.6
- pytorch 2.0.0
- CUDA 11.3

Other dependencies:

```
sh dependencies.sh
```

## Data Preparation

Please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for more details.


## Model Download
Our trained model is provided [[here](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing)], and could be downloaded via the command:
```
python download_sam6d-pem.py
```

## Training on MegaPose Training Set

To train the Pose Estimation Model of SAM-6D, please prepare the training data and run the folowing command:
```
python train.py --gpus 0,1,2,3 --model pose_estimation_model --config config/base.yaml
```
By default, we use four GPUs of 3090ti to train the model with batchsize set as 28.


## Evaluation on BOP Datasets

To evaluate the model on BOP datasets, please run the following command:
```
python test_bop.py --gpus 0 --model pose_estimation_model --config config/base.yaml --dataset $DATASET --view 42
```
The string "DATASET" could be set as `lmo`, `icbin`, `itodd`, `hb`, `tless`, `tudl`, `ycbv`, or `all`. Before evaluation, please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for rendering the object templates of BOP datasets, or download our [rendered templates](https://drive.google.com/drive/folders/1fXt5Z6YDPZTJICZcywBUhu5rWnPvYAPI?usp=drive_link). Besides, the instance segmentation should be done following [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)]; to test on your own segmentation results, you could change the "detection_paths" in the `test_bop.py` file.

One could also download our trained model for evaluation:
```
python test_bop.py --gpus 0 --model pose_estimation_model --config config/base.yaml --checkpoint_path checkpoints/sam-6d-pem-base.pth --dataset $DATASET --view 42
```


## Acknowledgements
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [GDRNPP](https://github.com/shanice-l/gdrnpp_bop2022)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [Flatten Transformer](https://github.com/LeapLabTHU/FLatten-Transformer)

