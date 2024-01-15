# Instance Segmentation Model (ISM) for SAM-6D 


## Requirements
The code has been tested with
- python 3.9.6
- pytorch 2.0.0
- CUDA 11.3

Create conda environment:

```
conda env create -f environment.yml
conda activate sam6d-ism

# for using SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# for using fastSAM
pip install ultralytics==8.0.135
```


## Data Preparation

Please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for more details.


## Foundation Model Download

Download model weights of [Segmenting Anything](https://github.com/facebookresearch/segment-anything):
```
python download_sam.py
```

Download model weights of [Fast Segmenting Anything](https://github.com/CASIA-IVA-Lab/FastSAM):
```
python download_fastsam.py
```

Download model weights of ViT pre-trained by [DINOv2](https://github.com/facebookresearch/dinov2):
```
python download_dinov2.py
```


## Evaluation on BOP Datasets

To evaluate the model on BOP datasets, please run the following commands:

```
# Specify a specific GPU
export CUDA_VISIBLE_DEVICES=0

# with sam
python run_inference.py dataset_name=$DATASET

# with fastsam
python run_inference.py dataset_name=$DATASET model=ISM_fastsam
```

The string "DATASET" could be set as `lmo`, `icbin`, `itodd`, `hb`, `tless`, `tudl` or `ycbv`.


## Acknowledgements

- [CNOS](https://github.com/nv-nguyen/cnos)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- [DINOv2](https://github.com/facebookresearch/dinov2)

                                                              
