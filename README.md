# <p align="center"> <font color=#008000>SAM-6D</font>: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation </p>

####  <p align="center"> [Jiehong Lin](https://jiehonglin.github.io/), [Lihua Liu](https://github.com/foollh), [Dekun Lu](https://github.com/WuTanKun), [Kui Jia](http://kuijia.site/)</p>
#### <p align="center">CVPR 2024 </p>
#### <p align="center">[[Paper]](https://arxiv.org/abs/2311.15707) </p>

<p align="center">
  <img width="100%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/vis.gif"/>
</p>


## News
- [2024/03/07] We publish an updated version of our paper on [ArXiv](https://arxiv.org/abs/2311.15707).
- [2024/02/29] Our paper is accepted by CVPR2024!


## Update Log
- [2024/03/05] We update the demo to support [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), you can do this by specifying `SEGMENTOR_MODEL=fastsam` in demo.sh.
- [2024/03/03] We upload a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for running custom data.
- [2024/03/01] We update the released [model](https://drive.google.com/file/d/1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7/view?usp=sharing) of PEM. For the new model, a larger batchsize of 32 is set, while that of the old is 12. 

## Overview
In this work, we employ Segment Anything Model as an advanced starting point for **zero-shot 6D object pose estimation** from RGB-D images, and propose a novel framework, named **SAM-6D**, which utilizes the following two dedicated sub-networks to realize the focused task:
- [x] [Instance Segmentation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Instance_Segmentation_Model)
- [x] [Pose Estimation Model](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Pose_Estimation_Model)


<p align="center">
  <img width="50%" src="https://github.com/JiehongLin/SAM-6D/blob/main/pics/overview_sam_6d.png"/>
</p>


## Getting Started

### 0. Prerequisites
- **Linux** (tested on Ubuntu 20.04 and 22.04)
- **NVIDIA drivers** (tested with 525 and 535)
- **CUDA** (tested with 12.0 and 12.2)
- **Anaconda** (see below for Installation)
- If running on **non-local machines**, i.e. cloud servers or VMs, ensure that x11 is available: `sudo apt install xorg`

<details>
<summary>Optional: Installation of Anaconda</summary>

Run the code below, to install the latest version of Miniconda, or refer to the [Anaconda documentation](https://docs.anaconda.com/) to install Anaconda.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh &&
chmod +x ~/miniconda.sh &&
bash ~/miniconda.sh -b -p ~/miniconda &&
rm ~/miniconda.sh &&
source ~/miniconda/bin/activate &&
conda init --all &&
source ~/.bashrc
```
</details>


### 1. Preparation
Please clone the repository locally:
```shell
git clone https://github.com/JiehongLin/SAM-6D.git
cd SAM-6D/SAM-6D
```
Install the environment:
```shell
conda env create -f environment.yaml
conda activate sam6d
```
Download the model checkpoints:
```shell
sh prepare.sh
```

We also provide a [docker image](https://hub.docker.com/r/lihualiu/sam-6d/tags) for convenience.

### 2. Run Inference on the example or custom data
To run inference the following **environment variables** must be set:

- `$CAD_PATH` path to a given cad model(mm)
- `$RGB_PATH` path to a given RGB image
- `$DEPTH_PATH` path to a given depth map(mm)
- `$CAMERA_PATH` path to given camera intrinsics
- `$OUTPUT_DIR` path to a pre-defined file for saving results


Run inference on the [**example data**](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data/Example):
```shell
# export example variables, must be executed from SAM-6D/SAM-6D directory containg the Data folder
export CAD_PATH=$PWD/Data/Example/obj_000005.ply
export RGB_PATH=$PWD/Data/Example/rgb.png
export DEPTH_PATH=$PWD/Data/Example/depth.png
export CAMERA_PATH=$PWD/Data/Example/camera.json
export OUTPUT_DIR=$PWD/Data/Example/outputs

sh demo.sh
```
All output will be saved unter the `$OUTPUT_DIR`.

To run inference on **custom data**, export the environment pointing to your data and then run `sh demo.sh`.

## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }


## Contact

If you have any questions, please feel free to contact the authors. 

Jiehong Lin: [mortimer.jh.lin@gmail.com](mailto:mortimer.jh.lin@gmail.com)

Lihua Liu: [lihualiu.scut@gmail.com](mailto:lihualiu.scut@gmail.com)

Dekun Lu: [derkunlu@gmail.com](mailto:derkunlu@gmail.com)

Kui Jia:  [kuijia@gmail.com](kuijia@gmail.com)

