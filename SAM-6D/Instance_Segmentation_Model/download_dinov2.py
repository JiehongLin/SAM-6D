import os
import logging
import os, sys
import os.path as osp
from utils.inout import get_root_project

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
model_dict = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "dinov2_vitg14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    }

def download_model(url, output_path):
    import os

    command = f"wget -O {output_path}/{url.split('/')[-1]} {url} --no-check-certificate"
    os.system(command)


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="download",
)
def download(cfg: DictConfig) -> None:
    model_name = "dinov2_vitl14" # default segmentation model used in CNOS
    save_dir = osp.join(get_root_project(), "checkpoints/dinov2")
    os.makedirs(save_dir, exist_ok=True)
    download_model(model_dict[model_name], save_dir)
    
if __name__ == "__main__":
    download()
    
    
