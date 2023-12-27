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
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # 2560 MB
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # 1250  MB
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }  # 375 GB

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
    model_name = "vit_h" # default segmentation model used in CNOS
    save_dir = osp.join(get_root_project(), "checkpoints/segment-anything")
    os.makedirs(save_dir, exist_ok=True)
    download_model(model_dict[model_name], save_dir)
    
if __name__ == "__main__":
    download()
    
    
