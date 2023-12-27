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

def download_model(output_path):
    import os
    command = f"gdown --no-cookies --no-check-certificate -O '{output_path}/FastSAM-x.pt' 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv"
    os.system(command)


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="download",
)
def download(cfg: DictConfig) -> None:
    save_dir = osp.join(get_root_project(), "checkpoints/FastSAM")
    os.makedirs(save_dir, exist_ok=True)
    download_model(save_dir)
    
if __name__ == "__main__":
    download()
    
    
