import os
import os.path as osp

def download_model(output_path):
    import os
    command = f"gdown --no-cookies --no-check-certificate -O '{output_path}/sam-6d-pem-base.pth' 1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7"
    os.system(command)

def download() -> None:
    root_dir = os.path.dirname((os.path.abspath(__file__)))
    save_dir = osp.join(root_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    download_model(save_dir)
    
if __name__ == "__main__":
    download()
    
    
