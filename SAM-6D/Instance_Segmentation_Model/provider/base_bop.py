import logging, os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
from utils.inout import load_json, save_json, casting_format_to_save_json
from utils.poses.pose_utils import load_index_level_in_level2
import torch
from utils.bbox_utils import CropResizePad
import pytorch_lightning as pl

pl.seed_everything(2023)

OBJ_IDS = {
    "icbin": [1, 2],
    "ycbv": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "tudl": [1, 2, 3],
    "lmo": [1, 5, 6, 8, 9, 10, 11, 12],
    "tless": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    "itodd": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    "hb": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
}


class BaseBOP(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        **kwargs,
    ):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = root_dir
        self.split = split

    def load_list_scene(self, split=None):
        if isinstance(split, str):
            if split is not None:
                split_folder = osp.join(self.root_dir, split)
            self.list_scenes = sorted(
                [
                    osp.join(split_folder, scene)
                    for scene in os.listdir(split_folder)
                    if os.path.isdir(osp.join(split_folder, scene))
                    and scene != "models"
                ]
            )
        elif isinstance(split, list):
            self.list_scenes = []
            for scene in split:
                if not isinstance(scene, str):
                    scene = f"{scene:06d}"
                if os.path.isdir(osp.join(self.root_dir, scene)):
                    self.list_scenes.append(osp.join(self.root_dir, scene))
            self.list_scenes = sorted(self.list_scenes)
        else:
            raise NotImplementedError
        logging.info(f"Found {len(self.list_scenes)} scenes")

    def load_scene(self, path, use_visible_mask=True):
        # Load rgb and mask images
        rgb_paths = sorted(Path(path).glob("rgb/*.[pj][pn][g]"))
        if use_visible_mask:
            mask_paths = sorted(Path(path).glob("mask_visib/*.[pj][pn][g]"))
        else:
            mask_paths = sorted(Path(path).glob("mask/*.[pj][pn][g]"))
        # load poses
        scene_gt = load_json(osp.join(path, "scene_gt.json"))
        scene_gt_info = load_json(osp.join(path, "scene_gt_info.json"))
        scene_camera = load_json(osp.join(path, "scene_camera.json"))
        return {
            "rgb_paths": rgb_paths,
            "mask_paths": mask_paths,
            "scene_gt": scene_gt,
            "scene_gt_info": scene_gt_info,
            "scene_camera": scene_camera,
        }

    def load_metaData(self, reset_metaData, mode="query", split="test", level=2):
        start_time = time.time()
        if mode == "query":
            metaData = {
                "scene_id": [],
                "frame_id": [],
                "rgb_path": [],
                "depth_path": [],
                "intrinsic": [],
            }
            logging.info(f"Loading metaData for split {split}")
            metaData_path = osp.join(self.root_dir, f"{split}_metaData.json")
            if reset_metaData:
                for scene_path in tqdm(self.list_scenes, desc="Loading metaData"):
                    scene_id = scene_path.split("/")[-1]
                    if osp.exists(osp.join(scene_path, "rgb")):
                        rgb_paths = sorted(Path(scene_path).glob("rgb/*.[pj][pn][g]"))
                        depth_paths = sorted(
                            Path(scene_path).glob("depth/*.[pj][pn][g]")
                        )
                    else:
                        rgb_paths = sorted(Path(scene_path).glob("gray/*.tif"))
                        depth_paths = sorted(Path(scene_path).glob("depth/*.tif"))
                    # assert len(rgb_paths) == len(depth_paths), f"{scene_path} rgb and depth mismatch"

                    depth_paths = [str(x) for x in depth_paths]
                    video_metaData = {}

                    # load poses
                    for json_name in ["scene_camera"]:
                        json_path = osp.join(scene_path, json_name + ".json")
                        if osp.exists(json_path):
                            video_metaData[json_name] = load_json(json_path)
                        else:
                            video_metaData[json_name] = None
                    assert len(rgb_paths) > 0, f"{scene_path} is empty"
                    for idx_frame in range(len(rgb_paths)):
                        # get rgb path
                        rgb_path = rgb_paths[idx_frame]
                        # get id frame
                        id_frame = int(str(rgb_path).split("/")[-1].split(".")[0])
                        depth_path = osp.join(
                            scene_path, "depth", f"{id_frame:06d}.png"
                        )
                        if depth_path in depth_paths:
                            metaData["depth_path"].append(depth_path)
                        else:
                            metaData["depth_path"].append(None)
                        metaData["scene_id"].append(scene_id)
                        metaData["frame_id"].append(id_frame)
                        metaData["rgb_path"].append(str(rgb_path))
                        metaData["intrinsic"].append(
                            video_metaData["scene_camera"][f"{id_frame}"]["cam_K"]
                        )
                # casting format of metaData
                metaData = casting_format_to_save_json(metaData)
                save_json(metaData_path, metaData)
            else:
                metaData = load_json(metaData_path)
        elif mode == "template":
            list_obj_ids, list_idx_template = [], []
            for obj_id in self.obj_ids:
                for idx_template in range(len(self.templates_poses)):
                    list_obj_ids.append(obj_id)
                    list_idx_template.append(idx_template)
            metaData = {
                "obj_id": list_obj_ids,
                "idx_template": list_idx_template,
            }

        self.metaData = pd.DataFrame.from_dict(metaData, orient="index")
        self.metaData = self.metaData.transpose()
        # # shuffle data
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index(
            drop=True
        )
        finish_time = time.time()
        logging.info(
            f"Finish loading metaData of size {len(self.metaData)} in {finish_time - start_time:.2f} seconds"
        )
        return

    def get_obj_ids(self, template_dir):
        obj_ids = None
        dataset_name = osp.basename(template_dir)
        if dataset_name in OBJ_IDS.keys():
            obj_ids = OBJ_IDS[dataset_name]
        return obj_ids

    def __len__(self):
        return len(self.metaData)
