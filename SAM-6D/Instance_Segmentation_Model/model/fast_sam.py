from ultralytics import YOLO
from pathlib import Path
from typing import Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.utils.amg import MaskData
import logging
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import pytorch_lightning as pl
from ultralytics import yolo  # noqa
from ultralytics.nn.autobackend import AutoBackend


class CustomYOLO(YOLO):
    def __init__(
        self,
        model,
        iou,
        conf,
        max_det,
        segmentor_width_size,
        selected_device="cpu",
        verbose=False,
    ):
        YOLO.__init__(
            self,
            model,
        )
        self.overrides["iou"] = iou
        self.overrides["conf"] = conf
        self.overrides["max_det"] = max_det
        self.overrides["verbose"] = verbose
        self.overrides["imgsz"] = segmentor_width_size

        self.overrides["conf"] = 0.25
        self.overrides["mode"] = "predict"
        self.overrides["save"] = False

        self.predictor = yolo.v8.segment.SegmentationPredictor(
            overrides=self.overrides, _callbacks=self.callbacks
        )

        self.not_setup = True
        self.selected_device = selected_device
        logging.info(f"Init CustomYOLO done!")

    def setup_model(self, device, verbose=False):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        model = self.predictor.model or self.predictor.args.model
        self.predictor.args.half &= (
            device.type != "cpu"
        )  # half precision only supported on CUDA
        self.predictor.model = AutoBackend(
            model,
            device=device,
            dnn=self.predictor.args.dnn,
            data=self.predictor.args.data,
            fp16=self.predictor.args.half,
            fuse=True,
            verbose=verbose,
        )
        self.predictor.device = device
        self.predictor.model.eval()
        logging.info(f"Setup model at device {device} done!")

    def __call__(self, source=None, stream=False):
        return self.predictor(source=source, stream=stream)


class FastSAM(object):
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: dict = None,
        segmentor_width_size=None,
        device=None,
    ):
        self.model = CustomYOLO(
            model=checkpoint_path,
            iou=config.iou_threshold,
            conf=config.conf_threshold,
            max_det=config.max_det,
            selected_device=device,
            segmentor_width_size=segmentor_width_size,
        )
        self.segmentor_width_size = segmentor_width_size
        self.current_device = device
        logging.info(f"Init FastSAM done!")

    def postprocess_resize(self, detections, orig_size, update_boxes=False):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        if update_boxes:
            scale = orig_size[1] / self.segmentor_width_size
            detections["boxes"] = detections["boxes"].float() * scale
            detections["boxes"][:, [0, 2]] = torch.clamp(
                detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
            )
            detections["boxes"][:, [1, 3]] = torch.clamp(
                detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
            )
        return detections

    @torch.no_grad()
    def generate_masks(self, image) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
        detections = self.model(image)

        masks = detections[0].masks.data
        boxes = detections[0].boxes.data[:, :4]  # two lasts:  confidence and class

        # define class data
        mask_data = {
            "masks": masks.to(self.current_device),
            "boxes": boxes.to(self.current_device),
        }
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data
