import numpy as np
import torch
import torch.nn.functional as F
import logging
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from copy import deepcopy
from typing import Tuple


class CustomResizeLongestSide(ResizeLongestSide):
    def __init__(self, target_length: int, dividable_size: int) -> None:
        ResizeLongestSide.__init__(
            self,
            target_length=target_length,
        )
        self.dividable_size = dividable_size

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int, dividable_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        (newh, neww) = make_bbox_dividable((newh, neww), dividable_size)
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0],
            image.shape[1],
            self.target_length,
            dividable_size=self.dividable_size,
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[2],
            image.shape[3],
            self.target_length,
            dividable_size=self.dividable_size,
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length, self.dividable_size
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)


class CropResizePad:
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)

    def __call__(self, images, boxes):
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factor = self.target_max / torch.max(box_sizes, dim=-1)[0]
        processed_images = []
        for image, box, scale in zip(images, boxes, scale_factor):
            # crop and scale
            image = image[:, box[1] : box[3], box[0] : box[2]]
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale.item())[0]
            # pad and resize
            original_h, original_w = image.shape[1:]
            original_ratio = original_w / original_h

            # check if the original and final aspect ratios are the same within a margin
            if self.target_ratio != original_ratio:
                padding_top = max((self.target_h - original_h) // 2, 0)
                padding_bottom = self.target_h - original_h - padding_top
                padding_left = max((self.target_w - original_w) // 2, 0)
                padding_right = self.target_w - original_w - padding_left
                image = F.pad(
                    image, (padding_left, padding_right, padding_top, padding_bottom)
                )
            assert image.shape[1] == image.shape[2], logging.info(
                f"image {image.shape} is not square after padding"
            )
            image = F.interpolate(
                image.unsqueeze(0), scale_factor=self.target_h / image.shape[1]
            )[0]
            processed_images.append(image)
        return torch.stack(processed_images)


def xyxy_to_xywh(bbox):
    if len(bbox.shape) == 1:
        """Convert [x1, y1, x2, y2] box format to [x, y, w, h] format."""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    elif len(bbox.shape) == 2:
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")


def xywh_to_xyxy(bbox):
    """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
    if len(bbox.shape) == 1:
        x, y, w, h = bbox
        return [x, y, x + w - 1, y + h - 1]
    elif len(bbox.shape) == 2:
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x, y, x + w, y + h], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")


def get_bbox_size(bbox):
    return [bbox[2] - bbox[0], bbox[3] - bbox[1]]


def make_bbox_dividable(bbox_size, dividable_size, ceil=True):
    if ceil:
        new_size = np.ceil(np.array(bbox_size) / dividable_size) * dividable_size
    else:
        new_size = np.floor(np.array(bbox_size) / dividable_size) * dividable_size
    return new_size


def make_bbox_square(old_bbox):
    size_to_fit = np.max([old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]])
    new_bbox = np.array(old_bbox)
    old_bbox_size = [old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]
    # Add padding into y axis
    displacement = int((size_to_fit - old_bbox_size[1]) / 2)
    new_bbox[1] = old_bbox[1] - displacement
    new_bbox[3] = old_bbox[3] + displacement
    # Add padding into x axis
    displacement = int((size_to_fit - old_bbox_size[0]) / 2)
    new_bbox[0] = old_bbox[0] - displacement
    new_bbox[2] = old_bbox[2] + displacement
    return new_bbox


def crop_image(image, bbox, format="xyxy"):
    if format == "xyxy":
        image_cropped = image[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
    elif format == "xywh":
        image_cropped = image[
            bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
        ]
    return image_cropped


def force_binary_mask(mask, threshold=0.):
    mask = np.where(mask > threshold, 1, 0)
    return mask

def get_bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def compute_iou(bb_a, bb_b):
    """Calculates the Intersection over Union (IoU) of two 2D bounding boxes.

    :param bb_a: 2D bounding box (x1, y1, x2, y2) -- see calc_2d_bbox.
    :param bb_b: 2D bounding box (x2, y2, x2, y2) -- see calc_2d_bbox.
    :return: The IoU value.
    """

    # Intersection rectangle.
    tl_inter = torch.max(bb_a[:, 0:2], bb_b[:, 0:2])
    br_inter = torch.min(bb_a[:, 2:4], bb_b[:, 2:4])
    wh_a = bb_a[:, 2:4] - bb_a[:, 0:2]
    wh_b = bb_b[:, 2:4] - bb_b[:, 0:2]

    # Width and height of the intersection rectangle.
    wh_inter = br_inter - tl_inter

    if (wh_inter > 0).all():
        area_inter = wh_inter[:, 0] * wh_inter[:, 1]
        area_a = wh_a[:, 0] * wh_a[:, 1]
        area_b = wh_b[:, 0] * wh_b[:, 1]
        iou = area_inter / (area_a + area_b - area_inter)
    else:
        iou = 0.0
    
    return iou