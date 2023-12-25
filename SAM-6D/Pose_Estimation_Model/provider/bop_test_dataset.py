
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as cocomask

import torch
import torchvision.transforms as transforms

from data_utils import (
    get_model_info,
    get_bop_depth_map,
    get_bop_image,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from bop_object_utils import load_objs


class BOPTestset():
    def __init__(self, cfg, eval_dataset_name='ycbv', detetion_path=None):
        assert detetion_path is not None

        self.cfg = cfg
        self.dataset = eval_dataset_name
        self.data_dir = cfg.data_dir
        self.rgb_mask_flag = cfg.rgb_mask_flag
        self.img_size = cfg.img_size
        self.n_sample_observed_point = cfg.n_sample_observed_point
        self.n_sample_model_point = cfg.n_sample_model_point
        self.n_sample_template_point = cfg.n_sample_template_point
        self.n_template_view = cfg.n_template_view
        self.minimum_n_point = cfg.minimum_n_point
        self.seg_filter_score = cfg.seg_filter_score
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

        if eval_dataset_name == 'tless':
            model_path = 'models_cad'
        else:
            model_path = 'models'
        self.template_folder = os.path.join(cfg.template_dir, eval_dataset_name)

        self.data_folder = os.path.join(self.data_dir, eval_dataset_name, 'test')
        self.model_folder = os.path.join(self.data_dir, eval_dataset_name, model_path)
        obj, obj_ids = load_objs(self.model_folder, self.template_folder, sample_num=self.n_sample_model_point, n_template_view=self.n_template_view)
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.objects = obj
        self.obj_idxs = obj_idxs

        with open(detetion_path) as f:
            dets = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation

        self.det_keys = []
        self.dets = {}
        for det in tqdm(dets, 'processing detection results'):
            scene_id = det['scene_id'] # data type: int
            img_id = det['image_id'] # data type: int
            key = str(scene_id).zfill(6) + '_' + str(img_id).zfill(6)
            if key not in self.det_keys:
                self.det_keys.append(key)
                self.dets[key] = []
            self.dets[key].append(det)
        del dets
        print('testing on {} images on {}...'.format(len(self.det_keys), eval_dataset_name))

    def __len__(self):
        return len(self.det_keys)

    def __getitem__(self, index):
        dets = self.dets[self.det_keys[index]]

        instances = []
        for det in dets:
            if det['score']>self.seg_filter_score:
                instance = self.get_instance(det)
                if instance is not None:
                    instances.append(instance)

        ret_dict = {}
        for key in instances[0].keys():
            ret_dict[key] = torch.stack([instance[key] for instance in instances])
        ret_dict['scene_id'] = torch.IntTensor([int(self.det_keys[index][0:6])])
        ret_dict['img_id'] = torch.IntTensor([int(self.det_keys[index][7:13])])
        ret_dict['seg_time'] = torch.FloatTensor([dets[0]['time']])
        return ret_dict

    def get_instance(self, data):
        scene_id = data['scene_id'] # data type: int
        img_id = data['image_id'] # data type: int
        obj_id = data['category_id'] # data type: int
        bbox = data['bbox'] # list, len:4
        seg = data['segmentation'] # keys: counts, size
        score = data['score']

        scene_folder = os.path.join(self.data_folder, f'{scene_id:06d}')
        scene_camera = json.load(open(os.path.join(scene_folder, 'scene_camera.json')))
        K = np.array(scene_camera[str(img_id)]['cam_K']).reshape((3, 3)).copy()
        depth_scale = scene_camera[str(img_id)]['depth_scale']
        inst = dict(scene_id=scene_id, img_id=img_id, data_folder=self.data_folder)

        obj_idx = self.obj_idxs[obj_id]
        model_points, _ = get_model_info(self.objects[obj_idx])

        # depth
        depth = get_bop_depth_map(inst) * depth_scale

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, depth > 0)
        if np.sum(mask) > self.minimum_n_point:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            return None
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = get_point_cloud_from_depth(depth, K, [y1, y2, x1, x2])
        cloud = cloud.reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < self.objects[obj_idx].diameter * 0.6
        if np.sum(flag) < self.minimum_n_point:
            return None
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb
        rgb = get_bop_image(inst, [y1,y2,x1,x2], self.img_size, mask if self.rgb_mask_flag else None)
        rgb = self.transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(cloud)
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['rgb_choose'] = torch.IntTensor(rgb_choose).long()
        ret_dict['obj'] = torch.IntTensor([obj_idx]).long()
        ret_dict['model'] = torch.FloatTensor(model_points)

        ret_dict['obj_id'] = torch.IntTensor([obj_id])
        ret_dict['score'] = torch.FloatTensor([score])
        return ret_dict


    def _get_template(self, obj, tem_index=1):
        rgb, mask, xyz = obj.get_template(tem_index)

        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))

        choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point, replace=False)
        choose = choose[choose_idx]
        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)
        return rgb, rgb_choose, xyz


    def get_templates(self):
        n_template_view = self.n_template_view
        all_tem_rgb = [[] for i in range(n_template_view)]
        all_tem_choose = [[] for i in range(n_template_view)]
        all_tem_pts = [[] for i in range(n_template_view)]

        for obj in self.objects:
            for i in range(n_template_view):
                tem_rgb, tem_choose, tem_pts = self._get_template(obj, i)
                all_tem_rgb[i].append(torch.FloatTensor(tem_rgb))
                all_tem_choose[i].append(torch.IntTensor(tem_choose).long())
                all_tem_pts[i].append(torch.FloatTensor(tem_pts))

        for i in range(n_template_view):
            all_tem_rgb[i] = torch.stack(all_tem_rgb[i]).cuda()
            all_tem_choose[i] = torch.stack(all_tem_choose[i]).cuda()
            all_tem_pts[i] = torch.stack(all_tem_pts[i]).cuda()

        return all_tem_rgb, all_tem_pts, all_tem_choose



