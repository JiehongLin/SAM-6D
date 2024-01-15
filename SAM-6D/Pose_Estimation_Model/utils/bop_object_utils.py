
'''
Modified from https://github.com/rasmushaugaard/surfemb/blob/master/surfemb/data/obj.py
'''

import os
import glob
import json
import numpy as np
import trimesh
from tqdm import tqdm

from data_utils import (
    load_im,
)

class Obj:
    def __init__(
            self, obj_id,
            mesh: trimesh.Trimesh,
            model_points,
            diameter: float,
            symmetry_flag: int,
            template_path: str,
            n_template_view: int,
    ):
        self.obj_id = obj_id
        self.mesh = mesh
        self.model_points = model_points
        self.diameter = diameter
        self.symmetry_flag = symmetry_flag
        self._get_template(template_path, n_template_view)

    def get_item(self, return_color=False, sample_num=2048):
        if return_color:
            model_points, _, model_colors = trimesh.sample.sample_surface(self.mesh, sample_num, sample_color=True)
            model_points = model_points.astype(np.float32) / 1000.0
            return (model_points, model_colors, self.symmetry_flag)
        else:
            return (self.model_points, self.symmetry_flag)

    def _get_template(self, path, nView):
        if nView > 0:
            total_nView = len(glob.glob(os.path.join(path, 'rgb_*.png')))

            self.template = []
            self.template_mask = []
            self.template_pts = []

            for v in range(nView):
                i = int(total_nView / nView * v)
                rgb_path = os.path.join(path, 'rgb_'+str(i)+'.png')
                xyz_path = os.path.join(path, 'xyz_'+str(i)+'.npy')
                mask_path = os.path.join(path, 'mask_'+str(i)+'.png')

                rgb = load_im(rgb_path).astype(np.uint8)
                xyz = np.load(xyz_path).astype(np.float32) / 1000.0 
                mask = load_im(mask_path).astype(np.uint8) == 255

                self.template.append(rgb)
                self.template_mask.append(mask)
                self.template_pts.append(xyz)
        else:
            self.template = None
            self.template_choose = None
            self.template_pts = None

    def get_template(self, view_idx):
        return self.template[view_idx], self.template_mask[view_idx], self.template_pts[view_idx]


def load_obj(
        model_path, obj_id: int, sample_num: int,
        template_path: str,
        n_template_view: int,
):
    models_info = json.load(open(os.path.join(model_path, 'models_info.json')))
    mesh = trimesh.load_mesh(os.path.join(model_path, f'obj_{obj_id:06d}.ply'))
    model_points = mesh.sample(sample_num).astype(np.float32) / 1000.0
    diameter = models_info[str(obj_id)]['diameter'] / 1000.0
    if 'symmetries_continuous' in models_info[str(obj_id)]:
        symmetry_flag = 1
    elif 'symmetries_discrete' in models_info[str(obj_id)]:
        symmetry_flag = 1
    else:
        symmetry_flag = 0
    return Obj(
        obj_id, mesh, model_points, diameter, symmetry_flag,
        template_path, n_template_view
    )


def load_objs(
        model_path='models',
        template_path='render_imgs',
        sample_num=512,
        n_template_view=0,
        show_progressbar=True
):
    objs = []
    obj_ids = sorted([int(p.split('/')[-1][4:10]) for p in glob.glob(os.path.join(model_path, '*.ply'))])

    if n_template_view>0:
        template_paths = sorted(glob.glob(os.path.join(template_path, '*')))
        assert len(template_paths) == len(obj_ids), '{} template_paths, {} obj_ids'.format(len(template_paths), len(obj_ids))
    else:
        template_paths = [None for _ in range(len(obj_ids))]

    cnt = 0
    for obj_id in tqdm(obj_ids, 'loading objects') if show_progressbar else obj_ids:
        objs.append(
            load_obj(model_path, obj_id, sample_num,
                    template_paths[cnt], n_template_view)
        )
        cnt+=1
    return objs, obj_ids

