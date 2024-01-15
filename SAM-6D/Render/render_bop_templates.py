import blenderproc as bproc

import os
import argparse
import json
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', help="The name of bop datasets")
args = parser.parse_args()

# set relative path of Data folder
render_dir = os.path.dirname(os.path.abspath(__file__))
bop_path = os.path.join(render_dir, '../Data/BOP')
output_dir = os.path.join(render_dir, '../Data/BOP-Templates')
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

bproc.init()

if args.dataset_name == 'tless':
    model_folder = 'models_cad'
else:
    model_folder = 'models'

model_path = os.path.join(bop_path, args.dataset_name, model_folder)
models_info = json.load(open(os.path.join(model_path, 'models_info.json')))
for obj_id in models_info.keys():
    diameter = models_info[obj_id]['diameter']
    scale = 1 / diameter
    obj_fpath = os.path.join(model_path, f'obj_{int(obj_id):06d}.ply')

    cam_poses = np.load(cnos_cam_fpath)
    for idx, cam_pose in enumerate(cam_poses[:]):
        
        bproc.clean_up()

        # load object
        obj = bproc.loader.load_obj(obj_fpath)[0]
        obj.set_scale([scale, scale, scale])
        obj.set_cp("category_id", obj_id)

        if args.dataset_name == 'tless':
            color = [0.4, 0.4, 0.4, 0.]
            material = bproc.material.create('obj')
            material.set_principled_shader_value('Base Color', color)
            obj.set_material(0, material)
        
        # convert cnos camera poses to blender camera poses
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
        cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
        bproc.camera.add_camera_pose(cam_pose)

        # set light
        light_energy = 1000
        light_scale = 2.5
        light1 = bproc.types.Light()
        light1.set_type("POINT")
        light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
        light1.set_energy(light_energy)

        bproc.renderer.set_max_amount_of_samples(1)
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())
        
        # check save folder
        save_fpath = os.path.join(output_dir, args.dataset_name, f'obj_{int(obj_id):06d}')
        if not os.path.exists(save_fpath):
            os.makedirs(save_fpath)

        # save rgb image
        color_bgr_0 = data["colors"][0]
        color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

        # save mask
        mask_0 = data["nocs"][0][..., -1]
        cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
        
        # save nocs
        xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
        np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))
