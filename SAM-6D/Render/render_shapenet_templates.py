import blenderproc as bproc

import os
import cv2
import numpy as np
import trimesh

# set relative path of Data folder
render_dir = os.path.dirname(os.path.abspath(__file__))
shapenet_path = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/shapenetcorev2')
shapenet_orig_path = os.path.join(shapenet_path, 'models_orig')
output_dir = os.path.join(render_dir, '../Data/MegaPose-Training-Data/MegaPose-ShapeNetCore/templates')

bproc.init()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1/(2*radius)


for synset_id in os.listdir(shapenet_orig_path):
    synset_fpath = os.path.join(shapenet_orig_path, synset_id)
    if not os.path.isdir(synset_fpath) or '.' in synset_id:
        continue
    print('---------------------------'+str(synset_id)+'-------------------------------------')
    for idx, source_id in enumerate(os.listdir(synset_fpath)):
        print('---------------------------'+str(source_id)+'-------------------------------------')
        save_synset_folder = os.path.join(output_dir, synset_id)
        if not os.path.exists(save_synset_folder):
            os.makedirs(save_synset_folder)

        save_fpath = os.path.join(save_synset_folder, source_id)
        if not os.path.exists(save_fpath):
            os.mkdir(save_fpath)
        else:
            continue

        cad_path = os.path.join(shapenet_orig_path, synset_id, source_id)
        obj_fpath = os.path.join(cad_path, 'models', 'model_normalized.obj')

        if not os.path.exists(obj_fpath):
            continue

        scale = get_norm_info(obj_fpath)

        bproc.clean_up()

        obj = bproc.loader.load_shapenet(shapenet_orig_path, synset_id, source_id, move_object_origin=False)
        obj.set_scale([scale, scale, scale])
        obj.set_cp("category_id", idx)

        # set light
        light1 = bproc.types.Light()
        light1.set_type("POINT")
        light1.set_location([-3, -3, -3])
        light1.set_energy(1000)

        light2 = bproc.types.Light()
        light2.set_type("POINT")
        light2.set_location([3, 3, 3])
        light2.set_energy(1000)

        location = [[-1, -1, -1], [1, 1, 1]]
        # set camera locations around the object
        for loc in location:
            # compute rotation based on vector going from location towards the location of object
            rotation_matrix = bproc.camera.rotation_from_forward_vec(obj.get_location() - loc)
            # add homog cam pose based on location and rotation
            cam2world_matrix = bproc.math.build_transformation_mat(loc, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)

        bproc.renderer.set_max_amount_of_samples(1)
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())

        # save rgb images
        color_bgr_0 = data["colors"][0]
        color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(0)+'.png'), color_bgr_0)
        color_bgr_1 = data["colors"][1]
        color_bgr_1[..., :3] = color_bgr_1[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(1)+'.png'), color_bgr_1)

        # save masks
        mask_0 = data["nocs"][0][..., -1]
        mask_1 = data["nocs"][1][..., -1]
        cv2.imwrite(os.path.join(save_fpath,'mask_'+str(0)+'.png'), mask_0*255)
        cv2.imwrite(os.path.join(save_fpath,'mask_'+str(1)+'.png'), mask_1*255)

        # save nocs
        xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
        xyz_1 = 2*(data["nocs"][1][..., :3] - 0.5)
        # xyz need to rotate 90 degree to match CAD
        rot90 = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]])
        h, w = xyz_0.shape[0], xyz_0.shape[1]

        xyz_0 = ((rot90 @ xyz_0.reshape(-1, 3).T).T).reshape(h, w, 3)
        xyz_1 = ((rot90 @ xyz_1.reshape(-1, 3).T).T).reshape(h, w, 3)
        np.save(os.path.join(save_fpath,'xyz_'+str(0)+'.npy'), xyz_0.astype(np.float16))
        np.save(os.path.join(save_fpath,'xyz_'+str(1)+'.npy'), xyz_1.astype(np.float16))

