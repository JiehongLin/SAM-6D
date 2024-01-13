import numpy as np
# import open3d as o3d
import os
from utils.poses.pose_utils import (
    get_obj_poses_from_template_level,
    get_root_project,
    NearestTemplateFinder,
)
import os.path as osp
# from utils.vis_3d_utils import convert_numpy_to_open3d, draw_camera

if __name__ == "__main__":
    for template_level in range(2):
        templates_poses_level0 = get_obj_poses_from_template_level(
            template_level, "all", return_cam=True
        )
        finder = NearestTemplateFinder(
            level_templates=2,
            pose_distribution="all",
            return_inplane=True,
        )
        obj_poses_level = get_obj_poses_from_template_level(template_level, "all", return_cam=False)
        idx_templates, inplanes = finder.search_nearest_template(obj_poses_level)
        print(len(obj_poses_level), len(idx_templates))
        root_repo = get_root_project()
        save_path = os.path.join(root_repo, f"utils/poses/predefined_poses/idx_all_level{template_level}_in_level2.npy")
        np.save(save_path, idx_templates)

    # level 2 in level 2 is just itself
    obj_poses_level = get_obj_poses_from_template_level(2, "all", return_cam=False)
    print(len(obj_poses_level))
    save_path = os.path.join(root_repo, "utils/poses/predefined_poses/idx_all_level2_in_level2.npy")
    np.save(save_path, np.arange(len(obj_poses_level)))