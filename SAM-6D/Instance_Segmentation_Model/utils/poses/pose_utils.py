import os
import numpy as np
import pathlib
from utils.inout import get_root_project
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import math
from scipy.spatial.distance import cdist
from utils.poses.fps import FPS

def opencv2opengl(cam_matrix_world):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if len(cam_matrix_world.shape) == 2:
        return np.matmul(transform, cam_matrix_world)
    else:
        transform = np.tile(transform, (cam_matrix_world.shape[0], 1, 1))
        return np.matmul(transform, cam_matrix_world)


def combine_R_and_T(R, T, scale_translation=1.0):
    matrix4x4 = np.eye(4)
    matrix4x4[:3, :3] = np.array(R).reshape(3, 3)
    matrix4x4[:3, 3] = np.array(T).reshape(-1) * scale_translation
    return matrix4x4


def read_template_poses(is_opengl_camera, dense=False):
    current_dir = pathlib.Path(__file__).parent.absolute()
    path = f"{current_dir}/predefined_poses/sphere_level"
    if dense:
        path += "3.npy"
    else:
        path += "2.npy"
    template_poses = np.load(path)
    if is_opengl_camera:
        for id_frame in range(len(template_poses)):
            template_poses[id_frame] = opencv2opengl(template_poses[id_frame])
    return template_poses


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))


def perspective(K, obj_pose, pts):
    results = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        R, T = obj_pose[:3, :3], obj_pose[:3, 3]
        rep = np.matmul(K, np.matmul(R, pts[i].reshape(3, 1)) + T.reshape(3, 1))
        results[i, 0] = np.int32(rep[0] / rep[2])  # as matplot flip  x axis
        results[i, 1] = np.int32(rep[1] / rep[2])
    return results


def inverse_transform(trans):
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


def get_obj_poses_from_template_level(
    level, pose_distribution, return_cam=False, return_index=False
):
    root_project = get_root_project()
    if return_cam:
        obj_poses_path = os.path.join(
            root_project, f"utils/poses/predefined_poses/cam_poses_level{level}.npy"
        )
        obj_poses = np.load(obj_poses_path)
    else:
        obj_poses_path = os.path.join(
            root_project, f"utils/poses/predefined_poses/obj_poses_level{level}.npy"
        )
        obj_poses = np.load(obj_poses_path)

    if pose_distribution == "all":
        if return_index:
            index = np.arange(len(obj_poses))
            return index, obj_poses
        else:
            return obj_poses
    elif pose_distribution == "upper":
        cam_poses_path = os.path.join(
            root_project, f"utils/poses/predefined_poses/cam_poses_level{level}.npy"
        )
        cam_poses = np.load(cam_poses_path)
        if return_index:
            index = np.arange(len(obj_poses))[cam_poses[:, 2, 3] >= 0]
            return index, obj_poses[cam_poses[:, 2, 3] >= 0]
        else:
            return obj_poses[cam_poses[:, 2, 3] >= 0]


def load_index_level_in_level2(level, pose_distribution):
    # created from https://github.com/nv-nguyen/DiffusionPose/blob/52e2c55b065c9637dcd284cc77a0bfb3356d218a/src/poses/find_neighbors.py
    root_repo = get_root_project()
    index_path = os.path.join(
        root_repo,
        f"utils/poses/predefined_poses/idx_{pose_distribution}_level{level}_in_level2.npy",
    )
    return np.load(index_path)


def load_mapping_id_templates_to_idx_pose_distribution(level, pose_distribution):
    """
    Return the mapping from the id of the template to the index of the pose distribution
    """
    index_range, _ = get_obj_poses_from_template_level(
        level=level,
        pose_distribution=pose_distribution,
        return_index=True,
    )
    mapping = {}
    for i in range(len(index_range)):
        mapping[int(index_range[i])] = i
    return mapping


def apply_transfrom(transform4x4, matrix4x4):
    # apply transform to a 4x4 matrix
    new_matrix4x4 = transform4x4.dot(matrix4x4)
    return new_matrix4x4


def load_rotation_transform(axis, degrees):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    return torch.from_numpy(transform).float()


def convert_openCV_to_openGL_torch(openCV_poses):
    openCV_to_openGL_transform = (
        torch.tensor(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            device=openCV_poses.device,
            dtype=openCV_poses.dtype,
        )
        .unsqueeze(0)
        .repeat(openCV_poses.shape[0], 1, 1)
    )
    return torch.bmm(openCV_to_openGL_transform, openCV_poses[:, :3, :3])


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True))


def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.0, 0.0, -1.0])
    # print warning when camera location is parallel to tmp
    norm = min(
        np.linalg.norm(cam_location - tmp, axis=-1),
        np.linalg.norm(cam_location + tmp, axis=-1),
    )
    if norm < 1e-3:
        tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def spherical_to_cartesian(azimuth, elevation, radius):
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation)
    return np.stack((x, y, z), axis=-1)


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x[:, :, :, None] * emb[None, None, None, :]  # WxHx3 to WxHxposEnc_size
        emb = emb.reshape(*x.shape[:2], -1)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def extract_inplane_from_pose(pose):
    inp = Rotation.from_matrix(pose).as_euler("zyx", degrees=True)[0]
    return inp


def convert_inplane_to_rotation(inplane):
    R_inp = Rotation.from_euler("z", -inplane, degrees=True).as_matrix()
    return R_inp


def adding_inplane_to_pose(pose, inplane):
    R_inp = convert_inplane_to_rotation(inplane)
    pose = np.dot(R_inp, pose)
    return pose


def compute_inplane(rot_query_openCV, rot_template_openCV):
    delta = rot_template_openCV.dot(rot_query_openCV.T)
    inp = extract_inplane_from_pose(delta)
    # double check to make sure that reconved rotation is correct
    R_inp = convert_inplane_to_rotation(inp)
    recovered_R1 = R_inp.dot(rot_template_openCV)
    err = geodesic_numpy(recovered_R1, rot_query_openCV)
    if err >= 15:
        print("WARINING, error of recovered pose is >=15, err=", err)
    return inp


class NearestTemplateFinder(object):
    def __init__(
        self,
        level_templates,
        pose_distribution,
        return_inplane,
        normalize_query_translation=True,
    ):
        self.level_templates = level_templates
        self.normalize_query_translation = normalize_query_translation
        self.pose_distribution = pose_distribution
        self.return_inplane = return_inplane

        self.avail_index, self.obj_template_poses = get_obj_poses_from_template_level(
            level_templates, pose_distribution, return_cam=False, return_index=True
        )

        # we use the location to find for nearest template on the sphere
        self.obj_template_openGL_poses = opencv2opengl(self.obj_template_poses)

    def search_nearest_template(self, obj_query_pose):
        # convert query pose to OpenGL coordinate
        obj_query_openGL_pose = opencv2opengl(obj_query_pose)
        obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3
        obj_template_openGL_locations = self.obj_template_openGL_poses[:, 2, :3]  # Nx3

        # find the nearest template
        distances = cdist(obj_query_openGL_location, obj_template_openGL_locations)
        best_index_in_pose_distribution = np.argmin(distances, axis=-1)  # M
        if self.return_inplane:
            nearest_poses = self.obj_template_poses[best_index_in_pose_distribution]
            inplanes = np.zeros(len(obj_query_pose))
            for idx in range(len(obj_query_pose)):
                rot_query_openCV = obj_query_pose[idx, :3, :3]
                rot_template_openCV = nearest_poses[idx, :3, :3]
                inplanes[idx] = compute_inplane(rot_query_openCV, rot_template_openCV)
            return self.avail_index[best_index_in_pose_distribution], inplanes
        else:
            return self.avail_index[best_index_in_pose_distribution]
    
    def search_nearest_query(self, obj_query_pose):
        """
        Search nearest query closest to our template_pose
        """
        obj_query_openGL_pose = opencv2opengl(obj_query_pose)
        obj_query_openGL_location = obj_query_openGL_pose[:, 2, :3]  # Mx3
        obj_template_openGL_locations = self.obj_template_openGL_poses[:, 2, :3]  # Nx3
        distances = cdist(obj_template_openGL_locations, obj_query_openGL_location)
        best_index = np.argmin(distances, axis=-1)  # M
        return best_index
    
def farthest_sampling(openCV_poses, num_points):
    # convert query pose to OpenGL coordinate
    openGL_pose = opencv2opengl(openCV_poses)
    openGL_pose_location = openGL_pose[:, 2, :3]  # Mx3
    # apply farthest point sampling
    _, farthest_idx = FPS(openGL_pose_location, num_points).fit()
    return farthest_idx