# blenderproc run poses/create_template_poses.py
import blenderproc
import bpy
import bmesh
import math
import numpy as np
import os


def get_camera_positions(nSubDiv):
    """
    * Construct an icosphere
    * subdived
    """

    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), enter_editmode=True)
    # bpy.ops.export_mesh.ply(filepath='./sphere.ply')
    icos = bpy.context.object
    me = icos.data

    # -- cut away lower part
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] < 0]

    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # -- subdivide and move new vertices out to the surface of the sphere
    #    nSubDiv = 3
    for i in range(nSubDiv):
        bpy.ops.mesh.subdivide()

        bm = bmesh.from_edit_mesh(me)
        for v in bm.verts:
            l = math.sqrt(v.co[0] ** 2 + v.co[1] ** 2 + v.co[2] ** 2)
            v.co[0] /= l
            v.co[1] /= l
            v.co[2] /= l
        bmesh.update_edit_mesh(me)

    # -- cut away zero elevation
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] <= 0]
    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # convert vertex positions to az,el
    positions = []
    angles = []
    bm = bmesh.from_edit_mesh(me)
    for v in bm.verts:
        x = v.co[0]
        y = v.co[1]
        z = v.co[2]
        az = math.atan2(x, y)  # *180./math.pi
        el = math.atan2(z, math.sqrt(x**2 + y**2))  # *180./math.pi
        # positions.append((az,el))
        angles.append((el, az))
        positions.append((x, y, z))

    bpy.ops.object.editmode_toggle()

    # sort positions, first by az and el
    data = zip(angles, positions)
    positions = sorted(data)
    positions = [y for x, y in positions]
    angles = sorted(angles)
    return angles, positions


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
        print("Warning: camera location is parallel to tmp")
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


def convert_location_to_rotation(locations):
    obj_poses = np.zeros((len(locations), 4, 4))
    for idx, pt in enumerate(locations):
        obj_poses[idx] = look_at(pt, np.array([0, 0, 0]))
    return obj_poses


def inverse_transform(poses):
    new_poses = np.zeros_like(poses)
    for idx_pose in range(len(poses)):
        rot = poses[idx_pose, :3, :3]
        t = poses[idx_pose, :3, 3]
        rot = np.transpose(rot)
        t = -np.matmul(rot, t)
        new_poses[idx_pose][3][3] = 1
        new_poses[idx_pose][:3, :3] = rot
        new_poses[idx_pose][:3, 3] = t
    return new_poses


save_dir = "utils/poses/predefined_poses"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for level in [0, 1, 2]:
    position_icosphere = np.asarray(get_camera_positions(level)[1])
    cam_poses = convert_location_to_rotation(position_icosphere)
    cam_poses[:, :3, 3] *= 1000.
    np.save(f"{save_dir}/cam_poses_level{level}.npy", cam_poses)
    obj_poses = inverse_transform(cam_poses)
    np.save(f"{save_dir}/obj_poses_level{level}.npy", obj_poses)

print("Output saved to: " + save_dir)
