import numpy as np
import trimesh
import torch


def load_mesh(path, ORIGIN_GEOMETRY="BOUNDS"):
    mesh = as_mesh(trimesh.load(path))
    if ORIGIN_GEOMETRY == "BOUNDS":
        AABB = mesh.bounds
        center = np.mean(AABB, axis=0)
        mesh.vertices -= center
    return mesh


def get_bbox_from_mesh(mesh):
    AABB = mesh.bounds
    OBB = AABB_to_OBB(AABB)
    return OBB


def get_obj_diameter(mesh_path):
    mesh = load_mesh(mesh_path)
    extents = mesh.extents * 2
    return np.linalg.norm(extents)


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        result = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()
            ]
        )
    else:
        result = scene_or_mesh
    return result


def AABB_to_OBB(AABB):
    """
    AABB bbox to oriented bounding box
    """
    minx, miny, minz, maxx, maxy, maxz = np.arange(6)
    corner_index = np.array(
        [
            minx,
            miny,
            minz,
            maxx,
            miny,
            minz,
            maxx,
            maxy,
            minz,
            minx,
            maxy,
            minz,
            minx,
            miny,
            maxz,
            maxx,
            miny,
            maxz,
            maxx,
            maxy,
            maxz,
            minx,
            maxy,
            maxz,
        ]
    ).reshape((-1, 3))

    corners = AABB.reshape(-1)[corner_index]
    return corners

def depth_image_to_pointcloud_translate_torch(depth, scale, K):
    u = torch.arange(0, depth.shape[2])
    v = torch.arange(0, depth.shape[1])

    u, v = torch.meshgrid(u, v, indexing="xy")
    u = u.to(depth.device)
    v = v.to(depth.device)

    # depth metric is mm, depth_scale metric is m
    # K metric is m
    Z = depth * scale / 1000
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    valid = Z > 0

    X = X * valid
    Y = Y * valid
    Z = Z * valid

    # average should run on valid point
    valid_num = torch.count_nonzero(valid, axis=(1, 2)) + 1e-8
    avg_X = torch.sum(X, axis=(1, 2)) / valid_num
    avg_Y = torch.sum(Y, axis=(1, 2)) / valid_num
    avg_Z = torch.sum(Z, axis=(1, 2)) / valid_num

    translate = torch.vstack((avg_X, avg_Y, avg_Z)).permute(1, 0)

    return translate


if __name__ == "__main__":
    mesh_path = (
        "/media/nguyen/Data/dataset/ShapeNet/ShapeNetCore.v2/"
        "03001627/1016f4debe988507589aae130c1f06fb/models/model_normalized.obj"
    )
    mesh = load_mesh(mesh_path)
    bbox = get_bbox_from_mesh(mesh)
    # create a visualization scene with rays, hits, and mesh
    scene = trimesh.Scene([mesh, trimesh.points.PointCloud(bbox)])
    # display the scene
    scene.show()
