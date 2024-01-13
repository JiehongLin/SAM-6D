import numpy as np
import pyrender
import trimesh
import os
from PIL import Image
import numpy as np
import os.path as osp
from tqdm import tqdm
import argparse
from utils.trimesh_utils import as_mesh
from utils.trimesh_utils import get_obj_diameter
os.environ["DISPLAY"] = ":1"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def render(
    mesh,
    output_dir,
    obj_poses,
    img_size,
    intrinsic,
    light_itensity=0.6,
    is_tless=False,
):
    # camera pose is fixed as np.eye(4)
    cam_pose = np.eye(4)
    # convert openCV camera
    cam_pose[1, 1] = -1
    cam_pose[2, 2] = -1
    # create scene config
    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])  # np.array([1.0, 1.0, 1.0, 1.0])
    if light_itensity != 0.6:
        ambient_light = np.array([1.0, 1.0, 1.0, 1.0])
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]), ambient_light=ambient_light
    )
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=light_itensity,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=cam_pose)

    # create camera and render engine
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000
    )
    scene.add(camera, pose=cam_pose)
    render_engine = pyrender.OffscreenRenderer(img_size[1], img_size[0])
    cad_node = scene.add(mesh, pose=np.eye(4), name="cad")

    for idx_frame in range(obj_poses.shape[0]):
        scene.set_pose(cad_node, obj_poses[idx_frame])
        rgb, depth = render_engine.render(scene, pyrender.constants.RenderFlags.RGBA)
        rgb = Image.fromarray(np.uint8(rgb))
        rgb.save(osp.join(output_dir, f"{idx_frame:06d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("obj_pose", nargs="?", help="Path to the model file")
    parser.add_argument(
        "output_dir", nargs="?", help="Path to where the final files will be saved"
    )
    parser.add_argument("gpus_devices", nargs="?", help="GPU devices")
    parser.add_argument("disable_output", nargs="?", help="Disable output of blender")
    parser.add_argument("light_itensity", nargs="?", type=float, default=0.6, help="Light itensity")
    parser.add_argument("radius", nargs="?", type=float, default=1, help="Distance from camera to object")
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_devices
    poses = np.load(args.obj_pose)
    # we can increase high energy for lightning but it's simpler to change just scale of the object to meter
    # poses[:, :3, :3] = poses[:, :3, :3] / 1000.0
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
    if args.radius != 1:
        poses[:, :3, 3] = poses[:, :3, 3] * args.radius
    if "tless" in args.output_dir:
        intrinsic = np.asarray(
            [1075.65091572, 0.0, 360, 0.0, 1073.90347929, 270, 0.0, 0.0, 1.0]
        ).reshape(3, 3)
        img_size = [540, 720]
        is_tless = True
    else:
        intrinsic = np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
        )
        img_size = [480, 640]
        is_tless = False

    # load mesh to meter
    mesh = trimesh.load_mesh(args.cad_path)
    diameter = get_obj_diameter(mesh)
    if diameter > 100: # object is in mm
        mesh.apply_scale(0.001)
    if is_tless:
        # setting uniform colors for mesh
        color = 0.4
        mesh.visual.face_colors = np.ones((len(mesh.faces), 3)) * color
        mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 3)) * color
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    else:
        mesh = pyrender.Mesh.from_trimesh(as_mesh(mesh))
    os.makedirs(args.output_dir, exist_ok=True)
    render(
        output_dir=args.output_dir,
        mesh=mesh,
        obj_poses=poses,
        intrinsic=intrinsic,
        img_size=(480, 640),
        light_itensity=args.light_itensity,
    )
