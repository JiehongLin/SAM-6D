# Define variables
OUTPUT_DIR=/home/mnardon/Desktop/GitHub/ai_prism/SAM-6D/RESULTS/20250311_no_occlusions
CAD_PATH=/home/mnardon/Desktop/GitHub/ai_prism/chairs/ike_chairs/chair_origin_edge_2.ply_mm.ply
RGB_PATH=/data/disk1/share/mnardon/ai_prismDATA/20250210_organized/20250210_moderate_occlusions/zed_cam/rgb/image_rect_color/000010.png
DEPTH_PATH=/data/disk1/share/mnardon/ai_prismDATA/20250210_organized/20250210_moderate_occlusions/zed_cam/depth/depth_registered/000010.npy
CAMERA_PATH=/home/mnardon/Desktop/GitHub/ai_prism/zed_cam.json


CAD_PATH=/home/mnardon/Desktop/GitHub/nextmagINT/Models_Granata/MODIFIED/PLY_after_BLEND/et9_mm.ply
OUTPUT_DIR=/home/mnardon/Desktop/GitHub/nextmagINT/SAM-6D/tmp/RESULTS/
OUTPUT_DIR=

# Render CAD templates
# cd Render
# blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --colorize True --custom-blender-path /data/home/mnardon/blender/blender-4.2.1-linux-x64

# echo "Rendering done"

# # # # Run instance segmentation model
# export SEGMENTOR_MODEL=fastsam

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

# # echo "Segmentation done"

# # Run pose estimation model
# export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

# cd ../Pose_Estimation_Model
# # python run_inference_custom_bypass.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH
# python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

# echo "Pose estimation done"