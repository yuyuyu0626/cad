# 1) OBJ -> BOP PLY
python /home/zhanght2504/zhanght2504/runspace_yyx5/utils/convert_obj_to_bop_ply_v2.py \
  --input_mesh /home/zhanght2504/zhanght2504/runspace_yyx5/head_left_rgb_raw.mp4/Osmo_Action_4.stl \
  --output_ply /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/models/obj_000001.ply \
  --axis_order xzy

# 2) In HCCEPose repo, generate models_info.json
python /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/s1_p3_obj_infos.py

# 3) Download cc0textures-512 (or full cc0textures), then run HCCEPose PBR rendering
rm -rf /data/zht_data/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/train_pbr
rm -rf /data/zht_data/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/custom_keypoints

cd /data/zht_data/zhanght2504/runspace_yyx5/HCCEPose

./s2_p1_gen_pbr_data.sh \
  0 \
  3 \
  /data/zht_data/zhanght2504/runspace_yyx5/cc0textures-512 \
  /data/zht_data/zhanght2504/runspace_yyx5/HCCEPose/dji_action4 \
  /data/zht_data/zhanght2504/runspace_yyx5/HCCEPose/s2_p1_gen_pbr_data.py \
  --num-scenes 1 \
  --bop-num-worker 1 \
  --num-objects 1

# 4) Generate 8-corner labels from BOP render result
python /home/zhanght2504/zhanght2504/runspace_yyx5/utils/gen_bbox8_labels_from_bop.py \
  --dataset_root /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4 \
  --obj_id 1 \
  --folder_name train_pbr \
  --visib_thresh 0.1 \
  --val_ratio 0.1



# train
cd /home/zhanght2504/zhanght2504/runspace_yyx5

python -m bbox8_pose.train \
  --labels_root /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/bbox8_labels_obj_000001 \
  --output_dir /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_obj_000001_resnet18 \
  --epochs 100 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 1e-3 \
  --image_width 512 \
  --image_height 512 \
  --heatmap_width 128 \
  --heatmap_height 128 \
  --sigma 2.5 \
  --device cuda \
  --backbone resnet18 \
  --vis_every 1 \
  --vis_num_samples 4


# 训练后验证
python -m bbox8_pose.infer \
  --checkpoint /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_obj_000001_resnet18/best.pt \
  --input /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/train_pbr/000000/rgb \
  --output_dir /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_val_infer \
  --camera_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/camera.json \
  --bbox3d_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/bbox8_labels_obj_000001/object_bbox_3d.json \
  --device cuda


# 真实视频测试
抽帧
mkdir -p /home/zhanght2504/zhanght2504/runspace_yyx5/test_video_frames

ffmpeg -i /home/zhanght2504/zhanght2504/runspace_yyx5/head_left_rgb_raw.mp4 \
  -qscale:v 2 \
  /home/zhanght2504/zhanght2504/runspace_yyx5/test_video_frames/%06d.jpg


推理
python -m bbox8_pose.infer \
  --checkpoint /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_obj_000001_resnet18/best.pt \
  --input /home/zhanght2504/zhanght2504/runspace_yyx5/test_video_frames \
  --output_dir /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_video_test \
  --camera_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/camera.json \
  --bbox3d_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/bbox8_labels_obj_000001/object_bbox_3d.json \
  --device cuda
