# 1) OBJ -> BOP PLY
python /home/zhanght2504/zhanght2504/runspace_yyx5/utils/convert_obj_to_bop_ply_v2.py \
  --input_mesh /home/zhanght2504/zhanght2504/runspace_yyx4.5/head_left_rgb_raw.mp4/dji_action4_black.obj \
  --output_ply /home/zhanght2504/zhanght2504/runspace_yyx4.5/dji_action4/models/obj_000001.ply \
  --axis_order xyz

python /home/zhanght2504/zhanght2504/runspace_yyx5/utils/convert_obj_to_bop_ply_v2.py \
  --input_mesh /home/zhanght2504/zhanght2504/runspace_yyx4.5/head_left_rgb_raw.mp4/dji_actions4_pbr/dj_actions4.obj \
  --output_ply /home/zhanght2504/zhanght2504/runspace_yyx4.5/dji_actions_pbr/models/obj_000001.ply \
  --axis_order xyz


# 2) In HCCEPose repo, generate models_info.json
python /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/s1_p3_obj_infos.py

# 3) Download cc0textures-512 (or full cc0textures), then run HCCEPose PBR rendering
rm -rf /home/zhanght2504/zhanght2504/runspace_yyx4.5/dji_action4/train_pbr
rm -rf /home/zhanght2504/zhanght2504/runspace_yyx4.5/dji_action4/custom_keypoints

cd /data/zht_data/zhanght2504/runspace_yyx5/HCCEPose

./s2_p1_gen_pbr_data.sh \
  5 \
  150 \
  /data/zht_data/zhanght2504/runspace_yyx5/cc0textures-512 \
  /home/zhanght2504/zhanght2504/runspace_yyx4.5/dji_action4 \
  /data/zht_data/zhanght2504/runspace_yyx5/HCCEPose/s2_p1_gen_pbr_data.py \
  --num-scenes 1 \
  --bop-num-worker 1 \
  --num-objects 1 \
  --render-model-obj /home/zhanght2504/zhanght2504/runspace_yyx4.5/head_left_rgb_raw.mp4/dji_action4_black.obj

# 4) Generate 8-corner labels from BOP render result
python /home/zhanght2504/zhanght2504/runspace_yyx5/utils/gen_bbox8_labels_from_bop.py \
  --dataset_root /home/zhanght2504/zhanght2504/runspace_yyx4.5/dji_action4 \
  --obj_id 1 \
  --folder_name train_pbr \
  --visib_thresh 0.1 \
  --val_ratio 0.1



# train
cd /data/zht_data/zhanght2504/runspace_yyx5

python -m bbox8_pose.train \
  --labels_root /home/zhanght2504/zhanght2504/runspace_yyx4.5/dji_action4/bbox8_labels_obj_000001 \
  --output_dir /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_obj_000001_boxdreamer_lite \
  --epochs 50 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 3e-4 \
  --image_width 384 \
  --image_height 384 \
  --heatmap_width 96 \
  --heatmap_height 96 \
  --sigma 2.5 \
  --device cuda \
  --backbone resnet18 \
  --decoder boxdreamer_lite \
  --decoder_dim 192 \
  --decoder_depth 3 \
  --decoder_heads 8 \
  --vis_every 1 \
  --vis_num_samples 4



# 训练后验证
CUDA_VISIBLE_DEVICES=1 python -m bbox8_pose.infer \
  --checkpoint /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_obj_000001_boxdreamer_lite/best_metric.pt \
  --input /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/train_pbr/000000/rgb \
  --output_dir /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_boxdreamer_lite_infer \
  --camera_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/camera.json \
  --bbox3d_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/bbox8_labels_obj_000001/object_bbox_3d.json \
  --device cuda

生成demo视频
ffmpeg -framerate 6 -i /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_boxdreamer_lite_infer/%06d_vis.jpg \
  -t 20 \
  -c:v libx264 -pix_fmt yuv420p \
  /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/demo_videos/demo_infer_vis.mp4





# 真实视频测试
抽帧
mkdir -p /home/zhanght2504/zhanght2504/runspace_yyx5/test_video_frames

ffmpeg -i /home/zhanght2504/zhanght2504/runspace_yyx5/head_left_rgb_raw.mp4/head_left_rgb_raw.mp4 \
  -qscale:v 2 \
  /home/zhanght2504/zhanght2504/runspace_yyx5/test_video_frames/%06d.jpg


推理
CUDA_VISIBLE_DEVICES=1 python -m bbox8_pose.infer \
  --checkpoint /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_obj_000001_boxdreamer_lite/best_metric.pt \
  --input /home/zhanght2504/zhanght2504/runspace_yyx5/test_video_frames \
  --output_dir /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_dji_action4_test \
  --camera_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/camera.json \
  --bbox3d_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/bbox8_labels_obj_000001/object_bbox_3d.json \
  --device cuda


生成demo视频
ffmpeg -framerate 2997/100 -i /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_dji_action4_test/%06d_vis.jpg \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p \
  /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_dji_action4_test/demo_real_vis.mp4



# 测试指标
python /home/zhanght2504/zhanght2504/runspace_yyx5/utils/analyze_bbox8_infer_results.py \
  --infer_dir /home/zhanght2504/zhanght2504/runspace_yyx5/outputs/bbox8_pose_dji_action4_test \
  --camera_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/camera.json \
  --bbox3d_json /home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/dji_action4/bbox8_labels_obj_000001/object_bbox_3d.json

生成文件信息：
每一帧的详细指标
temporal_metrics.csv
相邻帧之间的时序变化指标
summary.json
汇总统计
report.txt
人能直接看的一页结论摘要

关键指标：
in_frame_ratio 的平均值
越接近 1.0 越好
bbox_area_ratio
不要太小，也不要乱跳太大
center_disp
相邻帧不要跳得太夸张
rot_delta_deg / trans_delta
如果视频本身运动平滑，这些也应该比较平滑
frame_metrics.csv




python coderepo_other/recongs/render_freeview.py \
  --point_cloud_dir /home/zhanght2504/zhanght2504/runspace_yyx3/muiti_view_render/recongs/point_cloud \
  --camera_root /data/zht_data/researchdata/N3DV_processed/cook_spinach \
  --camera_subdir_pattern "frame{frame:06d}/sparse/0" \
  --output_dir /home/zhanght2504/zhanght2504/runspace_yyx3/muiti_view_render/recongs/output \
  --frame_start 2 \
  --frame_end 2 \
  --camera_names cam00.png,cam05.png,cam06.png,cam16.png,cam17.png
