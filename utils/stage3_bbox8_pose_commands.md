# Stage 3: instance-level bbox8 heatmap network

## 1. Complete phase-2 bbox8 label generation

```bash
python utils/gen_bbox8_labels_from_bop.py \
  --dataset_root /abs/path/to/HCCEPose/dji_action4 \
  --obj_id 1 \
  --folder_name train_pbr \
  --visib_thresh 0.1 \
  --val_ratio 0.1
```

Expected output:

- `/abs/path/to/HCCEPose/dji_action4/bbox8_labels_obj_000001/annotations.jsonl`
- `/abs/path/to/HCCEPose/dji_action4/bbox8_labels_obj_000001/train.txt`
- `/abs/path/to/HCCEPose/dji_action4/bbox8_labels_obj_000001/val.txt`
- `/abs/path/to/HCCEPose/dji_action4/bbox8_labels_obj_000001/object_bbox_3d.json`

## 2. Train the stage-3 network

```bash
python -m bbox8_pose.train \
  --labels_root /abs/path/to/HCCEPose/dji_action4/bbox8_labels_obj_000001 \
  --output_dir /abs/path/to/outputs/bbox8_pose_resnet \
  --epochs 30 \
  --batch_size 16 \
  --image_width 256 \
  --image_height 256 \
  --heatmap_width 64 \
  --heatmap_height 64 \
  --lr 1e-3
```

## 3. Run inference on a real image or a frame folder

```bash
python -m bbox8_pose.infer \
  --checkpoint /abs/path/to/outputs/bbox8_pose_resnet/best.pt \
  --input /abs/path/to/real_frames \
  --output_dir /abs/path/to/outputs/bbox8_pose_infer
```

## 4. Optional: recover pose with solvePnP

If you also provide camera intrinsics and the generated 3D bbox metadata:

```bash
python -m bbox8_pose.infer \
  --checkpoint /abs/path/to/outputs/bbox8_pose_resnet/best.pt \
  --input /abs/path/to/real_frames \
  --output_dir /abs/path/to/outputs/bbox8_pose_infer \
  --camera_json /abs/path/to/camera.json \
  --bbox3d_json /abs/path/to/HCCEPose/dji_action4/bbox8_labels_obj_000001/object_bbox_3d.json
```

Then each output json can additionally include:

- `cam_R_m2c`
- `cam_t_m2c`
