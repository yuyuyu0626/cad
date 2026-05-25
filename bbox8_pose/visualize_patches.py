"""
可视化脚本：展示从原始图像 → YOLO检测 → 裁剪patch → 网络输入的全过程
"""
import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .dataset import corners_to_crop_box, dynamic_resize_size
from .model import BBox8PoseNet
from .utils import draw_corners
from .infer import expand_box


def visualize_patches_pipeline(
    image_path: str,
    yolo_model_path: str,
    checkpoint_path: str,
    output_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    crop_margin: float = 0.15,
    yolo_conf: float = 0.25,
    dynamic_input: bool = False,
    dynamic_min_size: int = 128,
    dynamic_size_multiple: int = 32,
) -> None:
    """
    可视化从原始图像到网络输入的完整流程
    
    Args:
        image_path: 输入图像路径
        yolo_model_path: YOLO 模型路径 (可选)
        checkpoint_path: 网络检查点路径
        output_dir: 输出目录
        image_size: 网络输入尺寸
        crop_margin: 裁剪边界间距
        yolo_conf: YOLO 置信度阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取原始图像
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    
    print(f"[INFO] 原始图像尺寸: {orig_w} x {orig_h}")
    
    # 保存原始图像
    cv2.imwrite(
        os.path.join(output_dir, "01_original.jpg"),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    )
    print(f"[SAVE] 01_original.jpg")
    
    # 2. YOLO 检测
    boxes = []
    if yolo_model_path and os.path.exists(yolo_model_path):
        try:
            yolo_model = YOLO(yolo_model_path)
            results = yolo_model.predict(
                image_rgb,
                conf=yolo_conf,
                iou=0.7,
                imgsz=960,
                max_det=20,
                verbose=False,
            )
            if results and results[0].boxes is not None:
                xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
                for box in xyxy:
                    boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                print(f"[INFO] YOLO 检测到 {len(boxes)} 个物体")
        except Exception as e:
            print(f"[WARN] YOLO 检测失败: {e}")
    
    # 绘制 YOLO 检测框
    if boxes:
        image_with_yolo = image_rgb.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(image_with_yolo, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                image_with_yolo, f"Box {i}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        cv2.imwrite(
            os.path.join(output_dir, "02_yolo_detections.jpg"),
            cv2.cvtColor(image_with_yolo, cv2.COLOR_RGB2BGR)
        )
        print(f"[SAVE] 02_yolo_detections.jpg")
    
    # 3. 为每个检测框裁剪 patch
    patches_list = []
    for i, box in enumerate(boxes):
        # 扩展 bbox
        crop_box = expand_box(box, orig_w, orig_h, crop_margin)
        x1, y1, x2, y2 = crop_box
        
        # 裁剪原始 patch
        crop_rgb = image_rgb[y1:y2, x1:x2]
        patch_h, patch_w = crop_rgb.shape[:2]
        
        # 保存原始 patch
        patch_path = os.path.join(output_dir, f"03_patch_{i:02d}_original_{patch_w}x{patch_h}.jpg")
        cv2.imwrite(patch_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        print(f"[SAVE] {os.path.basename(patch_path)}")
        
        # 4. 缩放到网络输入尺寸
        network_size = image_size
        if dynamic_input:
            network_size = dynamic_resize_size(
                crop_rgb.shape[1],
                crop_rgb.shape[0],
                max_size=image_size,
                min_size=dynamic_min_size,
                size_multiple=dynamic_size_multiple,
            )
        resized = cv2.resize(crop_rgb, network_size, interpolation=cv2.INTER_LINEAR)
        resized_path = os.path.join(output_dir, f"04_patch_{i:02d}_resized_{network_size[0]}x{network_size[1]}.jpg")
        cv2.imwrite(resized_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        print(f"[SAVE] {os.path.basename(resized_path)}")
        
        patches_list.append({
            "index": i,
            "bbox": [x1, y1, x2, y2],
            "original_shape": (patch_h, patch_w),
            "resized_shape": network_size,
            "resized_path": os.path.basename(resized_path),
        })
    
    # 5. 创建对比网格 (将所有 patch 排列在一起)
    if patches_list:
        # 创建大网格
        grid_cols = 3
        grid_rows = (len(patches_list) + grid_cols - 1) // grid_cols
        grid_size = max(image_size)
        grid_img = np.ones((grid_rows * (grid_size + 20), grid_cols * (grid_size + 20), 3), dtype=np.uint8) * 255
        
        for idx, patch_info in enumerate(patches_list):
            row = idx // grid_cols
            col = idx % grid_cols
            
            # 读取缩放后的 patch
            resized_path = os.path.join(output_dir, patch_info["resized_path"])
            patch_img = cv2.imread(resized_path)
            if patch_img is not None:
                patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
                patch_h, patch_w = patch_img.shape[:2]
                scale = min(grid_size / float(patch_w), grid_size / float(patch_h))
                show_w = max(1, int(round(patch_w * scale)))
                show_h = max(1, int(round(patch_h * scale)))
                patch_img = cv2.resize(patch_img, (show_w, show_h), interpolation=cv2.INTER_LINEAR)
                # 放到网格中
                y_start = row * (grid_size + 20) + 10
                x_start = col * (grid_size + 20) + 10
                grid_img[y_start:y_start+show_h, x_start:x_start+show_w] = patch_img
                
                # 添加标签
                cv2.putText(
                    grid_img, f"Patch {idx}", (x_start, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
                )
        
        grid_path = os.path.join(output_dir, "05_all_patches_grid.jpg")
        cv2.imwrite(grid_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        print(f"[SAVE] 05_all_patches_grid.jpg")
    
    # 保存信息文件
    info = {
        "image_path": image_path,
        "original_size": [orig_w, orig_h],
        "num_detections": len(boxes),
        "patches": patches_list,
        "network_input_size": list(image_size),
        "dynamic_input": dynamic_input,
    }
    info_path = os.path.join(output_dir, "pipeline_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[SAVE] pipeline_info.json")
    
    print("\n[DONE] 可视化完成!")
    print(f"[INFO] 输出目录: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize image patch pipeline for corner detection")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--yolo_model", default=None, help="Optional YOLO model path")
    parser.add_argument("--checkpoint", default=None, help="Optional network checkpoint path")
    parser.add_argument("--output_dir", required=True, help="Output directory for visualizations")
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--crop_margin", type=float, default=0.15)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--dynamic_input", action="store_true")
    parser.add_argument("--dynamic_min_size", type=int, default=128)
    parser.add_argument("--dynamic_size_multiple", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_patches_pipeline(
        image_path=args.image,
        yolo_model_path=args.yolo_model,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        image_size=(args.image_width, args.image_height),
        crop_margin=args.crop_margin,
        yolo_conf=args.yolo_conf,
        dynamic_input=args.dynamic_input,
        dynamic_min_size=args.dynamic_min_size,
        dynamic_size_multiple=args.dynamic_size_multiple,
    )
