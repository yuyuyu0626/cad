import argparse
import json
import os
import random
from itertools import product

import cv2
import numpy as np
import trimesh


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_vertices_from_mesh(path: str) -> np.ndarray:
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f'No mesh geometry found in: {path}')
        mesh = trimesh.util.concatenate(geoms)
    return np.asarray(mesh.vertices, dtype=np.float64)


def bbox8_from_vertices(vertices: np.ndarray) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float64)
    return corners


def project_points(points_obj: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    pts_cam = (R @ points_obj.T + t.reshape(3, 1)).T
    z = pts_cam[:, 2]
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z, pts_cam


def find_rgb_path(scene_dir: str, image_id: str) -> str:
    for folder in ['rgb', 'gray']:
        folder_path = os.path.join(scene_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            p = os.path.join(folder_path, image_id + ext)
            if os.path.exists(p):
                return os.path.abspath(p)
    raise FileNotFoundError(f'RGB image not found for {scene_dir=} {image_id=}')


def find_mask_visib_path(scene_dir: str, image_id: str, ann_idx: int) -> str | None:
    folder_path = os.path.join(scene_dir, 'mask_visib')
    if not os.path.isdir(folder_path):
        return None
    suffix = f'{image_id}_{ann_idx:06d}'
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        p = os.path.join(folder_path, suffix + ext)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def load_image_wh(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Failed to read image: {path}')
    h, w = img.shape[:2]
    return w, h


def main():
    parser = argparse.ArgumentParser(description='Generate 8-corner 2D labels from a BOP-format train_pbr render dataset.')
    parser.add_argument('--dataset_root', required=True, help='Absolute path to dataset root, containing models/ and train_pbr/')
    parser.add_argument('--obj_id', type=int, default=1, help='Target object id')
    parser.add_argument('--folder_name', default='train_pbr', help='BOP render folder name')
    parser.add_argument('--visib_thresh', type=float, default=0.0, help='Only keep instances with visib_fract >= threshold')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--output_dir', default=None, help='Output dir. Default: <dataset_root>/bbox8_labels_obj_xxxxxx')
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    models_dir = os.path.join(dataset_root, 'models')
    train_dir = os.path.join(dataset_root, args.folder_name)
    model_path = os.path.join(models_dir, f'obj_{args.obj_id:06d}.ply')
    models_info_path = os.path.join(models_dir, 'models_info.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(models_info_path):
        raise FileNotFoundError(models_info_path)
    if not os.path.isdir(train_dir):
        raise NotADirectoryError(train_dir)

    output_dir = os.path.abspath(args.output_dir or os.path.join(dataset_root, f'bbox8_labels_obj_{args.obj_id:06d}'))
    os.makedirs(output_dir, exist_ok=True)

    vertices = load_vertices_from_mesh(model_path)
    corners_3d = bbox8_from_vertices(vertices)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    meta = {
        'dataset_root': dataset_root,
        'folder_name': args.folder_name,
        'obj_id': args.obj_id,
        'model_path': os.path.abspath(model_path),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_extent': (bbox_max - bbox_min).tolist(),
        'corners_3d_object': corners_3d.tolist(),
    }
    save_json(os.path.join(output_dir, 'object_bbox_3d.json'), meta)

    scene_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    records = []

    for scene_name in scene_names:
        scene_dir = os.path.join(train_dir, scene_name)
        scene_camera_path = os.path.join(scene_dir, 'scene_camera.json')
        scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
        scene_gt_info_path = os.path.join(scene_dir, 'scene_gt_info.json')
        if not (os.path.exists(scene_camera_path) and os.path.exists(scene_gt_path)):
            continue

        scene_camera = load_json(scene_camera_path)
        scene_gt = load_json(scene_gt_path)
        scene_gt_info = load_json(scene_gt_info_path) if os.path.exists(scene_gt_info_path) else {}

        for image_key, cam_item in scene_camera.items():
            image_id = str(image_key).rjust(6, '0')
            rgb_path = find_rgb_path(scene_dir, image_id)
            img_w, img_h = load_image_wh(rgb_path)
            anns = scene_gt.get(str(image_key), scene_gt.get(image_id, []))
            infos = scene_gt_info.get(str(image_key), scene_gt_info.get(image_id, [{} for _ in range(len(anns))]))

            for ann_idx, ann in enumerate(anns):
                if int(ann['obj_id']) != args.obj_id:
                    continue
                info = infos[ann_idx] if ann_idx < len(infos) else {}
                visib_fract = float(info.get('visib_fract', 1.0))
                if visib_fract < args.visib_thresh:
                    continue

                K = np.asarray(cam_item['cam_K'], dtype=np.float64).reshape(3, 3)
                R = np.asarray(ann['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
                t = np.asarray(ann['cam_t_m2c'], dtype=np.float64).reshape(3)
                corners_2d, z, corners_cam = project_points(corners_3d, R, t, K)

                valid = (z > 1e-6)
                valid &= (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < img_w)
                valid &= (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < img_h)

                rec = {
                    'sample_id': f'{scene_name}/{image_id}/{ann_idx:06d}',
                    'scene_id': scene_name,
                    'image_id': image_id,
                    'ann_idx': ann_idx,
                    'obj_id': args.obj_id,
                    'rgb_path': rgb_path,
                    'mask_visib_path': find_mask_visib_path(scene_dir, image_id, ann_idx),
                    'image_size': [img_w, img_h],
                    'cam_K': K.reshape(-1).tolist(),
                    'cam_R_m2c': R.reshape(-1).tolist(),
                    'cam_t_m2c': t.tolist(),
                    'bbox_visib': info.get('bbox_visib', None),
                    'bbox_obj': info.get('bbox_obj', None),
                    'visib_fract': visib_fract,
                    'px_count_visib': info.get('px_count_visib', None),
                    'px_count_all': info.get('px_count_all', None),
                    'corners_3d_object': corners_3d.tolist(),
                    'corners_3d_camera': corners_cam.tolist(),
                    'corners_2d': corners_2d.tolist(),
                    'corner_valid_mask': valid.astype(int).tolist(),
                }
                records.append(rec)

    ann_path = os.path.join(output_dir, 'annotations.jsonl')
    with open(ann_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    ids = [rec['sample_id'] for rec in records]
    rnd = random.Random(args.seed)
    rnd.shuffle(ids)
    n_val = int(round(len(ids) * args.val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = [x for x in ids if x not in val_ids]
    val_ids = [x for x in ids if x in val_ids]

    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for x in train_ids:
            f.write(x + '\n')
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for x in val_ids:
            f.write(x + '\n')

    print('[OK] annotations:', ann_path)
    print('[OK] total samples:', len(records))
    print('[OK] train samples:', len(train_ids))
    print('[OK] val samples:', len(val_ids))
    print('[OK] bbox meta:', os.path.join(output_dir, 'object_bbox_3d.json'))


if __name__ == '__main__':
    main()
