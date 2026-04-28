import argparse
import json
import os
from typing import List

from yolo_train.label import convert_train_pbr_2_yolo, generate_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a BOP train_pbr dataset to YOLO detection labels."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Dataset root containing models/ and train_pbr/, e.g. /path/to/dji_action4.",
    )
    parser.add_argument(
        "--train_pbr_path",
        default=None,
        help="Optional explicit train_pbr path. Defaults to <dataset_path>/train_pbr.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional YOLO output path. Defaults to <dataset_path>/yolo11/train_obj_s.",
    )
    parser.add_argument(
        "--obj_ids",
        default=None,
        help="Optional comma-separated BOP object ids. Defaults to all ids in models/models_info.json.",
    )
    return parser.parse_args()


def parse_obj_ids(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def load_obj_ids(dataset_path: str, obj_ids_arg: str = None) -> List[str]:
    if obj_ids_arg:
        return parse_obj_ids(obj_ids_arg)

    models_info_path = os.path.join(dataset_path, "models", "models_info.json")
    if not os.path.exists(models_info_path):
        raise FileNotFoundError(f"models_info.json not found: {models_info_path}")

    with open(models_info_path, "r", encoding="utf-8") as f:
        models_info = json.load(f)
    return [str(key) for key in sorted(models_info.keys(), key=lambda x: int(x))]


def main() -> None:
    args = parse_args()
    dataset_path = os.path.abspath(args.dataset_path)
    train_pbr_path = os.path.abspath(args.train_pbr_path or os.path.join(dataset_path, "train_pbr"))
    output_path = os.path.abspath(args.output_path or os.path.join(dataset_path, "yolo11", "train_obj_s"))

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset root not found: {dataset_path}")
    if not os.path.isdir(train_pbr_path):
        raise FileNotFoundError(f"train_pbr path not found: {train_pbr_path}")

    obj_id_list = load_obj_ids(dataset_path, args.obj_ids)
    print(f"[INFO] Dataset root: {dataset_path}")
    print(f"[INFO] train_pbr: {train_pbr_path}")
    print(f"[INFO] YOLO output: {output_path}")
    print(f"[INFO] Object ids: {obj_id_list}")

    convert_train_pbr_2_yolo(train_pbr_path, output_path, obj_id_list)
    yaml_path = generate_yaml(output_path, obj_id_list)
    print(f"[INFO] Dataset preparation complete: {yaml_path}")


if __name__ == "__main__":
    main()
