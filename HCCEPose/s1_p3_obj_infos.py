# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

import argparse
import glob
import os

import numpy as np
from kasal.utils import load_json2dict, load_ply_model, write_dict2json


def get_models_in_dataset(dataset_path: str):
    """Only collect direct PLY files under <dataset_path>/models."""
    models_dir = os.path.join(dataset_path, "models")
    pattern = os.path.join(models_dir, "obj_*.ply")
    return sorted(glob.glob(pattern))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate models_info.json from direct PLY files under <dataset_path>/models."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.getcwd(),
        help="Dataset root containing a models folder. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional single model file name to process, e.g. obj_000001.ply.",
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_path)
    models_path = get_models_in_dataset(dataset_path)

    if args.model_name is not None:
        target_path = os.path.join(dataset_path, "models", args.model_name)
        models_path = [p for p in models_path if os.path.abspath(p) == os.path.abspath(target_path)]

    models_info = {}
    for model_path in models_path:
        ply_info = load_ply_model(model_path)
        vertices = ply_info["vertices"]
        v_max = np.max(vertices, axis=0)
        v_min = np.min(vertices, axis=0)

        model_info = {
            "diameter": float(ply_info["diameter"]),
            "max_x": float(v_max[0]),
            "max_y": float(v_max[1]),
            "max_z": float(v_max[2]),
            "min_x": float(v_min[0]),
            "min_y": float(v_min[1]),
            "min_z": float(v_min[2]),
            "size_x": float(v_max[0] - v_min[0]),
            "size_y": float(v_max[1] - v_min[1]),
            "size_z": float(v_max[2] - v_min[2]),
        }

        symmetry_type_dict = None
        sym_type_file = os.path.join(
            os.path.dirname(model_path),
            os.path.basename(model_path).split(".")[0] + "_sym_type.json",
        )
        if os.path.exists(sym_type_file):
            symmetry_type_dict = load_json2dict(sym_type_file)

        if symmetry_type_dict is not None:
            current_obj_info = symmetry_type_dict.get("current_obj_info", {})
            if "symmetries_continuous" in current_obj_info:
                model_info["symmetries_continuous"] = current_obj_info["symmetries_continuous"]
            if "symmetries_discrete" in current_obj_info:
                model_info["symmetries_discrete"] = current_obj_info["symmetries_discrete"]

        model_id = str(int(os.path.basename(model_path).split(".")[0].split("obj_")[1]))
        models_info[model_id] = model_info

    if len(models_path) == 0:
        raise FileNotFoundError(
            f"No matching PLY files found under: {os.path.join(dataset_path, 'models')}"
        )

    models_info_path = os.path.join(dataset_path, "models", "models_info.json")
    write_dict2json(models_info_path, models_info)
    print(f"[OK] Wrote: {models_info_path}")
    print(f"[INFO] Processed {len(models_path)} model(s)")
