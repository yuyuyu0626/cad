#!/bin/bash
GPU_ID=$1
SCENE_NUM=$2
cc0textures=$3
dataset_path=$4
s2_p1_gen_pbr_data=$5
shift 5
BLENDER_ROOT=${BLENDER_ROOT:-/home/zhanght2504/zhanght2504/runspace_yyx5}
BOP_TOOLKIT_PATH=${BOP_TOOLKIT_PATH:-/home/zhanght2504/zhanght2504/runspace_yyx5/HCCEPose/bop_toolkit}

export PYTHONPATH="$BOP_TOOLKIT_PATH${PYTHONPATH:+:$PYTHONPATH}"
for (( SCENE_ID=0; SCENE_ID<$SCENE_NUM; SCENE_ID++ ))
do
    SCENE_ID_PADDED=$(printf "%06d" $SCENE_ID)
    echo "Running scene $SCENE_ID_PADDED on GPU $GPU_ID"
    export EGL_DEVICE_ID=$GPU_ID
    cd "$dataset_path"
    blenderproc run --blender-install-path "$BLENDER_ROOT" "$s2_p1_gen_pbr_data" "$GPU_ID" "$cc0textures" "$@"
done
