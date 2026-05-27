#!/bin/bash

# Loading the required module
# source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base

# Use the correct python virtual environment containing pyatb
PYTHON_EXEC="/home/goodchong/geths/infer/test_next/NextHAM/.venv/bin/python"

CUDA_VISIBLE_DEVICES=0,1 ${PYTHON_EXEC} infer.py \
    --output-dir './test_res/' \
    --model-name 'graph_attention_transformer_nonlinear_materials_ham_soc' \
    --input-irreps '64x0e' \
    --radius 8.0 \
    --is-accurate-label \
    --trace-out-len 81 \
    --batch-size 1 \
    --eval-batch-size 1 \
    --weight-decay 0 \
    --num-basis 64 \
    --workers 0 \
    --with-trace \
    --energy-weight 1 \
    --force-weight 80 \
    --test-interval 10000 \
    --target 'hamiltonian' \
    --target-blocks-type 'all' \
    --checkpoint-path1 ./pretrained_models/model_range0_curr.pth.tar \
    --checkpoint-path2 ./pretrained_models/model_range1_curr.pth.tar \
    --checkpoint-path3 ./pretrained_models/model_range2_curr.pth.tar \
    --checkpoint-path4 ./pretrained_models/model_range3_curr.pth.tar