#!/bin/bash

# Define base directory and target directories
BASE_DIR=$(pwd)
TARGET_DIR="${BASE_DIR}/1000atom"
DATA_DIR="${BASE_DIR}/data"
DATASET_DIR="${BASE_DIR}/datasets"

# Use the correct python virtual environment containing pyatb
PYTHON_EXEC="/home/goodchong/geths/infer/test_next/NextHAM/.venv/bin/python"

# Create necessary directories
mkdir -p "${DATA_DIR}"
mkdir -p "${DATASET_DIR}"

OUTPUT_PTH="${DATA_DIR}/cpp_input_inference.pth"
FERMI_ENERGY="13.97"

echo "========================================="
echo " Step 1: Pre-processing (C++ Engine)"
echo "========================================="
# Read CIF input directly in the C++ pre-processing executable.
./pre_post_process/cpp/build/nextham_preprocess \
    "${TARGET_DIR}/OUT.ABACUS/STRU.cif" \
    "${TARGET_DIR}/OUT.ABACUS/" \
    4 \
    8.0 \
    "${OUTPUT_PTH}" \
    --format cif

echo "========================================="
echo " Step 1.5: Generating infer_ori.txt"
echo "========================================="
# Write the generated .pth file path into datasets/infer_ori.txt
INFER_ROOT="${DATASET_DIR}/infer_ori.txt"
echo "${OUTPUT_PTH}" > "${INFER_ROOT}"
echo "Root saved to: ${INFER_ROOT}"

echo "========================================="
echo " Step 2: Combine Data and Run Inference"
echo "========================================="
# Combine data and run the inference script
${PYTHON_EXEC} combine_data_infer.py
