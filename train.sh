#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_train.parquet>"
    exit 1
fi

INPUT_DATA_PATH=$(realpath "$1")
TARGET_PATH=$(realpath -m "datasets/train.parquet")

echo "Starting Training Pipeline"
echo "Input Data: $INPUT_DATA_PATH"

mkdir -p datasets

if [ "$INPUT_DATA_PATH" = "$TARGET_PATH" ]; then
    echo "[Setup] Input file is already in the target location. Skipping symlink creation."
else
    if [ -f "$INPUT_DATA_PATH" ]; then
        ln -sf "$INPUT_DATA_PATH" datasets/train.parquet
        echo "[Setup] Symlink created: datasets/train.parquet -> $INPUT_DATA_PATH"
    else
        echo "[Error] File not found: $INPUT_DATA_PATH"
        exit 1
    fi
fi

echo "[Train] Running scripts/train.py..."
python3 scripts/train.py