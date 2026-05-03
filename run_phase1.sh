#!/bin/bash
# Phase 1: Basic Enhancements (Pretrained Weights + Data Augmentation)
# Uses Transfer_model.py — no class-weighted loss yet.

set -e
mkdir -p checkpoints/phase1

echo "===== Phase 1 (a): Base ====="
python Transfer_model.py --batch_size 16 --epochs 10 --lr 0.001  --optimizer adam \
    --save_model --model_path ./checkpoints/phase1/p1_base.pth

echo "===== Phase 1 (b): Increased Epochs ====="
python Transfer_model.py --batch_size 16 --epochs 20 --lr 0.001  --optimizer adam \
    --save_model --model_path ./checkpoints/phase1/p1_epoch20.pth

echo "===== Phase 1 (c): Reduced LR ====="
python Transfer_model.py --batch_size 16 --epochs 10 --lr 0.0005 --optimizer adam \
    --save_model --model_path ./checkpoints/phase1/p1_lr0005.pth

echo "===== Phase 1 (d): Change to SGD ====="
python Transfer_model.py --batch_size 16 --epochs 10 --lr 0.001  --optimizer sgd \
    --save_model --model_path ./checkpoints/phase1/p1_sgd.pth

echo "Phase 1 complete."
