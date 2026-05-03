#!/bin/bash
# Phase 2: Addressing Class Imbalance
# Uses weight_changed_model.py (class-weighted loss)
# Same 4 configurations as Phase 1, isolating the effect of weighted loss.

set -e
mkdir -p checkpoints/phase2

echo "===== Phase 2 (a): Base ====="
python weight_changed_model.py --batch_size 16 --epochs 10 --lr 0.001  --optimizer adam \
    --save_model --model_path ./checkpoints/phase2/p2_base.pth

echo "===== Phase 2 (b): Increased Epochs ====="
python weight_changed_model.py --batch_size 16 --epochs 20 --lr 0.001  --optimizer adam \
    --save_model --model_path ./checkpoints/phase2/p2_epoch20.pth

echo "===== Phase 2 (c): Reduced LR ====="
python weight_changed_model.py --batch_size 16 --epochs 10 --lr 0.0005 --optimizer adam \
    --save_model --model_path ./checkpoints/phase2/p2_lr0005.pth

echo "===== Phase 2 (d): Change to SGD ====="
python weight_changed_model.py --batch_size 16 --epochs 10 --lr 0.001  --optimizer sgd \
    --save_model --model_path ./checkpoints/phase2/p2_sgd.pth

echo "Phase 2 complete."
