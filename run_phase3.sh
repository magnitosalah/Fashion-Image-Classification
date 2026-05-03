#!/bin/bash
# Phase 3: Final Hyperparameter Optimization
# Full pipeline (pretrained + augmentation + class-weighted loss),
# SGD fixed, extended epochs (30), LR grid search.
# The lr=5e-4 run produces the FINAL model reported in the paper (87.08%).

set -e
mkdir -p checkpoints/phase3

echo "===== Phase 3 (a): SGD lr=1e-3 ====="
python final_model.py --batch_size 16 --epochs 30 --lr 0.001  --optimizer sgd \
    --save_model --model_path ./checkpoints/phase3/p3_lr001.pth

echo "===== Phase 3 (b): SGD lr=5e-4  [FINAL MODEL] ====="
python final_model.py --batch_size 16 --epochs 30 --lr 0.0005 --optimizer sgd \
    --save_model --model_path ./efficientnet_v2s.pth

echo "===== Phase 3 (c): SGD lr=1e-4 ====="
python final_model.py --batch_size 16 --epochs 30 --lr 0.0001 --optimizer sgd \
    --save_model --model_path ./checkpoints/phase3/p3_lr0001.pth

echo "Phase 3 complete. Final model: ./efficientnet_v2s.pth"
