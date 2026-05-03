#!/bin/bash
# Baseline: EfficientNet-V2-S trained from scratch, no enhancements.
set -e
python baseline_model.py --batch_size 16 --epochs 10 --lr 0.001 --optimizer adam \
    --save_model --model_path ./baseline.pth
echo "Baseline complete: ./baseline.pth"
