#!/bin/bash
# Sequential training of all 6 ACM model configurations
# Each config: 30 epochs, batch_size 8
# Total time estimate: ~9 hours on 4GB GPU

set -e

ROOT="c:/Users/administrator1/Desktop/大二/infrared_detection_code (1)"
export SIRST_DATA_ROOT="$ROOT/data/sirst/"
PYTHON="/c/Users/administrator1/miniconda3/envs/yolov8/python"
TRAIN="$ROOT/acm/train.py"

echo "=========================================="
echo " ACM Infrared Detection — Full Training"
echo " 6 configs × 30 epochs each"
echo " Start: $(date)"
echo "=========================================="

CONFIGS=(
    "FPN BiLocal"
    "FPN AsymBi"
    "FPN BiGlobal"
    "UNet BiLocal"
    "UNet AsymBi"
    "UNet BiGlobal"
)

TOTAL=${#CONFIGS[@]}
CURRENT=0

for cfg in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    BACKBONE=$(echo "$cfg" | cut -d' ' -f1)
    FUSE=$(echo "$cfg" | cut -d' ' -f2)

    echo ""
    echo "=========================================="
    echo " [$CURRENT/$TOTAL] $BACKBONE + $FUSE"
    echo " Start: $(date)"
    echo "=========================================="

    cd "$ROOT/acm"
    $PYTHON train.py \
        --backbone-mode "$BACKBONE" \
        --fuse-mode "$FUSE" \
        --batch-size 8 \
        --epochs 30 \
        --learning_rate 0.001

    echo " [$CURRENT/$TOTAL] $BACKBONE + $FUSE — DONE at $(date)"
done

echo ""
echo "=========================================="
echo " ALL 6 CONFIGS COMPLETE!"
echo " End: $(date)"
echo "=========================================="

# Print summary
echo ""
echo "====== FINAL RESULTS ======"
for dir in "$ROOT/acm/result/"*/; do
    name=$(basename "$dir")
    latest=$(ls -t "$dir/checkpoint/"*.pkl 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        fname=$(basename "$latest")
        echo "  $name  →  $fname"
    fi
done
