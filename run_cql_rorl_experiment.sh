#!/bin/bash

# Define paths
MERGE_SCRIPT="offline_sumo/data/merge_buffers.py"
OUTPUT_BUFFER="offline_sumo/data/merged_buffer_all.hdf5"
CQL_SCRIPT="offline_sumo/train/train_cql.py"
RORL_SCRIPT="deploy/offline_sumo/RORL/train_rorl.py"

echo "=========================================="
echo "Step 1: Merging Data Buffers"
echo "=========================================="

# Check if merged file already exists
if [ -f "$OUTPUT_BUFFER" ]; then
    echo "Merged buffer $OUTPUT_BUFFER already exists. Skipping merge."
else
    python $MERGE_SCRIPT --output $OUTPUT_BUFFER
fi

if [ ! -f "$OUTPUT_BUFFER" ]; then
    echo "Error: Merged buffer $OUTPUT_BUFFER not created."
    exit 1
fi

echo "Data ready: $OUTPUT_BUFFER"
echo ""

echo "=========================================="
echo "Step 2: Training CQL"
echo "=========================================="
# Using fixed steps per epoch for faster feedback
python $CQL_SCRIPT \
    --dataset $OUTPUT_BUFFER \
    --epochs 200 \
    --steps_per_epoch 1000 \
    --batch_size 2048 \
    --hidden_dim 32 \
    --cql_weight 0.2 \
    --learning_rate 1e-5

echo "CQL Training complete."
echo ""

echo "=========================================="
echo "Step 3: Training RORL"
echo "=========================================="
python $RORL_SCRIPT \
    --dataset $OUTPUT_BUFFER \
    --epochs 200 \
    --steps_per_epoch 1000 \
    --batch_size 2048 \
    --learning_rate 1e-5 \
    --hidden_dim 64 \
    --beta_ood 0.01 \
    --beta_uncertainty 2.0

echo "RORL Training complete."
echo "=========================================="
echo "All tasks finished."
