#!/bin/bash
set -e

# Configuration
OUTPUT_DIR="offline_sumo/data"
NUM_WORKERS=250

echo "=== Starting Full Offline V7 Evaluation Strategy ==="
echo "Target:"
echo "- Zero Policy: 2000 Episodes"
echo "- Heuristic (ac0be9): 500 Episodes"
echo "- 5 Expert Policies (V7): 500 Episodes each (Total 2500)"

# 1. Run Zero Policy Collection
echo ""
echo ">>> [Phase 1/3] Collecting 2000 episodes of ZERO Policy (No Control)..."
python3 offline_sumo/data/collect_data_parallel.py \
    --policy zero \
    --episodes 2000 \
    --output "${OUTPUT_DIR}/buffer_zero_parallel.hdf5" \
    --num_workers $NUM_WORKERS \
    --save_interval 10 \
    --policy_id 0.0

# 2. Run Heuristic Policy Collection
echo ""
echo ">>> [Phase 2/3] Collecting 500 episodes of HEURISTIC Policy (ac0be9 logic)..."
python3 offline_sumo/data/collect_data_parallel.py \
    --policy heuristic \
    --episodes 500 \
    --output "${OUTPUT_DIR}/buffer_heuristic_parallel.hdf5" \
    --num_workers $NUM_WORKERS \
    --save_interval 10 \
    --policy_id 0.1

# 3. Run Expert Policy Collection (Diversified)
echo ""
echo ">>> [Phase 3/3] Collecting 500 episodes of Diversified V7 Expert Policies (36, 37, 38, 39, 42)..."
 
# Expert List: 36, 37, 38, 39, 42
MODELS=(36 37 38 39 42)

for EP in "${MODELS[@]}"; do
    MODEL_PATH="/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/checkpoint_episode_${EP}_policy"
    OUTPUT_FILE="${OUTPUT_DIR}/buffer_expert_${EP}_parallel.hdf5"
    
    echo "--------------------------------------------------------"
    echo ">> Collecting 500 episodes using Expert Model: Episode ${EP}"
    echo ">> Policy ID Label: ${EP}.0"
    
    python3 offline_sumo/data/collect_data_parallel.py \
        --policy expert \
        --episodes 500 \
        --model_path "$MODEL_PATH" \
        --output "$OUTPUT_FILE" \
        --num_workers $NUM_WORKERS \
        --save_interval 10 \
        --policy_id "${EP}.0"
done

echo ""
echo "=== Data Collection Complete! ==="

# 4. Merge all per-policy HDF5 into one combined file
echo ""
echo ">>> [Phase 4/4] Merging all buffers into buffer_combined.hdf5..."
python3 offline_sumo/data/merge_all.py \
    --input_dir "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/buffer_combined.hdf5" \
    --pattern "buffer_*.hdf5"

echo ""
echo "=== All Done! buffer_combined.hdf5 is ready for transfer ==="
