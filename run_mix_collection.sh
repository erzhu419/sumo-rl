#!/bin/bash
set -e

# Configuration
EXPERT_MODEL="logs/sac_v2_bus_SUMO_best_result/sac_v2_episode_18_policy"
OUTPUT_DIR="offline_sumo/data"
NUM_WORKERS=14

echo "=== Starting Mixed Data Collection Strategy ==="
echo "Target: 2000 Zero Policy + 2000 Expert Policy"
echo "Features: Ladder Reward, Presence Filtering, Chunked Saving (Interval=20)"

# 1. Run Zero Policy Collection
echo ""
echo ">>> [Phase 1/2] Collecting 2000 episodes of ZERO Policy (No Control)..."
# Policy ID 0.0 for Zero/Uncontrolled
python offline_sumo/data/collect_data_parallel.py \
    --policy zero \
    --episodes 2000 \
    --output "${OUTPUT_DIR}/buffer_zero_parallel.hdf5" \
    --num_workers $NUM_WORKERS \
    --save_interval 20 \
    --policy_id 0.0

# 2. Run Expert Policy Collection (Diversified)
echo ""
echo ">>> [Phase 2/2] Collecting 2000 episodes of Diversified Expert Policies (400 eps each)..."
 
# Expert List: 15, 18, 20, 25, 30
# Total: 5 * 400 = 2000 episodes
MODELS=(15 18 20 25 30)

for EP in "${MODELS[@]}"; do
    MODEL_PATH="logs/sac_v2_bus_SUMO_best_result/sac_v2_episode_${EP}_policy"
    OUTPUT_FILE="${OUTPUT_DIR}/buffer_expert_${EP}_parallel.hdf5"
    
    echo "--------------------------------------------------------"
    echo ">> Collecting 400 episodes using Expert Model: Episode ${EP}"
    echo ">> Policy ID Label: ${EP}.0"
    
    python offline_sumo/data/collect_data_parallel.py \
        --policy expert \
        --episodes 400 \
        --model_path "$MODEL_PATH" \
        --output "$OUTPUT_FILE" \
        --num_workers $NUM_WORKERS \
        --save_interval 20 \
        --policy_id "${EP}.0"
done

echo ""
echo "=== Data Collection Complete! ==="
echo "You now have:"
echo "  - ${OUTPUT_DIR}/buffer_zero_parallel.hdf5 (2000 eps, ID 0.0)"
for EP in "${MODELS[@]}"; do
    echo "  - ${OUTPUT_DIR}/buffer_expert_${EP}_parallel.hdf5 (400 eps, ID ${EP}.0)"
done
