#!/bin/bash
# Environment comparison test: f543609 (old) vs HEAD (new) rl_bridge
# Runs in --train mode but with batch_size=999999 to prevent any training updates
# This matches the original training's episode-level behavior without corrupting the policy
# Usage: ./run_env_test.sh <f543609|head>

set -e
source /home/erzhu419/anaconda3/etc/profile.d/conda.sh
conda activate LSTM-RL

cd /home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version

CHECKPOINT="model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/checkpoint_episode_39"

if [ "$1" == "f543609" ]; then
    echo "=== Eval with f543609 rl_bridge (OLD env, update_freq=10) ==="
    python3 -u sac_ensemble_SUMO_linear_penalty.py \
      --train \
      --use_residual_control \
      --bang_bang \
      --max_episodes 41 \
      --batch_size 999999 \
      --resume_checkpoint "$CHECKPOINT" \
      --run_name "ENV_EVAL_f543609" \
      --update_passenger_freq 10 \
      --sumo_bridge "SUMO_ruiguang.online_control.rl_bridge_f543609:build_bridge"
elif [ "$1" == "head" ]; then
    echo "=== Eval with HEAD rl_bridge (NEW env, update_freq=1) ==="
    python3 -u sac_ensemble_SUMO_linear_penalty.py \
      --train \
      --use_residual_control \
      --bang_bang \
      --max_episodes 41 \
      --batch_size 999999 \
      --resume_checkpoint "$CHECKPOINT" \
      --run_name "ENV_EVAL_HEAD" \
      --update_passenger_freq 1 \
      --sumo_bridge "SUMO_ruiguang.online_control.rl_bridge:build_bridge"
else
    echo "Usage: $0 <f543609|head>"
    exit 1
fi
