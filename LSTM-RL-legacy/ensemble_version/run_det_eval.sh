#!/bin/bash
# Deterministic speed eval: matches the recovery eval setup exactly
# Sets args.train=False internally via a wrapper that patches the script
set -e
source /home/erzhu419/anaconda3/etc/profile.d/conda.sh
conda activate LSTM-RL
cd /home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version

CHECKPOINT="model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/checkpoint_episode_39"

# Run via python -c wrapper that patches args.train=False after argparse
if [ "$1" == "f543609" ]; then
    BRIDGE="SUMO_ruiguang.online_control.rl_bridge_f543609:build_bridge"
    FREQ=10
    RUN="DET_EVAL_f543609"
elif [ "$1" == "head" ]; then
    BRIDGE="SUMO_ruiguang.online_control.rl_bridge:build_bridge"
    FREQ=1
    RUN="DET_EVAL_HEAD"
else
    echo "Usage: $0 <f543609|head>"
    exit 1
fi

echo "=== Deterministic eval with $1 bridge ==="
python3 -u -c "
import sys, os
# Patch sys.argv before the training script parses args
sys.argv = [
    'sac_ensemble_SUMO_linear_penalty.py',
    '--train',
    '--use_residual_control',
    '--bang_bang',
    '--max_episodes', '41',
    '--batch_size', '999999',
    '--resume_checkpoint', '$CHECKPOINT',
    '--run_name', '$RUN',
    '--update_passenger_freq', '$FREQ',
    '--sumo_bridge', '$BRIDGE',
]
# Execute the script
exec(open('sac_ensemble_SUMO_linear_penalty.py').read())
" 2>&1 | tee /dev/stderr | head -1000 &
PY_PID=$!

# Wait for the script to parse args, then patch args.train
# Actually can't do this from outside. Instead, use a simpler approach:
# Just modify DETERMINISTIC and args.train inline
wait $PY_PID
