#!/usr/bin/env python3
"""
Minimal eval script: load ep39 checkpoint, run 1 episode with DETERMINISTIC speed mapping.
Matches the exact setup of eval_recovery_script.py that produced -655,626.

Usage:
  python3 eval_env_comparison.py --bridge f543609   # old env
  python3 eval_env_comparison.py --bridge head       # new env
"""
import sys, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bridge', choices=['f543609', 'head'], required=True)
cli_args = parser.parse_args()

# Set sys.argv for the training script
CHECKPOINT = "model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/checkpoint_episode_39"

if cli_args.bridge == 'f543609':
    bridge_module = "SUMO_ruiguang.online_control.rl_bridge_f543609:build_bridge"
    freq = "10"
    run_name = "DET_EVAL_f543609"
else:
    bridge_module = "SUMO_ruiguang.online_control.rl_bridge:build_bridge"
    freq = "1"
    run_name = "DET_EVAL_HEAD"

sys.argv = [
    'sac_ensemble_SUMO_linear_penalty.py',
    '--train',
    '--use_residual_control',
    '--bang_bang',
    '--max_episodes', '41',
    '--batch_size', '999999',
    '--resume_checkpoint', CHECKPOINT,
    '--run_name', run_name,
    '--update_passenger_freq', freq,
    '--sumo_bridge', bridge_module,
]

# Now exec the training script, but we'll monkey-patch after it loads
# Import the script's content but intercept args.train
script_path = os.path.join(os.path.dirname(__file__), 'sac_ensemble_SUMO_linear_penalty.py')
script_content = open(script_path).read()

# Replace the DETERMINISTIC flag and force args.train=False for speed mapping 
script_content = script_content.replace(
    'DETERMINISTIC = False',
    'DETERMINISTIC = False  # policy still stochastic\n# PATCH: Force deterministic speed mapping to match recovery eval\nimport atexit\ndef _patch_train():\n    global args\n    args.train = False\n_patch_train()'
)

exec(compile(script_content, script_path, 'exec'))
