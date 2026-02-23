import os
import sys
import torch
import numpy as np
import time

# Mock sys.argv so argparse in the imported module doesn't fail
sys.argv = ['sac_ensemble_SUMO_linear_penalty.py', '--test', '--use_sumo_env']

from sac_ensemble_SUMO_linear_penalty import (
    env, build_action_template, safe_initialize_state, 
    safe_step, sac_trainer, args, device, action_dim
)

MODEL_PATH = "model/sac_ensemble_SUMO_linear_penalty_ensemble_run/checkpoint_episode_19"
print(f"Loading evaluated model from: {MODEL_PATH}")
sac_trainer.load_model(MODEL_PATH)
sac_trainer.policy_net.eval()
sac_trainer.soft_q_net.eval()

DETERMINISTIC = True
render = False
station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1

print("Starting evaluation for 1 episode...")
state_dict, reward_dict, _ = safe_initialize_state(env, render=render)
action_dict = build_action_template(state_dict)

done = False
episode_reward = 0
step = 0

while not done:
    action_dict = build_action_template(state_dict, action_dict)
    for line_id, buses in state_dict.items():
        for bus_id, history in buses.items():
            if len(history) == 0:
                continue
            if len(history) >= 1:
                # get action
                state_vec = np.array(history[-1])
                if args.use_state_norm:
                    state_vec = sac_trainer.state_norm(state_vec)
                    
                action = sac_trainer.policy_net.get_action(torch.from_numpy(state_vec).float(), deterministic=DETERMINISTIC)
                action_dict[line_id][bus_id] = action

    state_dict, reward_dict, done, _ = safe_step(env, action_dict, render=render)
    
    # accumulation of reward
    for line_id, buses in reward_dict.items():
        for bus_id, r in buses.items():
            episode_reward += r
    
    step += 1

print(f"Evaluation finished! Reward for Episode 19 model: {episode_reward}")
