import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

# Create the original environment
import importlib
bridge_module = importlib.import_module("SUMO_ruiguang.online_control.rl_bridge")
bridge = bridge_module.build_bridge(root_dir="SUMO_ruiguang", gui=False, update_freq=10)
from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv

env_vanilla = SumoBusHoldingEnv(
    root_dir=os.path.join(PROJECT_ROOT, "SUMO_ruiguang/online_control"),
    schedule_file="initialize_obj/save_obj_bus.add.xml",
    decision_provider=bridge.get('decision_provider'),
    action_executor=bridge.get('action_executor'),
    reset_callback=bridge.get('reset_callback'),
    close_callback=bridge.get('close_callback'),
    debug=False,
    reward_type="linear_penalty",
)

def build_action_template(state_dict):
    action_dict = {}
    for line_id, buses in state_dict.items():
        action_dict[line_id] = {}
        for bus_id in buses:
            action_dict[line_id][bus_id] = np.zeros(1, dtype=np.float32)
    return action_dict

def get_reward_value(reward_dict, line_id, bus_id):
    if line_id in reward_dict and bus_id in reward_dict[line_id]:
        return reward_dict[line_id][bus_id][-1]
    return 0

def run_episode(env):
    env.reset()
    state_dict, reward_dict, _ = env.initialize_state(render=False)
    
    done = False
    episode_reward = 0
    
    while not done:
        action_dict = build_action_template(state_dict)
        state_dict, reward_dict, done, _ = env.step(action_dict, debug=False, render=False)
        
        # Accumulate rewards the naive way used in legacy code
        for line_id, buses in state_dict.items():
            for bus_id in buses:
                 # Check if history > 1 and stop advanced
                 if len(state_dict[line_id][bus_id]) >= 2:
                     if state_dict[line_id][bus_id][0][1] != state_dict[line_id][bus_id][1][1]:
                         episode_reward += get_reward_value(reward_dict, line_id, bus_id)
                         
    return episode_reward

r1 = run_episode(env_vanilla)
r2 = run_episode(env_vanilla) # Run again

print(f"Vanilla Pure Env Run 1: {r1}")
print(f"Vanilla Pure Env Run 2: {r2}")

