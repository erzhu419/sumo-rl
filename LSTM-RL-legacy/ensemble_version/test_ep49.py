import sys, os, copy
import numpy as np, torch
from collections import defaultdict

ENSEMBLE_DIR = os.path.dirname(os.path.abspath('recover_rewards_v2.py'))
# Spoof sys.argv
sys.argv = [
    'sac_ensemble_SUMO_linear_penalty.py',
    '--use_sumo_env',
    '--sumo_bridge', 'SUMO_ruiguang.online_control.rl_bridge:build_bridge',
    '--use_residual_control',
    '--bang_bang',
    '--run_name', 'Production_Augmented_BangBang_V7_Long',
]

from sac_ensemble_SUMO_linear_penalty import (
    env, sac_trainer, args, safe_initialize_state, safe_step,
    build_action_template, action_dim, get_reward_value
)

ckpt = f"model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/checkpoint_episode_49"
sac_trainer.load_model(ckpt)
sac_trainer.policy_net.eval()
sac_trainer.soft_q_net.eval()

env.reset()
state_dict, reward_dict, _ = safe_initialize_state(env, render=False)

done = False
episode_reward = 0
station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1
action_dict = build_action_template(state_dict)
last_action_history = defaultdict(lambda: defaultdict(lambda: np.zeros(action_dim, dtype=np.float32)))

step = 0
while not done:
    action_dict = build_action_template(state_dict, action_dict)
    for line_id, buses in state_dict.items():
        for bus_id, history in buses.items():
            if len(history) == 0: continue
            if len(history) == 1:
                if action_dict[line_id][bus_id] is None:
                    last_action = last_action_history[line_id][bus_id]
                    state_vec = np.concatenate([history[0], last_action])
                    if args.use_state_norm:
                        state_vec = sac_trainer.state_norm(state_vec)
                    sv = torch.from_numpy(state_vec).float()
                    action = sac_trainer.policy_net.get_action(sv, deterministic=True) # RECOVERY WAY
                    action_dict[line_id][bus_id] = action
            elif len(history) >= 2:
                current_action = action_dict[line_id][bus_id]
                if current_action is None:
                    current_action = np.zeros(action_dim, dtype=np.float32)
                elif np.isscalar(current_action):
                    current_action = np.array([current_action], dtype=np.float32)
                else:
                    current_action = np.array(current_action, dtype=np.float32).reshape(-1)

                if history[0][station_feature_idx] != history[1][station_feature_idx]:
                    current_reward = get_reward_value(reward_dict, line_id, bus_id)
                    episode_reward += current_reward

                state_dict[line_id][bus_id] = history[1:]
                state_vec_next = np.concatenate([state_dict[line_id][bus_id][0], current_action])
                last_action_history[line_id][bus_id] = current_action

                if args.use_state_norm:
                    state_vec_next = sac_trainer.state_norm(state_vec_next)
                sv = torch.from_numpy(state_vec_next).float()
                action = sac_trainer.policy_net.get_action(sv, deterministic=True) # RECOVERY WAY
                action_dict[line_id][bus_id] = action

    env_action_dict = copy.deepcopy(action_dict)
    for line_id, buses in env_action_dict.items():
        for bus_id, action_val in buses.items():
            if action_val is not None:
                a_hold, a_speed = action_val[0], action_val[1]
                hold = np.clip((a_hold + 1.0) * 60.0, 0.0, 120.0)
                if a_speed > 0.6: speed = 1.2
                elif a_speed > 0.2: speed = 1.1
                elif a_speed > -0.2: speed = 1.0
                elif a_speed > -0.6: speed = 0.9
                else: speed = 0.8
                env_action_dict[line_id][bus_id] = [hold, speed]

    state_dict, reward_dict, done, _ = safe_step(env, env_action_dict, render=False)

print(f"Deterministic Episode 49 Reward: {episode_reward}")
