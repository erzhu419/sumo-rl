import os
import sys
import torch
import numpy as np

# Let's just import the main script and use its SumoRLEnv!
from sac_ensemble_SUMO_linear_penalty import SumoRLEnv, SAC_Trainer, safe_initialize_state, safe_step

print("Initializing environment...")
env = SumoRLEnv(
    sumocfg_file='../SUMO_ruiguang/online_control/1_run_online_simulation_d.sumocfg',
    update_passenger_freq=1,
    mode='cli',
    total_time=3600,
    bang_bang=True,
    scale=1.0,
    gui_f=False
)

agent = SAC_Trainer(
    state_dim=29,
    action_space=env.action_space,
    max_agent_num=env.max_agent_num,
    ensemble_size=2,
    maximum_alpha=0.4,
    auto_entropy=True,
    beta=-2,
    log_dir=None
)

model_dir = "model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long"
recovered_rewards = []

for ep in range(49):
    ckpt_path = os.path.join(model_dir, f"checkpoint_episode_{ep}")
    if not os.path.exists(ckpt_path + "_policy"):
        recovered_rewards.append(0.0)
        continue
    
    print(f"Evaluating Episode {ep}...")
    agent.load_model(ckpt_path)
    env.reset()
    state_dict, _, _ = safe_initialize_state(env, render=False)
    
    episode_reward = 0
    done = False
    
    while not done:
        env_action_dict = {}
        for line_id, buses in state_dict.items():
            env_action_dict[line_id] = {}
            for bus_id, state in buses.items():
                action_val = agent.policy_net.get_action(state, deterministic=True)
                a_hold, a_speed = action_val[0], action_val[1]
                hold = np.clip((a_hold + 1.0) * 60.0, 0.0, 120.0)
                if a_speed > 0.6: speed = 1.2
                elif a_speed > 0.2: speed = 1.1
                elif a_speed > -0.2: speed = 1.0
                elif a_speed > -0.6: speed = 0.9
                else: speed = 0.8
                env_action_dict[line_id][bus_id] = [hold, speed]
        
        state_dict, reward_dict, done, _ = safe_step(env, env_action_dict, render=False)
        step_reward = sum(sum(bus.values()) for bus in reward_dict.values())
        episode_reward += step_reward
        
    print(f"Score: {episode_reward:.2f}")
    recovered_rewards.append(episode_reward)

env.close()
np.save(os.path.join(model_dir, "recovered_first_48_eval.npy"), np.array(recovered_rewards))
print(f"Max: {np.max(recovered_rewards)} at index {np.argmax(recovered_rewards)}")
