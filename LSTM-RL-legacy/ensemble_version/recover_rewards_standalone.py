import sys
import os
import numpy as np
import torch
import copy
from collections import defaultdict

# Put the root in path just as the main script does
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Spoof sys.argv to trick the module into initializing exactly like the training run!
sys.argv = [
    'sac_ensemble_SUMO_linear_penalty.py', 
    '--use_sumo_env', 
    '--sumo_bridge', 'SUMO_ruiguang.online_control.rl_bridge',
    '--use_residual_control',
    '--bang_bang',
    '--run_name', 'Production_Augmented_BangBang_V7_Long'
]

# Import everything needed natively!
from ensemble_version.sac_ensemble_SUMO_linear_penalty import env, sac_trainer, safe_initialize_state, safe_step, build_action_template

def main():
    eps_str = os.environ.get('EPS_TO_RUN', '')
    worker_id = os.environ.get('WORKER_ID', '0')
    if not eps_str: return
    
    run_name = "Production_Augmented_BangBang_V7_Long"
    model_dir = f"model/sac_ensemble_SUMO_linear_penalty_{run_name}"
    
    eps = [int(e) for e in eps_str.split(',')]
    results = []
    
    for ep in eps:
        ckpt_path = f"{model_dir}/checkpoint_episode_{ep}"
        if not os.path.exists(ckpt_path + "_policy"):
            results.append((ep, np.nan))
            continue
            
        print(f"Worker {worker_id} evaluating episode {ep}")
        sac_trainer.policy_net.load_state_dict(torch.load(ckpt_path + '_policy', map_location=torch.device('cpu'), weights_only=False))
        
        env.reset()
        state_dict, _, _ = safe_initialize_state(env, render=False)
        
        episode_reward = 0
        done = False
        action_dict = build_action_template(state_dict)
        last_action_history = defaultdict(lambda: defaultdict(lambda: np.zeros(2, dtype=np.float32)))
        
        while not done:
            action_dict = build_action_template(state_dict, action_dict)
            env_action_dict = {}
            for line_id, buses in state_dict.items():
                env_action_dict[line_id] = {}
                for bus_id, history in buses.items():
                    if len(history) == 0:
                        continue
                        
                    # Build state vector WITH action history properly
                    if len(history) == 1:
                        if action_dict[line_id][bus_id] is None:
                            last_action = last_action_history[line_id][bus_id]
                            state_vec = np.concatenate([history[0], last_action])
                            action_val = sac_trainer.policy_net.get_action(torch.from_numpy(state_vec).float(), deterministic=True)
                            action_dict[line_id][bus_id] = action_val
                            
                    elif len(history) >= 2:
                        prev_action = last_action_history[line_id][bus_id]
                        current_action = action_dict[line_id][bus_id]
                        if current_action is None:
                            current_action = np.zeros(2, dtype=np.float32)
                        elif np.isscalar(current_action):
                            current_action = np.array([current_action], dtype=np.float32)
                        else:
                            current_action = np.array(current_action, dtype=np.float32).reshape(-1)
                            
                        # Update the tracking structures correctly!
                        last_action_history[line_id][bus_id] = current_action
                        
                        state_vec = np.concatenate([history[1], current_action])
                        action_val = sac_trainer.policy_net.get_action(torch.from_numpy(state_vec).float(), deterministic=True)
                        action_dict[line_id][bus_id] = action_val
                        
                    # Retrieve the selected action!
                    action_val = action_dict[line_id][bus_id]
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
            step_reward = sum(sum(bus.values()) for bus in reward_dict.values())
            episode_reward += step_reward
            
        results.append((ep, episode_reward))
        np.save(f"results_{worker_id}.npy", np.array(results))
        
    env.close()
    print(f"Worker {worker_id} finished.")

if __name__ == '__main__':
    main()
