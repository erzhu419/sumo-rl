import os
import sys

run_name = "Production_Augmented_BangBang_V7_Long"
model_dir = f"model/sac_ensemble_SUMO_linear_penalty_{run_name}"
log_dir = f"logs/sac_ensemble_SUMO_linear_penalty_{run_name}"

episodes = list(range(49))
num_workers = 14
os.makedirs("eval_workers", exist_ok=True)

chunks = [episodes[i::num_workers] for i in range(num_workers)]

for w in range(num_workers):
    eps_to_run = chunks[w]
    if not eps_to_run: continue
    
    script_path = f"eval_workers/worker_{w}.py"
    with open(script_path, "w") as f:
        f.write(f"""import os
import sys
import numpy as np

# Nasty hack to get around all the relative paths in the code base
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import exactly like the main script
from sac_ensemble_SUMO_linear_penalty import SumoRLEnv, SAC_Trainer, safe_initialize_state, safe_step
import torch

def eval_ep(ep):
    ckpt_path = "{model_dir}/checkpoint_episode_" + str(ep)
    if not os.path.exists(ckpt_path + "_policy"):
        return ep, np.nan
        
    print(f"Worker {w} evaluating episode {{ep}}")
    try:
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
        agent.policy_net.load_state_dict(torch.load(ckpt_path + '_policy', map_location=torch.device('cpu'), weights_only=False))
        
        env.reset()
        state_dict, _, _ = safe_initialize_state(env, render=False)
        
        episode_reward = 0
        done = False
        while not done:
            env_action_dict = {{}}
            for line_id, buses in state_dict.items():
                env_action_dict[line_id] = {{}}
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
            
        env.close()
        return ep, episode_reward
    except Exception as e:
        print(f"Error on ep {{ep}}: {{e}}")
        try: env.close() 
        except: pass
        return ep, np.nan

if __name__ == '__main__':
    episodes = {eps_to_run}
    results = []
    for ep in episodes:
        results.append(eval_ep(ep))
        
    np.save(f"eval_workers/results_{w}.npy", np.array(results))
    print(f"Worker {w} finished.")
""")

print("Generated all worker scripts.")
