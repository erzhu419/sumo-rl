import os
import sys
import multiprocessing
import numpy as np

# Need to append the proper paths so the worker processes can resolve modules natively
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

def evaluate_checkpoint(args_tuple):
    ep_idx, model_dir, log_dir = args_tuple
    
    ckpt_path = os.path.join(model_dir, f"checkpoint_episode_{ep_idx}")
    if not os.path.exists(ckpt_path + "_policy"):
        print(f"Worker {ep_idx}: Checkpoint not found at {ckpt_path}")
        return ep_idx, np.nan
        
    print(f"Worker {ep_idx}: Starting evaluation for {ckpt_path}")
    
    # Import inside worker
    import torch
    from ensemble_version.sac_ensemble_SUMO_linear_penalty import SAC_Trainer, safe_initialize_state, safe_step
    from env.sim import SumoEnv as SumoRLEnv
    
    try:
        env = SumoRLEnv(
            sumo_cfg='../SUMO_ruiguang/online_control/1_run_online_simulation_d.sumocfg',
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
        
        # Load weights on CPU without weights_only=True CUDA unpickling crashes
        agent.policy_net.load_state_dict(torch.load(ckpt_path + '_policy', map_location=torch.device('cpu'), weights_only=False))
        try:
            agent.soft_q_net.load_state_dict(torch.load(ckpt_path + '_q', map_location=torch.device('cpu'), weights_only=False))
        except:
            pass
            
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
            
        env.close()
        print(f"Worker {ep_idx}: Completed with score {episode_reward:.2f}")
        return ep_idx, episode_reward
        
    except Exception as e:
        print(f"Worker {ep_idx}: FAILED with error {e}")
        try:
            env.close()
        except: pass
        return ep_idx, np.nan

if __name__ == '__main__':
    run_name = "Production_Augmented_BangBang_V7_Long"
    model_dir = f"model/sac_ensemble_SUMO_linear_penalty_{run_name}"
    log_dir = f"logs/sac_ensemble_SUMO_linear_penalty_{run_name}"
    
    target_episodes = list(range(49))
    num_workers = 14
    
    print(f"Starting parallel recovery for {len(target_episodes)} episodes using {num_workers} workers...")
    
    tasks = [(ep, model_dir, log_dir) for ep in target_episodes]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(evaluate_checkpoint, tasks)
        
    results.sort(key=lambda x: x[0])
    recovered_rewards = [r[1] for r in results]
    
    np.save(os.path.join(model_dir, "recovered_first_48_eval.npy"), np.array(recovered_rewards))
    
    main_npy_path = os.path.join(log_dir, 'rewards.npy')
    if os.path.exists(main_npy_path):
        main_r = np.load(main_npy_path)
        
        # Strip exact block of 49 leading NaNs that we patched earlier 
        if len(main_r) >= 49:
            # Check if all first 49 are NaNs to confirm it's our patch
            if np.all(np.isnan(main_r[:49])):
                main_r = main_r[49:]
            
        new_combined = np.concatenate((np.array(recovered_rewards), main_r))
        np.save(main_npy_path, new_combined)
        
        print(f"Successfully merged! New length: {len(new_combined)}")
    else:
        print("Main file not found.")
