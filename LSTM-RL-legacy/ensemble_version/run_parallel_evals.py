import os
import sys
import multiprocessing
import numpy as np

def evaluate_checkpoint(args_tuple):
    ep_idx, model_dir, target_script = args_tuple
    
    ckpt_path = os.path.join(model_dir, f"checkpoint_episode_{ep_idx}")
    if not os.path.exists(ckpt_path + "_policy"):
        print(f"Worker {ep_idx}: Checkpoint not found at {ckpt_path}")
        return ep_idx, np.nan
        
    print(f"Worker {ep_idx}: Starting evaluation for {ckpt_path}")
    
    # We must import inside the worker process to avoid TraCI socket conflicts
    from sac_ensemble_SUMO_linear_penalty import SumoRLEnv, SAC_Trainer, safe_initialize_state, safe_step
    import torch
    
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
        
        # Manually load avoiding CUDA errors if on headless CPU
        agent.policy_net.load_state_dict(torch.load(ckpt_path + '_policy', weights_only=False, map_location=torch.device('cpu')))
        try:
            agent.soft_q_net.load_state_dict(torch.load(ckpt_path + '_q', weights_only=False, map_location=torch.device('cpu')))
        except:
            pass # Q net not strictly needed for deterministic evaluation, but good to have
            
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
    
    # We want to recover 0 to 48
    target_episodes = list(range(49))
    num_workers = 14
    
    print(f"Starting parallel recovery for {len(target_episodes)} episodes using {num_workers} workers...")
    
    tasks = [(ep, model_dir, "sac_ensemble_SUMO_linear_penalty.py") for ep in target_episodes]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(evaluate_checkpoint, tasks)
        
    # Sort results to ensure correct ordering
    results.sort(key=lambda x: x[0])
    recovered_rewards = [r[1] for r in results]
    
    # Save purely the recovered ones just in case
    np.save(os.path.join(model_dir, "recovered_first_48_eval.npy"), np.array(recovered_rewards))
    
    # Now merge into the main rewards.npy
    main_npy_path = os.path.join(log_dir, 'rewards.npy')
    if os.path.exists(main_npy_path):
        import shutil
        shutil.copy(main_npy_path, os.path.join(log_dir, 'rewards_latest_backup2.npy'))
        
        main_r = np.load(main_npy_path)
        
        # Remove the NaNs we just padded earlier
        if len(main_r) >= 49 and np.isnan(main_r[0]):
            main_r = main_r[49:]
            
        print(f"Original valid array length: {len(main_r)}")
            
        new_combined = np.concatenate((np.array(recovered_rewards), main_r))
        np.save(main_npy_path, new_combined)
        
        print(f"Successfully merged! New total length: {len(new_combined)}")
        valid_indices = ~np.isnan(new_combined)
        max_val = np.nanmax(new_combined)
        max_idx = np.nanargmax(new_combined)
        print(f"Max reward is {max_val} at index {max_idx}")
    else:
        print("Could not find main rewards.npy to merge into.")
