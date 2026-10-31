import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import glob

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from offline_sumo.envs.sumo_env import SumoBusHoldingEnv

def calculate_reward(prev_obs, current_obs, current_action, fwd_present=True, bwd_present=True):
    """
    Reconstruct reward from transitions.
    Obs structure: [line, bus, station, time, dir, f_h, b_h, wait, target, duration, time]
    Indices: f_h=5, b_h=6, target=8
    """
    fwd_h = current_obs[5]
    bwd_h = current_obs[6]
    target = current_obs[8]
    
    # Logic reverted to match legacy data (Linear Penalty)
    def headway_reward(headway, tgt):
        return -abs(headway - tgt)
            
    forward_reward = headway_reward(fwd_h, target) if fwd_present else None
    backward_reward = headway_reward(bwd_h, target) if bwd_present else None
    
    if forward_reward is not None and backward_reward is not None:
        fwd_dev = abs(fwd_h - target)
        bwd_dev = abs(bwd_h - target)
        weight = fwd_dev / (fwd_dev + bwd_dev + 1e-6)
        
        similarity_bonus = -abs(fwd_h - bwd_h) * 0.5
        reward = forward_reward * weight + backward_reward * (1 - weight) + similarity_bonus
    elif forward_reward is not None:
        reward = forward_reward
    elif backward_reward is not None:
        reward = backward_reward
    else:
        reward = -50.0
        
    if (fwd_present and abs(fwd_h - target) > 180) or (bwd_present and abs(bwd_h - target) > 180):
        reward -= 20.0
        
    return reward

# --- Policy Network Definitions (Copied from collect_data.py for independence) ---
class EmbeddingLayer(nn.Module):
    def __init__(self, cat_dims, embedding_dims=None, layer_norm=False, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.cat_dims = cat_dims
        self.embedding_dims = []
        
        # Keys matching the saved model
        self.cat_names = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
        
        # Hardcoded dims matching the saved model inspection
        # line: 4, bus: 21, station: 1, time: 1, dir: 2
        manual_embed_dims = [4, 21, 1, 1, 2]
        
        self.embeddings = nn.ModuleDict()
        
        for i, (name, dim) in enumerate(zip(self.cat_names, manual_embed_dims)):
            self.embedding_dims.append(dim)
            self.embeddings[name] = nn.Embedding(cat_dims[i], dim)
            
        self.output_dim = sum(self.embedding_dims) 
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        embedded = []
        # MUST iterate in the order of valid input columns
        # We know x is [line, bus, station, time, dir]
        for i, name in enumerate(self.cat_names):
            val = x[:, i].long()
            val = torch.clamp(val, 0, self.cat_dims[i]-1)
            emb = self.embeddings[name]
            embedded.append(emb(val))
            
        out = torch.cat(embedded, dim=1)
        if self.layer_norm:
            out = self.layer_norm(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, action_range=1.):
        super(PolicyNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size) 
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.action_range = action_range

    def forward(self, state):
        num_cats = len(self.embedding_layer.cat_dims)
        cat_tensor = state[:, :num_cats]
        num_tensor = state[:, num_cats:]
        
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1) 
        
        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        
        mean = self.mean_linear(x)
        return mean, None # LogStd not needed for deterministic eval

    def get_action(self, state, deterministic=True):
        mean, _ = self.forward(state)
        return torch.tanh(mean).detach().cpu().numpy()

def collect_worker(worker_id, num_episodes, args, output_file):
    print(f"Worker {worker_id}: Starting collection of {num_episodes} episodes...")
    
    import time
    import random
    
    # Stagger start slightly to avoid disk I/O spikes
    time.sleep(1.0 + worker_id * 0.1) 

    # Initialize Environment inside worker
    # Note: GUI must be False for parallel workers usually, or only 1 worker if GUI
    env = SumoBusHoldingEnv(gui=False, max_steps=args.max_steps, seed='random')
    
    # Load Expert Model if needed
    expert_net = None
    if args.policy == 'expert':
        cat_dims = [12, 389, 1, 1, 2]
        embedding = EmbeddingLayer(cat_dims, layer_norm=True, dropout=0.0)
        input_dim = 29 + 6 # 35 inputs for Best Result model
        expert_net = PolicyNetwork(input_dim, 1, 32, embedding, action_range=1.0)
        try:
            state_dict = torch.load(args.model_path, map_location='cpu')
            expert_net.load_state_dict(state_dict)
            expert_net.eval()
        except Exception as e:
            print(f"Worker {worker_id}: Failed to load model: {e}")
            sys.exit(1)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    # Pending Cache: {bus_id: {'obs': ..., 'action': ...}}
    pending_transitions = {}
    
    for eps in range(num_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            # 1. Identify valid bus ID from last event
            try:
                current_bus_id = env.last_event.bus_id
            except AttributeError:
                # Should not happen in normal flow unless immediate done
                current_bus_id = "unknown"
                
            # 2. Check pending
            if current_bus_id in pending_transitions:
                prev_data = pending_transitions.pop(current_bus_id)
                prev_obs = prev_data['obs']
                prev_action = prev_data['action']
                
                # Compute delayed reward (Headway based)
                # Using local helper or simplified relative logic. 
                # Since calculate_reward is not imported, let's use the env's internal reward 
                # OR reimplement it. 
                # Actually, simpler: Use the reward returned by step() BUT associate it with prev_obs?
                # No, env.step() returns reward for the PREVIOUS event.
                # If step() returns reward for THIS current_bus_id's PREVIOUS action?
                prev_info = prev_data.get('info', {})
                
                # Compute reward driven by the *arrival* at this new station
                # Use presence flags from info for accurate ladder reward
                r = calculate_reward(prev_obs, obs, prev_action,
                                     fwd_present=prev_info.get('forward_bus_present', True),
                                     bwd_present=prev_info.get('backward_bus_present', True))
                
                # Store
                observations.append(prev_obs)
                actions.append(prev_action)
                rewards.append(r)
                next_observations.append(obs)
                terminals.append(False)
            
            # Policy Logic
            if args.policy == 'random':
                action = np.random.uniform(0, 60, size=(1,))
            elif args.policy == 'zero':
                action = np.zeros((1,), dtype=np.float32)
            elif args.policy == 'mixed':
                if np.random.rand() < 0.8:
                    action = np.zeros((1,), dtype=np.float32)
                else:
                    action = np.random.uniform(0, 60, size=(1,))
            elif args.policy == 'expert':
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                a_t = expert_net.get_action(obs_tensor)[0]
                action = (a_t + 1) / 2 * 60.0 
            else:
                action = np.zeros((1,), dtype=np.float32)

            # Store current into pending before stepping
            pending_transitions[current_bus_id] = {
                'obs': np.array(obs, copy=True),
                'action': np.array(action, copy=True),
                'info': info if 'info' in locals() else {}
            }

            # Step
            obs, reward, done, info = env.step(action)
            
            # Note: The 'reward' returned here corresponds to the bus that just arrived (env.last_event).
            # Which IS 'current_bus_id' (because step() processes the event that was waiting).
            # Wait. 
            # env.step(action) applies action to 'last_event' (Bus A).
            # Then it advances until NEW event (Bus B).
            # Then it computes reward for Bus B's arrival?
            # NO. sumo_env.py:
            # 1. apply_action(last_event) -> Bus A holds.
            # 2. _compute_reward(last_event) -> Reward for Bus A (Immediate deviation).
            # 3. _advance() -> Next Event (Bus B).
            # So 'reward' returned is for Bus A.
            # So `reward` matches `current_bus_id`.
            # BUT, we only want to store it when we have the NEXT State (Bus A @ Next Stop).
            # So we perform the cache storage logic AFTER Step?
            # NO.
            # The 'reward' we get from step() is Immediate Reward (State A deviation).
            # `calculate_reward` in collect_data.py re-calculates it.
            # If we trust `env.step` reward, we can cache it too?
            # 
            # Let's stick to `collect_data.py` logic which re-calculates reward at Next Stop?
            # `collect_data.py`: `r = calculate_reward(prev, curr, act)`.
            # This calculates reward based on the NEW state (Arrival at next stop).
            # The `sumo_env.py` reward is based on the OLD state (Arrival at current stop).
            # WHICH ONE IS CORRECT?
            # Standard RL: R(s, a).
            # If reward is "Headway Deviation", it is usually penalized at Arrival.
            # If we hold at Station 1. Reward should be for arriving at Station 2?
            # RL Env (Ruiguang) computes reward at Arrival.
            # So R(Arrival) is the reward for the Trip leading to it.
            # So R is associated with the Previous Action.
            # So `calculate_reward(prev, curr)` is correct (Reward determined by Next State).
            # `sumo_env.py` returns reward for `last_event` (Current Arrival).
            # So `step()` returns reward for the state we just saw (Obs). 
            # This is R(s_prev, a_prev, s_curr).
            # So `reward` from step() IS the reward for the transition ending at `obs` (Bus A, current stop).
            # BUT `obs` is the START of the new transition `(Obs -> Action -> NextObs)`.
            # We want R for `(Obs -> Action -> NextObs)`.
            # That R will be revealed when we reach `NextObs`.
            # So we must wait for NEXT arrival to get the reward for THIS action.
            # CONCLUSION: `collect_data.py` logic is Correct: Compute R when NextObs arrives.
            # `sumo_env.py` logic is for On-Policy loops where we might just sum deviations.
            
            # I will paste `calculate_reward` function and use it.
            
            steps += 1
        
        if (eps + 1) % args.save_interval == 0:
            chunk_file = output_file.replace(".hdf5", f"_chunk_{eps//args.save_interval}.hdf5")
            print(f"Worker {worker_id}: Saving chunk {eps//args.save_interval} to {chunk_file}")
            
            # Policy Type array
            policy_types = np.full((len(observations), 1), args.policy_id, dtype=np.float32)
            
            with h5py.File(chunk_file, 'w') as f:
                f.create_dataset('observations', data=np.array(observations, dtype=np.float32))
                f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
                f.create_dataset('rewards', data=np.array(rewards, dtype=np.float32))
                f.create_dataset('next_observations', data=np.array(next_observations, dtype=np.float32))
                f.create_dataset('terminals', data=np.array(terminals, dtype=bool))
                f.create_dataset('policy_types', data=policy_types)
                
            # Clear buffers to free memory
            observations = []
            actions = []
            rewards = []
            next_observations = []
            terminals = []

        print(f"Worker {worker_id}: Finished Episode {eps+1}/{num_episodes}")

    env.close()
    
    # Save REMAINING data if any
    if len(observations) > 0:
        chunk_file = output_file.replace(".hdf5", f"_chunk_final.hdf5")
        print(f"Worker {worker_id}: Saving final chunk to {chunk_file}")
        
        policy_types = np.full((len(observations), 1), args.policy_id, dtype=np.float32)
        
        with h5py.File(chunk_file, 'w') as f:
            f.create_dataset('observations', data=np.array(observations, dtype=np.float32))
            f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
            f.create_dataset('rewards', data=np.array(rewards, dtype=np.float32))
            f.create_dataset('next_observations', data=np.array(next_observations, dtype=np.float32))
            f.create_dataset('terminals', data=np.array(terminals, dtype=bool))
            f.create_dataset('policy_types', data=policy_types)

    print(f"Worker {worker_id}: Done.")
    return True

def merge_datasets(output_path, part_files):
    print("Merging datasets...")
    
    total_obs = []
    total_actions = []
    total_rewards = []
    total_next_obs = []
    total_terminals = []
    total_policy_types = []
    
    for pf in part_files:
        with h5py.File(pf, 'r') as f:
            total_obs.append(f['observations'][:])
            total_actions.append(f['actions'][:])
            total_rewards.append(f['rewards'][:])
            total_next_obs.append(f['next_observations'][:])
            total_terminals.append(f['terminals'][:])
            
            if 'policy_types' in f:
                total_policy_types.append(f['policy_types'][:])
            else:
                # Fallback if mixing old data (0.0 default)
                total_policy_types.append(np.zeros((len(f['observations']), 1), dtype=np.float32))
            
    # Concatenate
    all_obs = np.concatenate(total_obs, axis=0)
    all_actions = np.concatenate(total_actions, axis=0)
    all_rewards = np.concatenate(total_rewards, axis=0)
    all_next_obs = np.concatenate(total_next_obs, axis=0)
    all_terminals = np.concatenate(total_terminals, axis=0)
    all_policy_types = np.concatenate(total_policy_types, axis=0)
    
    print(f"Total Transitions: {len(all_obs)}")
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('observations', data=all_obs)
        f.create_dataset('actions', data=all_actions)
        f.create_dataset('rewards', data=all_rewards)
        f.create_dataset('next_observations', data=all_next_obs)
        f.create_dataset('terminals', data=all_terminals)
        f.create_dataset('policy_types', data=all_policy_types)
        
    print(f"Merged dataset saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--max_steps", type=int, default=86400)
    parser.add_argument("--output", type=str, default="offline_sumo/data/buffer.hdf5")
    parser.add_argument("--policy", type=str, default="mixed", choices=["random", "zero", "mixed", "expert"])
    parser.add_argument("--model_path", type=str, default="logs/sac_v2_bus_SUMO_best_result/sac_v2_episode_18_policy")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--save_interval", type=int, default=20, help="Save to disk every N episodes per worker")
    parser.add_argument("--policy_id", type=float, default=0.0, help="Label for this data batch (0=Zero, 1=Expert)")
    args = parser.parse_args()
    
    # Calculate episodes per worker
    episodes_per_worker = args.episodes // args.num_workers
    remainder = args.episodes % args.num_workers
    
    processes = []
    part_files = []
    
    # Ensure output dir
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    # Spawn workers
    for i in range(args.num_workers):
        count = episodes_per_worker + (1 if i < remainder else 0)
        if count == 0: continue
        
        part_file = args.output.replace(".hdf5", f"_part_{i}.hdf5")
        # part_files should be all eventual chunks? 
        # No, merge_datasets takes a list.
        # But workers now generate MULTIPLE files per part.
        # MAIN PROCESS needs to know to scan for them.
        # Let's just track the BASE part name, and use glob in merge?
        part_files.append(part_file)
        
        p = multiprocessing.Process(target=collect_worker, args=(i, count, args, part_file))
        p.start()
        processes.append(p)
        
    # Wait for workers
    for p in processes:
        p.join()
        
    # Check if all successful
    # Scan for ALL chunk files
    base_pattern = args.output.replace(".hdf5", "_part_*_chunk_*.hdf5")
    found_chunks = glob.glob(base_pattern)
    
    if found_chunks:
        merge_datasets(args.output, found_chunks)
        
        # Cleanup
        for f in found_chunks:
            os.remove(f)
    else:
        print("No data collected.")

if __name__ == "__main__":
    # Ensure start method is spawn or forkserver to be safe with Libsumo (C++ extension)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
