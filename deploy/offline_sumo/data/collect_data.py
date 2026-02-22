import os
import sys
import numpy as np
import h5py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Ensure SUMO_HOME tools are in path BEFORE importing anything that uses traci
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

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

# --- Policy Network for Expert Loading ---
class EmbeddingLayer(nn.Module):
    def __init__(self, cat_dims, embedding_dims=None, layer_norm=False, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.cat_dims = cat_dims 
        self.embedding_dims = []
        
        # Keys matching the saved model
        self.cat_names = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']

        # Hardcoded dims matching the saved model inspection
        # line: 4, bus: 21, station: 1, time: 1, dir: 2
        # cat_dims: [12, 389, 1, 1, 2]
        manual_embed_dims = [4, 21, 1, 1, 2]
        
        self.embeddings = nn.ModuleDict()
        
        for i, (name, dim) in enumerate(zip(self.cat_names, manual_embed_dims)):
            self.embedding_dims.append(dim)
            self.embeddings[name] = nn.Embedding(cat_dims[i], dim)
            
        self.output_dim = sum(self.embedding_dims) # 29
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        embedded = []
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
        self.linear4 = nn.Linear(hidden_size, hidden_size) # 4th layer
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.action_range = action_range

    def forward(self, state):
        # Expect state: [cat...cat, cont...cont]
        num_cats = len(self.embedding_layer.cat_dims)
        cat_tensor = state[:, :num_cats]
        num_tensor = state[:, num_cats:]
        
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1) # 29 + 5 = 34
        
        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def get_action(self, state, deterministic=True):
        mean, _ = self.forward(state)
        return torch.tanh(mean).detach().cpu().numpy()

def collect_data(args):
    # Setup paths
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    
    # Load Expert Model if needed
    expert_net = None
    if args.policy == 'expert':
        print(f"Loading Expert Policy from {args.model_path}")
        # Hardcoded dims from inspection
        cat_dims = [12, 389, 1, 1, 2]
        embedding = EmbeddingLayer(cat_dims, layer_norm=True, dropout=0.0) # Matches 'layer_norm: True' in weights
        # Input to linear1 is 35 (Confirmed for Best Result Ep 18). Embed=29. So Num Cont = 6.
        # Original obs has 11. Cats=5. Conts=6.
        input_dim = 29 + 6 
        expert_net = PolicyNetwork(input_dim, 1, 32, embedding, action_range=1.0)
        
        try:
            state_dict = torch.load(args.model_path, map_location='cpu')
            expert_net.load_state_dict(state_dict)
            expert_net.eval()
            print("Expert Model Loaded Successfully!")
        except Exception as e:
            print(f"Failed to load expert model: {e}")
            sys.exit(1)

    # Initialize Env
    env = SumoBusHoldingEnv(gui=args.gui, max_steps=args.max_steps)

    
    # Dataset buffers
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    # Pending Cache: {bus_id: {'obs': ..., 'action': ...}}
    pending_transitions = {}
    
    obs = env.reset()
    total_steps = 0
    
    pbar = tqdm(total=args.episodes)
    
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        
        while not done:
            # 1. Identify which bus involves this observation
            # Obs[1] is bus_idx. Note: Env maps bus_id string to float idx.
            # We need the bus ID string to robustly track distinct vehicles.
            # env.last_event.bus_id is reliable.
            current_bus_id = env.last_event.bus_id
            # 2. Check if we have a pending transition for THIS bus
            if current_bus_id in pending_transitions:
                prev_data = pending_transitions.pop(current_bus_id)
                prev_obs = prev_data['obs']
                prev_action = prev_data['action']
                prev_info = prev_data.get('info', {})
                
                # The reward for the arrival at the NEW stop was calculated 
                # inside the environment's _compute_reward and returned by env.step (see below)
                # But for consistency in this loop, we retrieve it from the info/reward of the transition
                # Actually, let's use the calculate_reward with flags for explicit logging
                r = calculate_reward(prev_obs, obs, prev_action, 
                                     fwd_present=prev_info.get('forward_bus_present', True),
                                     bwd_present=prev_info.get('backward_bus_present', True))
                
                # Store transition
                observations.append(prev_obs)
                actions.append(prev_action)
                rewards.append(r)
                next_observations.append(obs)
                terminals.append(False)
                
                # VERIFICATION LOG
                if total_steps % 10 == 0:
                    print(f"VERIFY_LINK: Bus {current_bus_id} | Stop {prev_obs[2]} -> {obs[2]} | Act {prev_action[0]:.2f} | Rew {r:.2f} | FwdPres {info.get('forward_bus_present')} | BwdPres {info.get('backward_bus_present')}")
                
                total_steps += 1
            
            # 3. Select Action (Random or Rule-based)
            # Simple rule: Hold to match target headway?
            # Or simplified: Random for exploration
            # 3. Select Action
            if args.policy == 'random':
                action = np.random.uniform(0, 60, size=(1,))
            elif args.policy == 'zero':
                action = np.zeros((1,), dtype=np.float32)
            elif args.policy == 'mixed':
                # 80% Zero (Natural), 20% Random (Exploration)
                if np.random.rand() < 0.8:
                    action = np.zeros((1,), dtype=np.float32)
                else:
                    action = np.random.uniform(0, 60, size=(1,))
            elif args.policy == 'expert':
                # Prepare Obs: keep all features for Best Result model (Input=35)
                # Obs: [line, bus, station, time, dir, fwd, bwd, wait, target, dur, sim]
                # Expert expects 5 cats + 6 conts = 35 inputs.
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Get Action (-1 to 1)
                a_t = expert_net.get_action(obs_tensor) # shape (1, 1)
                a_t = a_t[0]
                
                # Map to 0-60
                action = (a_t + 1) / 2 * 60.0
            else:
                # Simple rule: if forward headway < target, hold diff
                fwd_h = obs[5]
                target = obs[8]
                hold = max(0, target - fwd_h)
                action = np.array([hold])
            
            # 4. Cache current decision
            # We also store the 'info' from the step that led to this arrival
            # to know if neighbors were present at THIS arrival event.
            pending_transitions[current_bus_id] = {
                'obs': obs,
                'action': action,
                'info': info if 'info' in locals() else {}
            }
            
            # 5. Step
            obs, reward, done, info = env.step(action)
            
        pbar.update(1)
        
    env.close()
    
    # Save to HDF5
    print(f"Collected {len(observations)} transitions.")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('observations', data=np.array(observations, dtype=np.float32))
        f.create_dataset('actions', data=np.array(actions, dtype=np.float32))
        f.create_dataset('rewards', data=np.array(rewards, dtype=np.float32))
        f.create_dataset('next_observations', data=np.array(next_observations, dtype=np.float32))
        f.create_dataset('terminals', data=np.array(terminals, dtype=bool))
    
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=86400)
    parser.add_argument("--output", type=str, default="offline_sumo/data/dataset.hdf5")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "rule", "zero", "mixed", "expert"])
    parser.add_argument("--model_path", type=str, default="logs/sac_v2_bus_SUMO_best_result/sac_v2_episode_18_policy")
    args = parser.parse_args()
    collect_data(args)
