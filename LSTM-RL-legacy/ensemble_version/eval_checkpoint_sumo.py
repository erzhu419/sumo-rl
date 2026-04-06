"""
eval_checkpoint_sumo.py
=======================
Evaluate a trained SAC policy checkpoint on the SumoBusHoldingEnv
(same env as sac_ensemble_SUMO_linear_penalty.py).

Loads the checkpoint_episode_39 policy from H2Oplus/collect_policy/ and
runs 1 episode (or more) through the SUMO env with linear_penalty reward,
then reports cumulative reward.

Usage:
    cd /home/erzhu419/mine_code/sumo-rl
    python LSTM-RL-legacy/ensemble_version/eval_checkpoint_sumo.py
"""

import os
import sys
import time
import copy
import importlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import defaultdict

# ── Path setup ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# normalization from collect_policy dir (same module used in training)
COLLECT_DIR = os.path.join(PROJECT_ROOT, 'H2Oplus', 'collect_policy')
sys.path.insert(0, COLLECT_DIR)
from normalization import Normalization, RunningMeanStd

device = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# Network Architecture (matches training checkpoint exactly)
# ══════════════════════════════════════════════════════════════════════

class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols, layer_norm=False, dropout=0.0):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)
        self.cardinalities = {}
        modules = {}
        total_dim = 0
        for col in self.cat_cols:
            codes = list(cat_code_dict[col].values())
            card = max(codes) + 1
            self.cardinalities[col] = card
            dim = min(32, max(2, int(round(card ** 0.5)) + 1)) if card > 1 else 1
            modules[col] = nn.Embedding(card, dim)
            total_dim += dim
        self.embeddings = nn.ModuleDict(modules)
        self.output_dim = total_dim
        self.layer_norm = nn.LayerNorm(total_dim) if layer_norm and total_dim > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, cat_tensor):
        if cat_tensor.dim() == 1:
            cat_tensor = cat_tensor.unsqueeze(0)
        parts = []
        for idx, col in enumerate(self.cat_cols):
            indices = torch.clamp(cat_tensor[:, idx].long(), 0, self.cardinalities[col] - 1)
            parts.append(self.embeddings[col](indices))
        embed = torch.cat(parts, dim=1) if parts else torch.empty(cat_tensor.size(0), 0)
        if self.layer_norm:
            embed = self.layer_norm(embed)
        if self.dropout:
            embed = self.dropout(embed)
        return embed

    def clone(self):
        return copy.deepcopy(self)


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer,
                 action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]

        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)

        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state, deterministic=True):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.float().to(device)
        mean, log_std = self.forward(state)
        if deterministic:
            action_0 = torch.tanh(mean)
        else:
            std = log_std.exp()
            z = torch.randn_like(mean)
            action_0 = torch.tanh(mean + std * z)
        # Residual Control: scale=1.0, bias=0.0 → raw tanh output [-1, 1]
        return action_0.detach().cpu().numpy()[0]


# ══════════════════════════════════════════════════════════════════════
# Load checkpoint
# ══════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_prefix):
    """Load checkpoint with architecture matching the training setup."""
    # 5 categorical columns (matches the SUMO env's get_feature_spec)
    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},
        'bus_id':      {i: i for i in range(389)},
        'station_id':  {0: 0},
        'time_period': {0: 0},
        'direction':   {0: 0, 1: 1},
    }
    num_cont_features = 10  # continuous obs features
    action_dim = 2          # 2D residual control

    emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
    state_dim = emb.output_dim + num_cont_features + action_dim  # 29 + 10 + 2 = 41
    hidden_dim = 48

    policy = PolicyNetwork(state_dim, action_dim, hidden_dim, emb.clone(), action_range=1.0).to(device)
    policy.load_state_dict(torch.load(checkpoint_prefix + "_policy", map_location=device, weights_only=True))
    policy.eval()

    # Load normalization
    norm_data = torch.load(checkpoint_prefix + "_norm", map_location=device, weights_only=False)
    num_cat = len(cat_cols)
    num_num = num_cont_features + action_dim
    running_ms = RunningMeanStd(shape=(num_num,))
    if isinstance(norm_data, dict):
        running_ms.mean = norm_data.get('mean', running_ms.mean)
    elif hasattr(norm_data, 'running_ms'):
        running_ms = norm_data.running_ms
    state_norm = Normalization(num_categorical=num_cat, num_numerical=num_num, running_ms=running_ms)

    print(f"  Policy loaded: input_dim={state_dim}, action_dim={action_dim}, hidden={hidden_dim}")
    print(f"  Embedding output dim: {emb.output_dim}")
    return policy, state_norm, action_dim


# ══════════════════════════════════════════════════════════════════════
# Env setup (reuses the same SUMO bridge as sac_ensemble)
# ══════════════════════════════════════════════════════════════════════

def build_env(sumo_root, schedule_rel='initialize_obj/save_obj_bus.add.xml',
              bridge_entry='SUMO_ruiguang.online_control.rl_bridge:build_bridge',
              traffic_scale=1.0, gui=False):
    """Build the same environment that sac_ensemble_SUMO_linear_penalty.py uses."""
    module_name, _, attr_name = bridge_entry.partition(':')
    bridge_module = importlib.import_module(module_name)
    factory = getattr(bridge_module, attr_name or 'build_bridge')
    bridge = factory(root_dir=sumo_root, gui=gui, update_freq=1, scale=traffic_scale)

    if isinstance(bridge, tuple):
        decision_provider = bridge[0]
        action_executor = bridge[1]
        reset_cb = bridge[2] if len(bridge) > 2 else None
        close_cb = bridge[3] if len(bridge) > 3 else None
    elif isinstance(bridge, dict):
        decision_provider = bridge.get('decision_provider')
        action_executor = bridge.get('action_executor')
        reset_cb = bridge.get('reset_callback')
        close_cb = bridge.get('close_callback')
    else:
        raise ValueError("Bridge returns unsupported format")

    from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv
    env = SumoBusHoldingEnv(
        root_dir=sumo_root,
        schedule_file=schedule_rel,
        decision_provider=decision_provider,
        action_executor=action_executor,
        reset_callback=reset_cb,
        close_callback=close_cb,
        debug=False,
        reward_type="linear_penalty",
    )
    return env


def build_action_template(state_dict, previous=None):
    """Same helper as in sac_ensemble."""
    if previous is None:
        template = {}
    else:
        template = previous
    for line_id, buses in state_dict.items():
        if line_id not in template:
            template[line_id] = {}
        for bus_id in buses.keys():
            if bus_id not in template[line_id] or template[line_id][bus_id] is None:
                template[line_id][bus_id] = None
    return template


def get_reward_value(reward_dict, line_id, bus_id):
    return reward_dict.get(line_id, {}).get(bus_id, 0.0)


# ══════════════════════════════════════════════════════════════════════
# Run eval episode
# ══════════════════════════════════════════════════════════════════════

def run_eval_episode(env, policy_net, state_norm, action_dim, deterministic=True):
    """Run one episode using the env's step() interface (same as training loop)."""
    env.reset()
    state_dict, reward_dict, done = env.initialize_state()

    action_dict = build_action_template(state_dict)
    last_action_history = defaultdict(lambda: defaultdict(lambda: np.zeros(action_dim, dtype=np.float32)))

    episode_reward = 0.0
    episode_steps = 0
    station_feature_idx = 2  # station_id is at index 2 in the 15-dim obs

    while not done:
        action_dict = build_action_template(state_dict, action_dict)

        for line_id, buses in state_dict.items():
            for bus_id, history in buses.items():
                if len(history) == 0:
                    continue

                if len(history) == 1:
                    if action_dict[line_id][bus_id] is None:
                        last_action = last_action_history[line_id][bus_id]
                        state_vec = np.concatenate([history[0], last_action])
                        # Apply normalization (update=False for eval)
                        state_vec = state_norm(state_vec, update=False)
                        action = policy_net.get_action(
                            torch.from_numpy(state_vec).float(),
                            deterministic=deterministic
                        )
                        action_dict[line_id][bus_id] = action

                elif len(history) >= 2:
                    prev_action = last_action_history[line_id][bus_id]
                    current_action = action_dict[line_id][bus_id]
                    if current_action is None:
                        current_action = np.zeros(action_dim, dtype=np.float32)
                    elif np.isscalar(current_action):
                        current_action = np.array([current_action], dtype=np.float32)
                    else:
                        current_action = np.array(current_action, dtype=np.float32).reshape(-1)

                    if history[0][station_feature_idx] != history[1][station_feature_idx]:
                        current_reward = get_reward_value(reward_dict, line_id, bus_id)
                        episode_steps += 1
                        episode_reward += current_reward

                    state_dict[line_id][bus_id] = history[1:]
                    last_action_history[line_id][bus_id] = current_action

                    state_vec_next = np.concatenate([state_dict[line_id][bus_id][0], current_action])
                    state_vec_next = state_norm(state_vec_next, update=False)
                    action = policy_net.get_action(
                        torch.from_numpy(state_vec_next).float(),
                        deterministic=deterministic
                    )
                    action_dict[line_id][bus_id] = action

        # Map raw tanh actions to env actions (Residual Control + Bang-Bang)
        env_action_dict = copy.deepcopy(action_dict)
        for line_id, buses in env_action_dict.items():
            for bus_id, action_val in buses.items():
                if action_val is not None:
                    a_hold = action_val[0]
                    a_speed = action_val[1]

                    # Holding: Continuous mapping (0 to 120 seconds)
                    hold = np.clip((a_hold + 1.0) * 60.0, 0.0, 120.0)

                    # Speed: Deterministic Bang-Bang (eval mode)
                    if a_speed > 0.6:
                        speed = 1.2
                    elif a_speed > 0.2:
                        speed = 1.1
                    elif a_speed > -0.2:
                        speed = 1.0
                    elif a_speed > -0.6:
                        speed = 0.9
                    else:
                        speed = 0.8

                    env_action_dict[line_id][bus_id] = [hold, speed]

        state_dict, reward_dict, done, _ = env.step(env_action_dict)

        if done:
            break

    return episode_reward, episode_steps


def run_baseline_episode(env, action_dim):
    """Run one episode with zero-hold baseline for comparison."""
    env.reset()
    state_dict, reward_dict, done = env.initialize_state()

    action_dict = build_action_template(state_dict)
    station_feature_idx = 2

    episode_reward = 0.0
    episode_steps = 0

    while not done:
        action_dict = build_action_template(state_dict, action_dict)

        for line_id, buses in state_dict.items():
            for bus_id, history in buses.items():
                if len(history) == 0:
                    continue
                if len(history) == 1:
                    if action_dict[line_id][bus_id] is None:
                        action_dict[line_id][bus_id] = np.zeros(action_dim, dtype=np.float32)
                elif len(history) >= 2:
                    if history[0][station_feature_idx] != history[1][station_feature_idx]:
                        current_reward = get_reward_value(reward_dict, line_id, bus_id)
                        episode_steps += 1
                        episode_reward += current_reward
                    state_dict[line_id][bus_id] = history[1:]
                    action_dict[line_id][bus_id] = np.zeros(action_dim, dtype=np.float32)

        # Baseline: zero hold, normal speed
        env_action_dict = copy.deepcopy(action_dict)
        for line_id, buses in env_action_dict.items():
            for bus_id, action_val in buses.items():
                if action_val is not None:
                    env_action_dict[line_id][bus_id] = [0.0, 1.0]

        state_dict, reward_dict, done, _ = env.step(env_action_dict)
        if done:
            break

    return episode_reward, episode_steps


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate SAC checkpoint on SUMO env (linear_penalty)')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(COLLECT_DIR, 'checkpoint_episode_39'),
                        help='Checkpoint prefix path')
    parser.add_argument('--sumo_root', type=str,
                        default=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'),
                        help='SUMO root dir')
    parser.add_argument('--traffic_scale', type=float, default=1.0)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--no_baseline', action='store_true', help='Skip baseline eval')
    parser_args = parser.parse_args()

    print("=" * 70)
    print("Eval: checkpoint_episode_39 on SumoBusHoldingEnv (linear_penalty)")
    print("=" * 70)

    # Load checkpoint
    print("\n[1/3] Loading checkpoint...")
    policy, state_norm, action_dim = load_checkpoint(parser_args.checkpoint)

    # Build env
    print("\n[2/3] Building SUMO env...")
    env = build_env(parser_args.sumo_root, traffic_scale=parser_args.traffic_scale, gui=parser_args.gui)
    print("  Env built successfully.")

    # Run SAC policy
    print("\n[3/3] Running SAC policy episode...")
    t0 = time.time()
    sac_reward, sac_steps = run_eval_episode(env, policy, state_norm, action_dim, deterministic=True)
    sac_time = time.time() - t0
    print(f"  SAC Policy:  Reward = {sac_reward:,.1f}  |  Steps = {sac_steps}  |  Time = {sac_time:.1f}s")

    if not parser_args.no_baseline:
        print("\n  Running zero-hold baseline...")
        t0 = time.time()
        base_reward, base_steps = run_baseline_episode(env, action_dim)
        base_time = time.time() - t0
        print(f"  Baseline:    Reward = {base_reward:,.1f}  |  Steps = {base_steps}  |  Time = {base_time:.1f}s")

        print("\n" + "=" * 70)
        print("SUMMARY (SumoBusHoldingEnv, linear_penalty)")
        print("=" * 70)
        print(f"  SAC Policy:   {sac_reward:>14,.1f}  ({sac_steps} steps, {sac_time:.1f}s)")
        print(f"  Zero-Hold:    {base_reward:>14,.1f}  ({base_steps} steps, {base_time:.1f}s)")
        if base_reward != 0:
            improvement = (sac_reward - base_reward) / abs(base_reward) * 100
            print(f"  Improvement:  {improvement:>+13.1f}%")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"  SAC Policy:   {sac_reward:>14,.1f}  ({sac_steps} steps, {sac_time:.1f}s)")
        print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
