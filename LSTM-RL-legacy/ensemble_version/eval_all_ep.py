"""
Evaluate ALL checkpoints from the original V7_Long training run.
Reconstructs the deterministic evaluation reward curve by loading each
checkpoint_episode_N and running one full evaluation episode.

Usage:
    conda run -n LSTM-RL python eval_all_ep.py [--start_ep 0] [--end_ep 149] [--sumo_gui]
"""
import psutil, tracemalloc
import gym
import copy
import importlib
import sys
from typing import Any, Dict, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import math
import gc
import time
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from normalization import Normalization, RewardScaling, RunningMeanStd

from IPython.display import clear_output
import matplotlib.pyplot as plt
from env.sim import env_bus
import argparse
import numpy as np
import random

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import traci

from copy import deepcopy
from collections import defaultdict


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Evaluate all checkpoints from original training.')
parser.add_argument('--use_gradient_clip', type=bool, default=True)
parser.add_argument("--use_state_norm", type=bool, default=False)
parser.add_argument("--use_reward_norm", type=bool, default=False)
parser.add_argument("--use_reward_scaling", type=bool, default=False)
parser.add_argument("--gamma", type=float, default=0.80)
parser.add_argument("--training_freq", type=int, default=10)
parser.add_argument("--plot_freq", type=int, default=1)
parser.add_argument('--weight_reg', type=float, default=0.01)
parser.add_argument('--auto_entropy', type=bool, default=True)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=10000)
parser.add_argument("--maximum_alpha", type=float, default=0.6)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--max_episodes", type=int, default=100)
parser.add_argument('--save_root', type=str, default=os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--run_name', type=str, default='eval_all_ep')
parser.add_argument('--render', action='store_true')
parser.add_argument('--use_sumo_env', action='store_true')
parser.add_argument('--no_sumo_env', action='store_true')
parser.add_argument('--sumo_root', type=str, default=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'))
parser.add_argument('--sumo_schedule', type=str, default='initialize_obj/save_obj_bus.add.xml')
parser.add_argument('--sumo_bridge', type=str, default='SUMO_ruiguang.online_control.rl_bridge:build_bridge')
parser.add_argument('--sumo_gui', action='store_true')
parser.add_argument('--update_passenger_freq', type=int, default=1)
parser.add_argument('--profile', action='store_true')

# Ensemble args
parser.add_argument("--ensemble_size", type=int, default=2)
parser.add_argument("--beta_bc", type=float, default=0.001)
parser.add_argument("--beta", type=float, default=-2)
parser.add_argument("--beta_ood", type=float, default=0.01)
parser.add_argument('--critic_actor_ratio', type=int, default=2)
parser.add_argument('--use_residual_control', action='store_true')
parser.add_argument('--use_1d_mapping', action='store_true')
parser.add_argument('--holding_only', action='store_true')
parser.add_argument('--speed_only', action='store_true')
parser.add_argument('--bang_bang', action='store_true')
parser.add_argument('--traffic_scale', type=float, default=1.0)
parser.add_argument('--resume_checkpoint', type=str, default=None)

# Eval-specific args
parser.add_argument('--model_dir', type=str,
                    default='model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long',
                    help='Directory containing the checkpoints to evaluate')
parser.add_argument('--start_ep', type=int, default=0, help='First episode checkpoint to evaluate')
parser.add_argument('--end_ep', type=int, default=149, help='Last episode checkpoint to evaluate')
parser.add_argument('--output_file', type=str, default=None, help='Output .npy file for eval rewards')

args = parser.parse_args()

# Force eval mode
args.train = False
args.test = True
# Force residual control (matching the original V7_Long training)
args.use_residual_control = True

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_NAME = args.run_name.strip() if args.run_name else None
SAVE_ROOT = os.path.abspath(args.save_root)
EXPERIMENT_ID = f"{SCRIPT_NAME}_{RUN_NAME}" if RUN_NAME else SCRIPT_NAME

PIC_DIR = os.path.join(SAVE_ROOT, 'pic', EXPERIMENT_ID)
LOG_DIR = os.path.join(SAVE_ROOT, 'logs', EXPERIMENT_ID)
MODEL_DIR = os.path.join(SAVE_ROOT, 'model', EXPERIMENT_ID)

for directory in (PIC_DIR, LOG_DIR, MODEL_DIR):
    os.makedirs(directory, exist_ok=True)


# ============================================================================
# Model definitions (must match the training script exactly)
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity, last_episode_step=5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[int(self.position)] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), np.stack(actions), np.array(rewards, dtype=np.float32), \
               np.stack(next_states), np.array(dones, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols, embedding_dims=None, layer_norm=False, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = list(cat_cols)
        self.embedding_dims = {}
        self.cardinalities = {}
        modules = {}
        for col in self.cat_cols:
            codes = list(cat_code_dict[col].values())
            if len(codes) == 0:
                raise ValueError(f"Categorical column '{col}' has no encoding values defined.")
            cardinality = max(codes) + 1
            self.cardinalities[col] = cardinality
            dim = embedding_dims[col] if embedding_dims and col in embedding_dims else self._suggest_dim(cardinality)
            self.embedding_dims[col] = dim
            modules[col] = nn.Embedding(cardinality, dim)
        self.embeddings = nn.ModuleDict(modules)
        self.output_dim = sum(self.embedding_dims.values())
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm and self.output_dim > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def _suggest_dim(cardinality: int) -> int:
        if cardinality <= 1:
            return 1
        return min(32, max(2, int(round(cardinality ** 0.5)) + 1))

    def forward(self, cat_tensor):
        if cat_tensor.dim() == 1:
            cat_tensor = cat_tensor.unsqueeze(0)
        embedding_tensor_group = []
        for idx, col in enumerate(self.cat_cols):
            indices = cat_tensor[:, idx].long()
            max_index = self.cardinalities[col] - 1
            indices = torch.clamp(indices, 0, max_index)
            embedding_tensor_group.append(self.embeddings[col](indices))
        if embedding_tensor_group:
            embed_tensor = torch.cat(embedding_tensor_group, dim=1)
            if self.layer_norm is not None:
                embed_tensor = self.layer_norm(embed_tensor)
            if self.dropout is not None:
                embed_tensor = self.dropout(embed_tensor)
        else:
            embed_tensor = torch.empty(cat_tensor.size(0), 0, device=cat_tensor.device)
        return embed_tensor

    def clone(self):
        return copy.deepcopy(self)


class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight + self.bias


class VectorizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_critics, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class SoftQNetwork(VectorizedCritic):
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer, ensemble_size=10):
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
            num_critics=ensemble_size, embedding_layer=embedding_layer
        )
        self.ensemble_size = ensemble_size

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        return super().forward(state_with_embeddings, action)


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
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

    def get_action(self, state, deterministic):
        if getattr(state, 'dim', lambda: 0)() == 1:
            state = state.unsqueeze(0)
        state = state.float().to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)

        if mean.shape[-1] == 1:
            if args.speed_only:
                scale = torch.tensor([0.2], device=device)
                bias = torch.tensor([1.0], device=device)
            elif args.use_1d_mapping or args.use_residual_control:
                scale = torch.tensor([1.0], device=device)
                bias = torch.tensor([0.0], device=device)
            else:
                scale = torch.tensor([30.0], device=device)
                bias = torch.tensor([30.0], device=device)
        else:
            if args.use_residual_control:
                scale = torch.tensor([1.0, 1.0], device=device)
                bias = torch.tensor([0.0, 0.0], device=device)
            else:
                scale = torch.tensor([30.0, 0.2], device=device)
                bias = torch.tensor([30.0, 1.0], device=device)

        if deterministic:
            action_0 = torch.tanh(mean)
        else:
            action_0 = torch.tanh(mean + std * z)

        action = scale * action_0 + bias
        return action.detach().cpu().numpy()[0]


class SAC_Trainer():
    def __init__(self, env, replay_buffer, hidden_dim, action_range, ensemble_size=10):
        if hasattr(env, 'get_feature_spec'):
            spec = env.get_feature_spec()
            cat_cols = spec['cat_cols']
            cat_code_dict = {col: {i: i for i in range(spec['cat_sizes'][col])} for col in cat_cols}
            num_cont_features = spec['num_cont_features']
        else:
            cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
            cat_code_dict = {
                'bus_id': {i: i for i in range(env.max_agent_num)},
                'station_id': {i: i for i in range(max(round(len(env.stations) / 2), 1))},
                'time_period': {i: i for i in range(int(env.timetables[-1].launch_time // 3600) + 2)},
                'direction': {0: 0, 1: 1}
            }
            num_cont_features = env.state_dim - len(cat_cols)

        if args.use_residual_control:
            log_scale_shift = 0.0
        elif action_dim == 1:
            if args.speed_only:
                log_scale_shift = np.log(0.2)
            else:
                log_scale_shift = np.log(30.0)
        else:
            log_scale_shift = np.log(30.0) + np.log(0.2)

        self.target_entropy = -float(action_dim) + log_scale_shift
        self.cat_cols = cat_cols
        self.num_cat_features = len(cat_cols)
        self.num_cont_features = num_cont_features + action_dim
        self.station_feature_idx = cat_cols.index('station_id') if 'station_id' in cat_cols else None

        embedding_template = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
        state_dim_net = embedding_template.output_dim + self.num_cont_features

        self.replay_buffer = replay_buffer
        self.soft_q_net = SoftQNetwork(state_dim_net, action_dim, hidden_dim, embedding_template.clone(), ensemble_size=ensemble_size).to(device)
        self.target_soft_q_net = deepcopy(self.soft_q_net)
        self.policy_net = PolicyNetwork(state_dim_net, action_dim, hidden_dim, embedding_template.clone(), action_range).to(device)

        init_alpha = 0.1 if args.auto_entropy else args.alpha
        self.alpha = init_alpha
        self.log_alpha = torch.tensor([np.log(init_alpha)], dtype=torch.float32, requires_grad=True, device=device)

        self.soft_q_criterion = nn.MSELoss()
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        initial_mean = np.zeros(self.num_cont_features)
        initial_std = np.ones(self.num_cont_features)
        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=initial_mean, init_std=initial_std)
        self.state_norm = Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)
        self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    def save_model(self, path):
        torch.save(self.soft_q_net.state_dict(), path + '_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch.save(self.state_norm, path + '_norm')

    def load_model(self, path):
        self.soft_q_net.load_state_dict(torch.load(path + '_q', map_location=device, weights_only=False))
        self.policy_net.load_state_dict(torch.load(path + '_policy', map_location=device, weights_only=False))
        self.soft_q_net.eval()
        self.policy_net.eval()


# ============================================================================
# Helper functions
# ============================================================================
def safe_initialize_state(env, render=False):
    try:
        return env.initialize_state(render=render)
    except TypeError:
        return env.initialize_state()


def safe_step(env, action_dict, render=False):
    try:
        return env.step(action_dict)
    except TypeError:
        return env.step(action_dict)


def build_action_template(state_dict, previous=None):
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


# ============================================================================
# Environment setup
# ============================================================================
replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)

debug = False
render = bool(getattr(args, 'render', False))
if getattr(args, 'sumo_gui', False):
    render = True
use_sumo_env = True

if use_sumo_env:
    module_name, _, attr_name = args.sumo_bridge.partition(':')
    bridge_module = importlib.import_module(module_name)
    factory = getattr(bridge_module, attr_name or 'build_bridge')
    bridge = factory(
        root_dir=args.sumo_root,
        gui=getattr(args, 'sumo_gui', False) or render,
        update_freq=args.update_passenger_freq,
        scale=args.traffic_scale
    )
    if isinstance(bridge, dict):
        decision_provider = bridge.get('decision_provider')
        action_executor = bridge.get('action_executor')
        reset_cb = bridge.get('reset_callback')
        close_cb = bridge.get('close_callback')
    elif isinstance(bridge, tuple):
        decision_provider = bridge[0]
        action_executor = bridge[1]
        reset_cb = bridge[2] if len(bridge) > 2 else None
        close_cb = bridge[3] if len(bridge) > 3 else None
    else:
        raise ValueError("Bridge returns unsupported format")

    from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv
    env = SumoBusHoldingEnv(
        root_dir=args.sumo_root,
        schedule_file=args.sumo_schedule,
        decision_provider=decision_provider,
        action_executor=action_executor,
        reset_callback=reset_cb,
        close_callback=close_cb,
        debug=debug,
        reward_type="linear_penalty",
    )

# action_dim for residual control
if args.use_1d_mapping:
    action_dim = 1
elif args.use_residual_control:
    action_dim = 2
else:
    action_dim = 1 if (args.holding_only or args.speed_only) else 2
action_range = 1.0

hidden_dim = 48
DETERMINISTIC = True  # Deterministic evaluation: user confirmed -0.65M was achieved with deterministic=True

tracemalloc.start()
sac_trainer = SAC_Trainer(env, replay_buffer, hidden_dim=hidden_dim, action_range=action_range, ensemble_size=args.ensemble_size)

# ============================================================================
# Main evaluation loop
# ============================================================================
if __name__ == '__main__':
    model_dir = args.model_dir
    start_ep = args.start_ep
    end_ep = args.end_ep

    # Discover available checkpoints
    available_eps = []
    for ep in range(start_ep, end_ep + 1):
        ckpt = os.path.join(model_dir, f"checkpoint_episode_{ep}")
        if os.path.exists(ckpt + "_policy"):
            available_eps.append(ep)

    print(f"Found {len(available_eps)} checkpoints in {model_dir}")
    print(f"Evaluating episodes: {available_eps[0]} to {available_eps[-1]}")

    eval_rewards = []
    step = 0  # Global step counter for action mapping warmup logic

    for idx, ep in enumerate(available_eps):
        ckpt = os.path.join(model_dir, f"checkpoint_episode_{ep}")
        print(f"\n[{idx+1}/{len(available_eps)}] Loading checkpoint episode {ep}...")
        sac_trainer.load_model(ckpt)

        # Reset environment
        env.reset()
        state_dict, reward_dict, _ = safe_initialize_state(env, render=render)

        done = False
        episode_steps = 0
        action_dict = build_action_template(state_dict)
        last_action_history = defaultdict(lambda: defaultdict(lambda: np.zeros(action_dim, dtype=np.float32)))

        episode_reward = 0
        station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1
        episode_start_time = time.time()

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

                            if args.use_state_norm:
                                state_vec = sac_trainer.state_norm(state_vec)
                            action = sac_trainer.policy_net.get_action(
                                torch.from_numpy(state_vec).float(), deterministic=DETERMINISTIC)
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

                        state_vec = np.concatenate([history[0], prev_action])
                        next_state_vec = np.concatenate([history[1], current_action])

                        if history[0][station_feature_idx] != history[1][station_feature_idx]:
                            current_reward = get_reward_value(reward_dict, line_id, bus_id)
                            episode_steps += 1
                            episode_reward += current_reward

                        state_dict[line_id][bus_id] = history[1:]
                        state_vec_next = np.concatenate([state_dict[line_id][bus_id][0], current_action])

                        last_action_history[line_id][bus_id] = current_action
                        if args.use_state_norm:
                            state_vec_next = sac_trainer.state_norm(state_vec_next)
                        action = sac_trainer.policy_net.get_action(
                            torch.from_numpy(state_vec_next).float(), deterministic=DETERMINISTIC)
                        action_dict[line_id][bus_id] = action

            # Map policy actions to environment actions (deterministic, no warmup override)
            env_action_dict = copy.deepcopy(action_dict)
            for line_id, buses in env_action_dict.items():
                for bus_id, action_val in buses.items():
                    if action_val is not None:
                        if args.use_residual_control:
                            a_hold = action_val[0]
                            a_speed = action_val[1]

                            # Holding: Continuous Mapping (0 to 120 seconds)
                            hold = np.clip((a_hold + 1.0) * 60.0, 0.0, 120.0)

                            # Speed: Deterministic selection for evaluation
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
                        elif args.use_1d_mapping:
                            a = action_val[0]
                            if a > 0:
                                env_action_dict[line_id][bus_id] = [a * 60.0, 1.0]
                            elif a < 0:
                                env_action_dict[line_id][bus_id] = [0.0, 1.0 + abs(a) * 0.2]
                            else:
                                env_action_dict[line_id][bus_id] = [0.0, 1.0]
                        elif args.holding_only:
                            env_action_dict[line_id][bus_id] = [action_val[0], 1.0]
                        elif args.speed_only:
                            env_action_dict[line_id][bus_id] = [0.0, action_val[0]]

            state_dict, reward_dict, done, _ = safe_step(env, env_action_dict, render=render)

            if done:
                break

        episode_duration = time.time() - episode_start_time
        eval_rewards.append(episode_reward)

        print(f"  Episode {ep}: Reward = {episode_reward:.0f} | "
              f"Steps = {episode_steps} | Duration = {episode_duration:.1f}s | "
              f"CPU Mem = {psutil.Process().memory_info().rss / 1024**2:.0f} MB")

        # Save incrementally
        output_file = args.output_file or os.path.join(LOG_DIR, "eval_all_rewards.npy")
        np.save(output_file, np.array(eval_rewards))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Final summary
    eval_rewards = np.array(eval_rewards)
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Checkpoints evaluated: {len(available_eps)}")
    print(f"Best reward: {max(eval_rewards):.0f} at checkpoint episode {available_eps[np.argmax(eval_rewards)]}")
    print(f"Worst reward: {min(eval_rewards):.0f} at checkpoint episode {available_eps[np.argmin(eval_rewards)]}")
    print(f"Mean reward: {np.mean(eval_rewards):.0f}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    # Also save the mapping of episode indices
    np.save(output_file.replace('.npy', '_episodes.npy'), np.array(available_eps))

    # Plot
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(available_eps, eval_rewards, 'b-o', markersize=3, label='Deterministic Eval')
    plt.xlabel('Checkpoint Episode')
    plt.ylabel('Episode Reward')
    plt.title('Deterministic Evaluation Across All Checkpoints')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Load and overlay original training rewards if available
    orig_rewards_path = os.path.join(os.path.dirname(model_dir),
                                     '..', 'logs',
                                     os.path.basename(model_dir).replace('model/', ''),
                                     'rewards.npy')
    # Try the expected path
    orig_log_dir = model_dir.replace('model/', 'logs/')
    orig_rewards_path = os.path.join(orig_log_dir, 'rewards.npy')
    if not os.path.exists(orig_rewards_path):
        orig_rewards_path = os.path.join(
            os.path.dirname(model_dir), '..', 'logs',
            os.path.basename(model_dir), 'rewards.npy')

    plt.subplot(1, 2, 2)
    if os.path.exists(orig_rewards_path):
        orig_rewards = np.load(orig_rewards_path)
        plt.plot(range(len(orig_rewards)), orig_rewards, 'r-', alpha=0.5, label='Training (w/ exploration)')
    plt.plot(available_eps, eval_rewards, 'b-o', markersize=3, label='Deterministic Eval')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training vs Deterministic Evaluation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, 'eval_all_ep_comparison.png'), dpi=150)
    plt.close()
    print(f"Plot saved to: {os.path.join(PIC_DIR, 'eval_all_ep_comparison.png')}")
