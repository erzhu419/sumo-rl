import psutil, tracemalloc
import gym
import copy
import importlib
import sys
from typing import Any, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import math
import gc
import time
import sys
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
import os
import argparse
import numpy as np
import random
# SUMO loading handled dynamically inside the rl_bridge.py
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import traci
import cProfile
import pstats
import io
from copy import deepcopy


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--use_gradient_clip', type=bool, default=True, help="Trick 1:gradient clipping")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
parser.add_argument("--training_freq", type=int, default=10, help="frequency of training the network")
parser.add_argument("--plot_freq", type=int, default=1, help="frequency of plotting the result")
parser.add_argument('--weight_reg', type=float, default=0.01, help='weight of regularization')
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.6, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--max_episodes", type=int, default=100, help="max episodes")
parser.add_argument('--save_root', type=str, default=os.path.dirname(os.path.abspath(__file__)), help='Base directory for saving models, logs, and figures')
parser.add_argument('--run_name', type=str, default='ensemble_run', help='Optional identifier appended to save directories')
parser.add_argument('--render', action='store_true', help='Enable environment rendering (SUMO GUI when available)')
parser.add_argument('--use_sumo_env', action='store_true', help='(deprecated) retained for backwards compatibility')
parser.add_argument('--no_sumo_env', action='store_true', help='Force legacy LSTM-RL env instead of SUMO bridge')
parser.add_argument('--sumo_root', type=str, default=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'), help='Base directory containing SUMO scenario assets')
parser.add_argument('--sumo_schedule', type=str, default='initialize_obj/save_obj_bus.add.xml', help='Path to schedule xml relative to --sumo_root')
parser.add_argument('--sumo_bridge', type=str, default='SUMO_ruiguang.online_control.rl_bridge:build_bridge', help='Python entrypoint returning decision_provider/action_executor callbacks')
parser.add_argument('--sumo_gui', action='store_true', help='Launch SUMO with GUI (implies rendering)')
parser.add_argument('--passenger_update_freq', type=int, default=10, help='Frequency of passenger/stop state updates (in steps)')
parser.add_argument('--profile', action='store_true', help='Enable cProfile performance analysis')

# Ensemble args
parser.add_argument("--ensemble_size", type=int, default=10, help="Number of models in the ensemble")
parser.add_argument("--beta_bc", type=float, default=0.001, help="weight of behavior cloning loss")
parser.add_argument("--beta", type=float, default=-2, help="weight of variance")
parser.add_argument("--beta_ood", type=float, default=0.01, help="weight of OOD loss")
parser.add_argument('--critic_actor_ratio', type=int, default=2, help="ratio of critic and actor training")
parser.add_argument('--holding_only', action='store_true', help='Force 1D action space (Holding Time only) instead of 2D [Hold, Speed]')
parser.add_argument('--speed_only', action='store_true', help='Force 1D action space (Speed Ratio only) instead of 2D [Hold, Speed]')
parser.add_argument('--use_1d_mapping', action='store_true', help='Use 1D scalar mapping to avoid action conflicts (Plan A)')
parser.add_argument("--use_residual_control", action="store_true", help="RL action offsets a Hard-Rule baseline")

args = parser.parse_args()


SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_NAME = args.run_name.strip() if args.run_name else None
SAVE_ROOT = os.path.abspath(args.save_root)
EXPERIMENT_ID = f"{SCRIPT_NAME}_{RUN_NAME}" if RUN_NAME else SCRIPT_NAME

PIC_DIR = os.path.join(SAVE_ROOT, 'pic', EXPERIMENT_ID)
LOG_DIR = os.path.join(SAVE_ROOT, 'logs', EXPERIMENT_ID)
MODEL_DIR = os.path.join(SAVE_ROOT, 'model', EXPERIMENT_ID)

for directory in (PIC_DIR, LOG_DIR, MODEL_DIR):
    os.makedirs(directory, exist_ok=True)

MODEL_PREFIX = os.path.join(MODEL_DIR, 'sac_v2_bus')


class ReplayBuffer:
    def __init__(self, capacity, last_episode_step=5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add new data to buffer, overwriting old if full"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[int(self.position)] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample batch_size elements in O(1)"""
        batch_indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return np.stack(states), np.stack(actions), np.array(rewards, dtype=np.float32), \
               np.stack(next_states), np.array(dones, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class LegacyEnvAdapter:
    def __init__(self, legacy_env):
        self.legacy_env = legacy_env
        self.action_space = legacy_env.action_space
        self.max_agent_num = legacy_env.max_agent_num
        self.expects_nested_actions = True

    def reset(self):
        return self.legacy_env.reset()

    def initialize_state(self, render=False):
        state, reward, done = self.legacy_env.initialize_state(render=render)
        return {'default': state}, {'default': reward}, done

    def step(self, action_dict, debug=False, render=False):
        flat_actions = action_dict.get('default', {})
        full_action = {key: flat_actions.get(key, None) for key in range(self.legacy_env.max_agent_num)}
        state, reward, done = self.legacy_env.step(full_action, debug=debug, render=render)
        return {'default': state}, {'default': reward}, done

    def get_feature_spec(self):
        cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
        cat_sizes = {
            'bus_id': self.legacy_env.max_agent_num,
            'station_id': max(round(len(self.legacy_env.stations) / 2), 1),
            'time_period': int(self.legacy_env.timetables[-1].launch_time // 3600) + 2,
            'direction': 2,
        }
        num_cont = self.legacy_env.state_dim - len(cat_cols)
        return {
            'cat_cols': cat_cols,
            'cat_sizes': cat_sizes,
            'num_cont_features': num_cont,
        }

    @property
    def line_codes(self):
        return ['default']

    @property
    def bus_codes(self):
        return list(range(self.legacy_env.max_agent_num))

    def __getattr__(self, item):
        return getattr(self.legacy_env, item)


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
        template = previous  # Keep tracking actions for buses currently between stations
        
    for line_id, buses in state_dict.items():
        if line_id not in template:
            template[line_id] = {}
        for bus_id in buses.keys():
            if bus_id not in template[line_id] or template[line_id][bus_id] is None:
                template[line_id][bus_id] = None
    return template


def get_reward_value(reward_dict, line_id, bus_id):
    return reward_dict.get(line_id, {}).get(bus_id, 0.0)


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
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_critics=ensemble_size,
            embedding_layer=embedding_layer
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
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        if args.use_residual_control:
            # Start perfectly at the Hard Rule Baseline and explore very conservatively (std ≈ 0.05)
            self.mean_linear.weight.data.fill_(0)
            self.mean_linear.bias.data.fill_(0)
            self.log_std_linear.weight.data.fill_(0)
            self.log_std_linear.bias.data.fill_(-3.0)
        else:
            self.mean_linear.weight.data.uniform_(-init_w, init_w)
            self.mean_linear.bias.data.uniform_(-init_w, init_w)
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

        mean = (self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))
        
        # Action Dim 0: Holding Time -> Map [-1, 1] to [0, 60] -> 30 * action_0 + 30
        # Action Dim 1: Speed Ratio -> Map [-1, 1] to [0.8, 1.2] -> 0.2 * action_0 + 1.0
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
        action = scale * action_0 + bias
        
        # log_prob adjustment: log(det(J)) for linear transformation is log(scale)
        # log_prob = log_prob(Gaussian) - sum(log(1 - tanh(x)^2) + log(scale))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) - torch.log(scale)
        log_prob = log_prob.sum(dim=1)
        return action, log_prob, z, mean, log_std

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

        # Fetch from global action_dim instead of hardcoding
        # The true log_prob is shifted down by log(det(J)) = log(scale). We must offset target_entropy UP
        # so target = -action_dim + sum(log(scale)), to make log_prob + target_entropy > 0 
        if action_dim == 1:
            if args.speed_only:
                log_scale_shift = np.log(0.2) # speed scale
            elif args.use_1d_mapping or args.use_residual_control:
                log_scale_shift = 0.0 # np.log(1.0) = 0
            else:
                log_scale_shift = np.log(30.0) # hold scale
        else:
            log_scale_shift = np.log(30.0) + np.log(0.2) # hold and speed scale
            
        self.target_entropy = -float(action_dim) + log_scale_shift

        self.cat_cols = cat_cols
        self.num_cat_features = len(cat_cols)
        # Extend continuous features by action_dim to include 'last_action'
        self.num_cont_features = num_cont_features + action_dim
        self.station_feature_idx = cat_cols.index('station_id') if 'station_id' in cat_cols else None
        
        embedding_template = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
        state_dim = embedding_template.output_dim + self.num_cont_features

        self.replay_buffer = replay_buffer

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), ensemble_size=ensemble_size).to(device)
        self.target_soft_q_net = deepcopy(self.soft_q_net)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), action_range).to(device)
        
        # SAC Entropy Temperature Tuning
        # Initialize alpha to 0.1 (log_alpha = ln(0.1)) to avoid pure maximum-entropy uniform noise exploration at the start.
        init_alpha = 0.1
        self.alpha = init_alpha
        self.log_alpha = torch.tensor([np.log(init_alpha)], dtype=torch.float32, requires_grad=True, device=device)
        
        print('Soft Q Network: ', self.soft_q_net)
        print('Policy Network: ', self.policy_net)

        self.soft_q_criterion = nn.MSELoss()

        # Update to standard SAC learning rates (3e-4) instead of extremely slow 1e-5.
        # This allows the alpha temperature and actor/critic to properly learn and converge.
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        initial_mean = np.zeros(self.num_cont_features)
        initial_std = np.ones(self.num_cont_features)

        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=initial_mean, init_std=initial_std)

        self.state_norm = Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)
        
    def compute_q_loss(self, state, action, reward, next_state, done, new_next_action, next_log_prob, reg_norm, gamma):
        predicted_q_value = self.soft_q_net(state, action)
        target_q_next = self.target_soft_q_net(next_state, new_next_action)
        next_log_prob = next_log_prob.unsqueeze(0).repeat(self.soft_q_net.num_critics, 1)
        reg_norm_exp = reg_norm.unsqueeze(-1).repeat(1, args.batch_size)
        target_q_next = target_q_next - self.alpha * next_log_prob + args.weight_reg * reg_norm_exp
        target_q_value = reward + (1 - done) * gamma * target_q_next.unsqueeze(-1)

        ood_loss = predicted_q_value.std(0).mean()
        q_value_loss = self.soft_q_criterion(predicted_q_value, target_q_value.squeeze(-1).detach())
        loss = q_value_loss + args.beta_ood * ood_loss
        return loss, predicted_q_value, ood_loss

    def compute_policy_loss(self, state, action, new_action, log_prob, reg_norm):
        reg_norm_exp = reg_norm.unsqueeze(-1).repeat(1, args.batch_size)
        q_values_dist = self.soft_q_net(state, new_action) + args.weight_reg * reg_norm_exp - self.alpha * log_prob

        q_mean = q_values_dist.mean(dim=0)
        q_std = q_values_dist.std(dim=0)
        q_loss = -(q_mean + args.beta * q_std).mean()

        bc_loss = F.mse_loss(new_action, action.detach() if isinstance(action, torch.Tensor) else torch.tensor(action, device=device))
        loss = args.beta_bc * bc_loss + q_loss

        return loss, q_loss, q_std

    def compute_alpha_loss(self, log_prob, target_entropy):
        alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
        return alpha_loss

    def compute_reg_norm(self, model):
        weight_norm, bias_norm = [], []
        for name, param in model.named_parameters():
            if 'critic' in name:
                if 'weight' in name:
                    weight_norm.append(torch.norm(param, p=1, dim=[1, 2]))
                elif 'bias' in name:
                    bias_norm.append(torch.norm(param, p=1, dim=[1, 2]))
        reg_norm = torch.sum(torch.stack(weight_norm), dim=0) + torch.sum(torch.stack(bias_norm[:-1]), dim=0)
        return reg_norm
    
    def update(self, batch_size, training_steps, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        global q_values, reg_norms1, reg_norms2, log_probs, alpha_values, ood_losses, q_stds
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        
        reward = reward * reward_scale

        if auto_entropy:
            alpha_loss = self.compute_alpha_loss(log_prob, target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=False)
            if training_steps % 1000 == 0: print(f"Alpha Loss: {alpha_loss.item():.4f}, log_prob: {log_prob.mean().item():.4f}, target_ent: {target_entropy:.4f}, Alpha: {self.alpha:.4f}")
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 1.
            alpha_loss = 0

        reg_norm = self.compute_reg_norm(self.target_soft_q_net)

        q_value_loss, predicted_q_value, ood_loss = self.compute_q_loss(state, action, reward, next_state, done, new_next_action, next_log_prob, reg_norm, gamma)
        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.soft_q_net.parameters(), max_norm=1.0)
        self.soft_q_optimizer.step()

        if training_steps % args.critic_actor_ratio == 0:
            policy_loss, _, q_std = self.compute_policy_loss(state, action, new_action, log_prob, reg_norm)
            q_stds.append(q_std.mean().item())

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            self.policy_optimizer.step()

        for target_param, param in zip(self.target_soft_q_net.parameters(), self.soft_q_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        # Logging vars
        q_values.append(predicted_q_value.mean().item())
        if reg_norm.numel() > 1:
            reg_norms1.append(args.weight_reg * reg_norm[0].item())
            reg_norms2.append(args.weight_reg * reg_norm[1].item())
        else:
            reg_norms1.append(args.weight_reg * reg_norm.item())
            reg_norms2.append(args.weight_reg * reg_norm.item())
             
        log_probs.append(-log_prob.mean().item())
        alpha_values.append(self.alpha)
        ood_losses.append(ood_loss.item())

        return predicted_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net.state_dict(), path + '_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch.save(self.state_norm, path + '_norm')

    def load_model(self, path):
        self.soft_q_net.load_state_dict(torch.load(path + '_q', weights_only=True))
        self.policy_net.load_state_dict(torch.load(path + '_policy', weights_only=True))
        self.soft_q_net.eval()
        self.policy_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.title(f"Training Reward (weight_reg={args.weight_reg}, auto_entropy={args.auto_entropy})")
    plt.subplot(1, 2, 2)

    plt.plot(q_values_episode, label="Q-Value Mean")
    plt.plot(reg_norms1_episode, label="Reg Norm 1")
    plt.plot(reg_norms2_episode, label="Reg Norm 2")
    plt.plot(log_probs_episode, label="Log Prob")
    plt.plot(alpha_values_episode, label="Alpha")
    plt.plot(ood_losses_episode, label="OOD Loss")
    plt.plot(q_stds_episode, label="Q Std")

    plt.legend()
    plt.title(f"Monitoring (weight_reg={args.weight_reg})")

    if not os.path.exists(PIC_DIR):
        os.makedirs(PIC_DIR, exist_ok=True)
    plt.savefig(os.path.join(PIC_DIR, 'sac_monitoring.png'))
    plt.close()

# Evaluate policy is not fundamentally changed but can be added if needed, skipping for brevity or add if requested.

replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)

debug = False
render = bool(getattr(args, 'render', False))
if getattr(args, 'sumo_gui', False):
    render = True
use_sumo_env = True
if getattr(args, 'no_sumo_env', False):
    use_sumo_env = False
elif getattr(args, 'use_sumo_env', False):
    use_sumo_env = True

if use_sumo_env:
    if not args.sumo_bridge:
        raise ValueError("--sumo_bridge must be provided when --use_sumo_env is enabled")
    module_name, _, attr_name = args.sumo_bridge.partition(':')
    bridge_module = importlib.import_module(module_name)
    factory = getattr(bridge_module, attr_name or 'build_bridge')
    bridge = factory(root_dir=args.sumo_root, gui=getattr(args, 'sumo_gui', False) or render, update_freq=args.passenger_update_freq)
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
        root_dir=args.sumo_root,
        schedule_file=args.sumo_schedule,
        decision_provider=decision_provider,
        action_executor=action_executor,
        reset_callback=reset_cb,
        close_callback=close_cb,
        debug=debug,
        reward_type="linear_penalty",
    )
else:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env')
    env = env_bus(path, debug=debug)

if not getattr(env, 'expects_nested_actions', False):
    env = LegacyEnvAdapter(env)

env.reset()

# Initialize Trajectory Log
with open(os.path.join(LOG_DIR, "trajectory_log.csv"), "w") as f:
    f.write("LineID,BusID,Step,S_Stop,NS_Stop,Action,Reward,S_Fwd_Headway,S_Bwd_Headway\n")

# Initialize Action Log
with open(os.path.join(LOG_DIR, "action_log.csv"), "w") as f:
    f.write("LineID,BusID,Step,StopID,Action,Fwd_Headway,Bwd_Headway\n")

# Initialize buffers for logging
trajectory_log_buffer = []
action_log_buffer = []

# action_dim = 2 # Forced 2D action for Speed Control
action_dim = 1 if (args.holding_only or args.speed_only or args.use_1d_mapping) else 2
action_range = 1.0

step = 0
step_trained = 0
explore_steps = 0
update_itr = 1
DETERMINISTIC = False
hidden_dim = 32

def format_action_for_log(action_val, use_1d, use_residual=False):
    if action_val is None:
        return "0.0"
    if use_residual:
        if isinstance(action_val, (np.ndarray, list)) and len(action_val) >= 2:
            return f"res({action_val[0]:.4f}_{action_val[1]:.4f})"
        else:
            a = action_val[0] if isinstance(action_val, (np.ndarray, list)) else action_val
            return f"res({a:.4f})"
    if use_1d:
        a = action_val[0] if isinstance(action_val, (np.ndarray, list)) else action_val
        if a > 0:
            print_val = [a * 60.0, 1.0]
        elif a < 0:
            print_val = [0.0, 1.0 + abs(a) * 0.2]
        else:
            print_val = [0.0, 1.0]
        return "_".join([f"{v:.4f}" for v in print_val]) + f"({a:.4f})"
    if isinstance(action_val, (np.ndarray, list)):
        return "_".join([f"{v:.4f}" for v in action_val])
    return f"{action_val:.4f}"

rewards = []
q_values = []
reg_norms1 = []
reg_norms2 = []
log_probs = []
alpha_values = []
ood_losses = []
q_stds = []

q_values_episode = []
reg_norms1_episode = []
reg_norms2_episode = []
log_probs_episode = []
alpha_values_episode = []
ood_losses_episode = []
q_stds_episode = []

tracemalloc.start()

sac_trainer = SAC_Trainer(env, replay_buffer, hidden_dim=hidden_dim, action_range=action_range, ensemble_size=args.ensemble_size)

if __name__ == '__main__':
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.train:
        for eps in range(args.max_episodes):
            episode_start_time = time.time()
            if eps != 0:
                env.reset()
            state_dict, reward_dict, _ = safe_initialize_state(env, render=render)

            done = False
            episode_steps = 0
            training_steps = 0
            action_dict = build_action_template(state_dict)
            # Track last actions to include in the state
            last_action_history = defaultdict(lambda: defaultdict(lambda: np.zeros(action_dim, dtype=np.float32)))
            
            episode_reward = 0
            station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1
            
            while not done:
                action_dict = build_action_template(state_dict, action_dict)
                for line_id, buses in state_dict.items():
                    for bus_id, history in buses.items():
                        if len(history) == 0:
                            continue
                        if len(history) == 1:
                            if action_dict[line_id][bus_id] is None:
                                # Append previous action to numerical state features
                                last_action = last_action_history[line_id][bus_id]
                                state_vec = np.concatenate([history[0], last_action])
                                
                                if args.use_state_norm:
                                    state_vec = sac_trainer.state_norm(state_vec)
                                action = sac_trainer.policy_net.get_action(torch.from_numpy(state_vec).float(), deterministic=DETERMINISTIC)
                                action_dict[line_id][bus_id] = action
                                # NOTE: last_action_history stays as A_{N-1} until the reward for A_N is collected at the next station.


                                # Action Logging (Case 1: First Stop)
                                action_val = action
                                action_str = format_action_for_log(action_val, args.use_1d_mapping, args.use_residual_control)
                                    
                                stop_id = history[0][station_feature_idx]
                                fwd_h = history[0][5]
                                bwd_h = history[0][6]
                                log_entry = f"{line_id},{bus_id},{step},{stop_id},{action_str},{fwd_h:.2f},{bwd_h:.2f}\n"
                                action_log_buffer.append(log_entry)

                        elif len(history) >= 2:
                            state_vec = np.array(history[0])
                            next_state_vec = np.array(history[1])
                            
                            if history[0][station_feature_idx] != history[1][station_feature_idx]:
                                if args.use_state_norm:
                                    state_vec_norm = sac_trainer.state_norm(state_vec)
                                    next_state_vec_norm = sac_trainer.state_norm(next_state_vec)
                                else:
                                    state_vec_norm = state_vec
                                    next_state_vec_norm = next_state_vec
                                    
                                current_reward = get_reward_value(reward_dict, line_id, bus_id)
                                if args.use_reward_scaling:
                                    scaled_reward = sac_trainer.reward_scaling(current_reward)
                                else:
                                    scaled_reward = current_reward

                                # Retrieve stored action
                                stored_action = action_dict[line_id][bus_id]
                                if stored_action is None:
                                    stored_action = np.zeros(action_dim, dtype=np.float32)
                                elif np.isscalar(stored_action):
                                    stored_action = np.array([stored_action], dtype=np.float32)
                                else:
                                    stored_action = np.array(stored_action, dtype=np.float32).reshape(-1)

                                # Heuristic Guidance Penalty: Penalize Hold > 0 AND Speed > 1.0 (Acceleration)
                                # Only applies in 2D Mixed Action mode
                                guidance_penalty = 0.0
                                if action_dim == 2:
                                    hold_val = stored_action[0]
                                    speed_val = stored_action[1]
                                    # Penalize if holding while speeding (inconsistent intent)
                                    # Penalty scale matches headway penalty order of magnitude
                                    if hold_val > 5.0 and speed_val > 1.05:
                                        guidance_penalty = 100.0 * (hold_val / 30.0) * (speed_val - 1.0)
                                
                                scaled_reward -= guidance_penalty

                                # State Extension: The buffer state MUST match the policy input
                                # We need history[0] + last_action_at_T_minus_1
                                # and history[1] + last_action_at_T (stored_action)
                                last_action_prev = last_action_history[line_id][bus_id] # This was updated ONLY in the 'get_action' block
                                # Wait, I need to be careful with the timing.
                                # At station N, we get obs S_N, take action A_N.
                                # At station N+1, we get obs S_{N+1}, reward R_{N+1}.
                                # The transition is ( [S_N, A_{N-1}], A_N, R_{N+1}, [S_{N+1}, A_N] )
                                
                                state_ext = np.concatenate([state_vec, last_action_prev])
                                next_state_ext = np.concatenate([next_state_vec, stored_action])

                                if args.use_state_norm:
                                    state_ext = sac_trainer.state_norm(state_ext)
                                    next_state_ext = sac_trainer.state_norm(next_state_ext)

                                # Push to buffer
                                replay_buffer.push(state_ext, stored_action, scaled_reward, next_state_ext, done)
                                
                                # Now update last_action_history for the NEXT decision point
                                last_action_history[line_id][bus_id] = stored_action
                                
                                # Trajectory Logging
                                raw_reward = current_reward
                                
                                action_val = stored_action
                                action_str = format_action_for_log(action_val, args.use_1d_mapping)
                                    
                                s_stop = history[0][station_feature_idx]
                                ns_stop = history[1][station_feature_idx]
                                s_fwd_h = history[0][5]
                                s_bwd_h = history[0][6]
                                log_entry = f"{line_id},{bus_id},{step},{s_stop},{ns_stop},{action_str},{raw_reward:.4f},{s_fwd_h:.2f},{s_bwd_h:.2f}\n"
                                trajectory_log_buffer.append(log_entry)

                                episode_steps += 1
                                step += 1
                                episode_reward += current_reward

                            state_dict[line_id][bus_id] = history[1:]
                            # Subsequent Stops: Decision for Station N+1
                            # Next State Extension: history[1] (S_{N+1}) + stored_action (A_N)
                            state_vec_next = np.concatenate([history[1], stored_action])
                            if args.use_state_norm:
                                state_vec_next = sac_trainer.state_norm(state_vec_next)
                                
                            action_dict[line_id][bus_id] = sac_trainer.policy_net.get_action(torch.from_numpy(state_vec_next).float(), deterministic=DETERMINISTIC)
                            
                            # Action Logging (Case 2: Subsequent Stops)
                            new_action = action_dict[line_id][bus_id]
                            action_val = new_action
                            action_str = format_action_for_log(action_val, args.use_1d_mapping, args.use_residual_control)
                                
                            stop_id = state_dict[line_id][bus_id][0][station_feature_idx]
                            fwd_h = state_dict[line_id][bus_id][0][5]
                            bwd_h = state_dict[line_id][bus_id][0][6]
                            log_entry = f"{line_id},{bus_id},{step},{stop_id},{action_str},{fwd_h:.2f},{bwd_h:.2f}\n"
                            action_log_buffer.append(log_entry)

                # Map 1D policy actions to 2D environment actions
                env_action_dict = copy.deepcopy(action_dict)
                for line_id, buses in env_action_dict.items():
                    for bus_id, action_val in buses.items():
                        if action_val is not None:
                            if args.use_residual_control:
                                a = action_val[0]
                                bus_state_history = state_dict.get(line_id, {}).get(bus_id, [])
                                if len(bus_state_history) > 0:
                                    gap = bus_state_history[0][11] # Index of newly added gap feature
                                else:
                                    gap = 0.0
                                
                                # 1. Compute Base Hard Rule
                                if gap > 0:
                                    base_hold = min(60.0, gap)
                                    base_speed = 1.0
                                else:
                                    base_hold = 0.0
                                    base_speed = 1.2
                                
                                # 2. Apply 2D RL Residual Offset (-1 to 1)
                                a_hold = action_val[0]
                                a_speed = action_val[1]
                                
                                hold = min(60.0, max(0.0, base_hold + a_hold * 20.0))
                                speed = min(1.2, max(0.1, base_speed + a_speed * 0.2))
                                    
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
                
                if len(replay_buffer) > args.batch_size and len(replay_buffer) % args.training_freq == 0 and step_trained != step:
                    step_trained = step
                    for i in range(update_itr):
                        _ = sac_trainer.update(args.batch_size, training_steps, reward_scale=0.1, auto_entropy=args.auto_entropy, target_entropy=sac_trainer.target_entropy)
                        training_steps += 1

                if len(action_log_buffer) > 50:
                    with open(os.path.join(LOG_DIR, "trajectory_log.csv"), "a") as f:
                        f.writelines(trajectory_log_buffer)
                    trajectory_log_buffer.clear()
                    
                    with open(os.path.join(LOG_DIR, "action_log.csv"), "a") as f:
                        f.writelines(action_log_buffer)
                    action_log_buffer.clear()

                if done:
                    replay_buffer.last_episode_step = episode_steps
                    break
            
            # Flush Logs to file periodically or at end of episode to avoid huge memory usage
            if trajectory_log_buffer:
                with open(os.path.join(LOG_DIR, "trajectory_log.csv"), "a") as f:
                    f.writelines(trajectory_log_buffer)
                trajectory_log_buffer.clear()
                
            if action_log_buffer:
                with open(os.path.join(LOG_DIR, "action_log.csv"), "a") as f:
                    f.writelines(action_log_buffer)
                action_log_buffer.clear()
            
            def safe_mean(arr):
                if len(arr) == 0:
                    return 0.0
                return np.mean(arr)

            rewards.append(episode_reward)
            
            num_added_q = len(q_values)
            num_added_stds = len(q_stds)
            
            if num_added_q > 0:
                q_values_episode.append(safe_mean(q_values))
                reg_norms1_episode.append(safe_mean(reg_norms1))
                reg_norms2_episode.append(safe_mean(reg_norms2))
                log_probs_episode.append(safe_mean(log_probs))
                alpha_values_episode.append(safe_mean(alpha_values))
                ood_losses_episode.append(safe_mean(ood_losses))
            else:
                q_values_episode.append(0)
                reg_norms1_episode.append(0)
                reg_norms2_episode.append(0)
                log_probs_episode.append(0)
                alpha_values_episode.append(0)
                ood_losses_episode.append(0)
                
            if num_added_stds > 0:
                q_stds_episode.append(safe_mean(q_stds))
            else:
                q_stds_episode.append(0)
            
            q_values.clear()
            reg_norms1.clear()
            reg_norms2.clear()
            log_probs.clear()
            alpha_values.clear()
            ood_losses.clear()
            q_stds.clear()

            if eps % args.plot_freq == 0:
                plot(rewards)
                
                np.save(os.path.join(LOG_DIR, 'rewards.npy'), rewards)
                np.save(os.path.join(LOG_DIR, 'q_values_episode.npy'), q_values_episode)
                np.save(os.path.join(LOG_DIR, 'reg_norms1_episode.npy'), reg_norms1_episode)
                np.save(os.path.join(LOG_DIR, 'reg_norms2_episode.npy'), reg_norms2_episode)
                np.save(os.path.join(LOG_DIR, 'log_probs_episode.npy'), log_probs_episode)
                np.save(os.path.join(LOG_DIR, 'alpha_values_episode.npy'), alpha_values_episode)
                np.save(os.path.join(LOG_DIR, 'ood_losses_episode.npy'), ood_losses_episode)
                np.save(os.path.join(LOG_DIR, 'q_stds_episode.npy'), q_stds_episode)
                
                sac_trainer.save_model(os.path.join(MODEL_DIR, f"checkpoint_episode_{eps}"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            replay_buffer_usage = len(replay_buffer) / parser.parse_args().batch_size * 100 if parser.parse_args().batch_size > 0 else 0
            # Just approximate usage since replay buffer capacity isn't passed from args directly but initialized via replay_buffer_size = 1e6.
            replay_buffer_usage = len(replay_buffer) / replay_buffer_size * 100
            episode_duration = time.time() - episode_start_time
            
            print(
                f"Episode: {eps} | Episode Reward: {episode_reward} | Duration: {episode_duration:.2f}s "
                f"| CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | "
                f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | "
                f"Replay Buffer Usage: {replay_buffer_usage:.2f}%")
        
        sac_trainer.save_model(os.path.join(MODEL_DIR, "final"))

    if args.profile:
        profiler.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(30)
        print(s.getvalue())
