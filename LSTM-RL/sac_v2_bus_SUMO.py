import psutil,tracemalloc
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
from normalization import Normalization, RewardScaling, RunningMeanStd

from IPython.display import clear_output
import matplotlib.pyplot as plt
from env.sim import env_bus
import os
import argparse
import numpy as np
import random
import libsumo as traci
import cProfile
import pstats
import io

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
parser.add_argument("--training_freq", type=int, default=10, help="frequency of training the network")
parser.add_argument("--plot_freq", type=int, default=1, help="frequency of plotting the result")
parser.add_argument('--weight_reg', type=float, default=0.1, help='weight of regularization')
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.3, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--max_episodes", type=int, default=100, help="max episodes")
parser.add_argument('--save_root', type=str, default='.', help='Base directory for saving models, logs, and figures')
parser.add_argument('--run_name', type=str, default='gpt_version', help='Optional identifier appended to save directories to avoid overwriting previous runs')
parser.add_argument('--render', action='store_true', help='Enable environment rendering (SUMO GUI when available)')
parser.add_argument('--use_sumo_env', action='store_true', help='(deprecated) retained for backwards compatibility')
parser.add_argument('--no_sumo_env', action='store_true', help='Force legacy LSTM-RL env instead of SUMO bridge')
parser.add_argument('--sumo_root', type=str, default=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'), help='Base directory containing SUMO scenario assets')
parser.add_argument('--sumo_schedule', type=str, default='initialize_obj/save_obj_bus.add.xml', help='Path to schedule xml relative to --sumo_root')
parser.add_argument('--sumo_bridge', type=str, default='SUMO_ruiguang.online_control.rl_bridge:build_bridge', help='Python entrypoint returning decision_provider/action_executor callbacks, format module:function')
parser.add_argument('--sumo_gui', action='store_true', help='Launch SUMO with GUI (implies rendering)')
parser.add_argument('--passenger_update_freq', type=int, default=10, help='Frequency of passenger/stop state updates (in steps)')
parser.add_argument('--profile', action='store_true', help='Enable cProfile performance analysis')
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
        self.last_episode_step = last_episode_step  # 预估每个 episode 的 step 数
        self.buffer = {}
        self.position = 0  # 用作 dict 的 key

    def push(self, state, action, reward, next_state, done):
        """添加新数据"""
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position += 1

        # 当 buffer 过大时，删除最早的 episode 数据
        if len(self.buffer) > self.capacity:
            keys_to_remove = list(self.buffer.keys())[:self.last_episode_step]  # 找到最早的 N 条数据
            for key in keys_to_remove:
                del self.buffer[key]  # 直接删除，提高性能

    def sample(self, batch_size):
        """随机采样 batch_size 大小的数据，确保数据格式正确"""
        batch = random.sample(list(self.buffer.values()), batch_size)  # 直接从 dict 的值采样
        states, actions, rewards, next_states, dones = zip(*batch)

        # 确保维度正确，防止 PyTorch 计算时出现广播错误
        states = np.stack(states)                      # (batch_size, state_dim)
        actions = np.stack(actions)                    # (batch_size, action_dim) 或 (batch_size,)
        rewards = np.array(rewards, dtype=np.float32)  # (batch_size,)
        next_states = np.stack(next_states)            # (batch_size, state_dim)
        dones = np.array(dones, dtype=np.float32)      # (batch_size,)

        return states, actions, rewards, next_states, dones

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
    template = {}
    for line_id, buses in state_dict.items():
        template[line_id] = {}
        for bus_id in buses.keys():
            if previous and line_id in previous and bus_id in previous[line_id]:
                template[line_id][bus_id] = previous[line_id][bus_id]
            else:
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

    @classmethod
    def compute_output_dim(cls, cat_code_dict, cat_cols, embedding_dims=None) -> int:
        total = 0
        for col in cat_cols:
            codes = list(cat_code_dict[col].values())
            if len(codes) == 0:
                continue
            cardinality = max(codes) + 1
            if embedding_dims and col in embedding_dims:
                total += embedding_dims[col]
            else:
                total += cls._suggest_dim(cardinality)
        return total

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


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]  # Assuming first columns are categorical
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]  # The rest are numerical

        # cat_tensor = torch.clamp(cat_tensor, min=0, max=max(self.embedding_layer.cat_code_dict.values()))
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)  # Concatenate embedding and numerical features
        x = torch.cat([state_with_embeddings, action], 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


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

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range/2 * action_0 + self.action_range/2  # bounded action
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range/2 * torch.tanh(mean + std * z) + self.action_range/2

        action = self.action_range/2 * torch.tanh(mean).detach().cpu().numpy()[0] + self.action_range/2 if deterministic else action.detach().cpu().numpy()[0]
        return action

class SAC_Trainer():
    def __init__(self, env, replay_buffer, hidden_dim, action_range):
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

        self.cat_cols = cat_cols
        self.num_cat_features = len(cat_cols)
        self.num_cont_features = num_cont_features
        self.station_feature_idx = cat_cols.index('station_id') if 'station_id' in cat_cols else None
        # 创建嵌入层模板，并为每个网络提供独立副本，避免目标网络与在线网络共享参数
        embedding_template = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
        state_dim = embedding_template.output_dim + self.num_cont_features  # 状态维度 = 嵌入维度 + 数值特征维度

        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone()).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone()).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone()).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone()).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 1e-5
        policy_lr = 1e-5
        alpha_lr = 1e-5

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # 初始化RunningMeanStd
        initial_mean = np.zeros(self.num_cont_features)
        initial_std = np.ones(self.num_cont_features)

        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=initial_mean, init_std=initial_std)

        self.state_norm = Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)
    
    def update(self, batch_size,training_steps, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=False)
            self.alpha_optimizer.step()
            self.alpha = min(args.maximum_alpha, self.log_alpha.exp().item())
        else:
            self.alpha = 1.
            alpha_loss = 0

        # 计算 reg_norm
        reg_norm1, weight_norm1, bias_norm1 = 0, [], []
        reg_norm2, weight_norm2, bias_norm2 = 0, [], []

        for layer in self.target_soft_q_net1.children():
            if isinstance(layer, nn.Linear):
                weight_norm1.append(torch.norm(layer.state_dict()['weight']) ** 2)
                bias_norm1.append(torch.norm(layer.state_dict()['bias']) ** 2)

        reg_norm1 = torch.sqrt(torch.sum(torch.stack(weight_norm1)) + torch.sum(torch.stack(bias_norm1[0:-1])))

        for layer in self.target_soft_q_net2.children():
            if isinstance(layer, nn.Linear):
                weight_norm2.append(torch.norm(layer.state_dict()['weight']) ** 2)
                bias_norm2.append(torch.norm(layer.state_dict()['bias']) ** 2)

        reg_norm2 = torch.sqrt(torch.sum(torch.stack(weight_norm2)) + torch.sum(torch.stack(bias_norm2[0:-1])))


        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action) - args.weight_reg * reg_norm1,
                                 self.target_soft_q_net2(next_state, new_next_action) - args.weight_reg * reg_norm2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(sac_trainer.soft_q_net1.parameters(), max_norm=1.0)  # Q 网络梯度裁剪
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(sac_trainer.soft_q_net2.parameters(), max_norm=1.0)  # Q 网络梯度裁剪
        self.soft_q_optimizer2.step()

        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action) + args.weight_reg * reg_norm1, self.soft_q_net2(state, new_action)
                                          + args.weight_reg * reg_norm2)
        # Training Policy Function
        if training_steps % 2 == 0:
            policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            self.policy_optimizer.step()

            # print('q loss: ', q_value_loss1.item(), q_value_loss2.item())
            # print('policy loss: ', policy_loss.item())

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        # 记录 Q 值和 V 值用于绘图
        q_values.append(predicted_new_q_value.mean().item())
        reg_norms1.append(args.weight_reg * reg_norm1.item())
        reg_norms2.append(args.weight_reg * reg_norm2.item())
        log_probs.append(-log_prob.mean().item())
        alpha_values.append(self.alpha)

        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1', weights_only=True))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2', weights_only=True))
        self.policy_net.load_state_dict(torch.load(path + '_policy',weights_only=True))

        self.soft_q_net1.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def evaluate_policy(sac_trainer, env, num_eval_episodes=5, deterministic=True):
    """
    评估当前策略，返回多次评估的平均奖励和方差
    :param sac_trainer: SAC_Trainer实例
    :param env: 环境
    :param num_eval_episodes: 评估的episode数量
    :param deterministic: 是否使用确定性策略
    :return: (平均奖励, 奖励方差)
    """
    eval_rewards = []
    
    for eval_ep in range(num_eval_episodes):
        print(f"Evaluating episode {eval_ep + 1}/{num_eval_episodes}...")
        env.reset()
        state_dict, reward_dict, _ = safe_initialize_state(env, render=False)
        
        done = False
        episode_reward = 0
        action_dict = build_action_template(state_dict)
        station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1
        
        while not done:
            action_dict = build_action_template(state_dict, action_dict)
            for line_id, buses in state_dict.items():
                for bus_id, history in buses.items():
                    if len(history) == 0:
                        continue
                    if len(history) == 1:
                        if action_dict[line_id][bus_id] is None:
                            state_input = np.array(history[0])
                            state_tensor = torch.from_numpy(state_input).float()
                            a = sac_trainer.policy_net.get_action(state_tensor, deterministic=deterministic)
                            action_dict[line_id][bus_id] = a
                    elif len(history) >= 2:
                        if history[0][station_feature_idx] != history[1][station_feature_idx]:
                            episode_reward += get_reward_value(reward_dict, line_id, bus_id)
                        state_dict[line_id][bus_id] = history[1:]
                        state_input = np.array(state_dict[line_id][bus_id][0])
                        state_tensor = torch.from_numpy(state_input).float()
                        action_dict[line_id][bus_id] = sac_trainer.policy_net.get_action(state_tensor, deterministic=deterministic)

            state_dict, reward_dict, done = safe_step(env, action_dict, render=False)

        eval_rewards.append(episode_reward)
    
    if len(eval_rewards) > 0:
        mean_reward = np.mean(eval_rewards)
        reward_std = np.std(eval_rewards)
    else:
        mean_reward = 0.0
        reward_std = 0.0
    
    return mean_reward, reward_std


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.title(f"Training Reward (weight_reg={args.weight_reg}, auto_entropy={args.auto_entropy}, reward_scaling={args.use_reward_scaling}, maximum_alpha={args.maximum_alpha})")
    plt.subplot(1, 2, 2)

    plt.plot(q_values_episode, label="Q-Value")
    plt.plot(reg_norms1_episode, label="Regularization Term1")
    plt.plot(reg_norms2_episode, label="Regularization Term2")
    plt.plot(log_probs_episode, label="Log Prob")
    plt.plot(alpha_values_episode, label="Alpha")

    plt.legend()
    plt.title(f"Q-Value & V-Value and log_prob & regularization Monitoring (weight_reg={args.weight_reg})")

    if not os.path.exists(PIC_DIR):
        os.makedirs(PIC_DIR, exist_ok=True)

    plt.savefig(os.path.join(PIC_DIR, f'sac_monitoring_weight_reg_{args.weight_reg}.png'))
    plt.close()

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
    module_name, _, attr_name = args.sumo_bridge.partition(':')
    bridge_module = importlib.import_module(module_name)
    factory = getattr(bridge_module, attr_name or 'build_bridge')
    bridge = factory(root_dir=args.sumo_root, gui=getattr(args, 'sumo_gui', False) or render, update_freq=args.passenger_update_freq)
    if isinstance(bridge, tuple):
        decision_provider = bridge[0] if len(bridge) > 0 else None
        action_executor = bridge[1] if len(bridge) > 1 else None
        reset_cb = bridge[2] if len(bridge) > 2 else None
        close_cb = bridge[3] if len(bridge) > 3 else None
    elif isinstance(bridge, dict):
        decision_provider = bridge.get('decision_provider')
        action_executor = bridge.get('action_executor')
        reset_cb = bridge.get('reset_callback')
        close_cb = bridge.get('close_callback')
    else:
        raise ValueError("Bridge factory must return a tuple or dict with decision_provider/action_executor")
    if decision_provider is None or action_executor is None:
        raise ValueError("Bridge must provide decision_provider and action_executor")
    from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv
    env = SumoBusHoldingEnv(
        root_dir=args.sumo_root,
        schedule_file=args.sumo_schedule,
        decision_provider=decision_provider,
        action_executor=action_executor,
        reset_callback=reset_cb,
        close_callback=close_cb,
        debug=debug,
    )
else:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env')
    env = env_bus(path, debug=debug)

if not getattr(env, 'expects_nested_actions', False):
    env = LegacyEnvAdapter(env)

env.reset()

# Initialize Trajectory Log
with open("trajectory_log.csv", "w") as f:
    f.write("LineID,BusID,Step,S_Stop,NS_Stop,Action,Reward,S_Fwd_Headway,S_Bwd_Headway\n")

# Initialize Action Log
with open("action_log.csv", "w") as f:
    f.write("LineID,BusID,Step,StopID,Action,Fwd_Headway,Bwd_Headway\n")

# Initialize buffers for logging
trajectory_log_buffer = []
action_log_buffer = []

action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

# hyperparameters for RL training

step = 0
step_trained = 0
frame_idx = 0
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = 32

rewards = []    # 记录奖励
q_values = []  # 记录 Q 值变化
reg_norms1 = []  # 记录正则化项1
reg_norms2 = []  # 记录正则化项2
log_probs = []  # 记录 log_prob
alpha_values = []  # 记录 alpha 值

q_values_episode = []  # 记录每个 episode 的 Q 值
reg_norms1_episode = []  # 记录每个 episode 的正则化项1
reg_norms2_episode = []  # 记录每个 episode 的正则化项2
log_probs_episode = []  # 记录每个 episode 的 log_prob
alpha_values_episode = []  # 记录每个 episode 的 alpha 值

# 创建用于存储评估结果的列表
eval_episodes = []      # 记录进行评估的 episode 编号
eval_mean_rewards = []  # 记录评估的平均奖励
eval_reward_stds = []   # 记录评估的奖励标准差

# 修改模型保存路径
model_path = MODEL_PREFIX

tracemalloc.start()

sac_trainer = SAC_Trainer(env, replay_buffer, hidden_dim=hidden_dim, action_range=action_range)

if __name__ == '__main__':
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.train:
        # training loop
        for eps in range(args.max_episodes):
            if eps != 0:
                env.reset()
            state_dict, reward_dict, _ = safe_initialize_state(env, render=render)

            done = False
            episode_steps = 0
            training_steps = 0 # 记录已经训练了多少次
            action_dict = build_action_template(state_dict)
            episode_reward = 0
            station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1

            while not done:
                action_dict = build_action_template(state_dict, action_dict)
                for line_id, buses in state_dict.items():
                    for i, (bus_id, history) in enumerate(buses.items()):
                        # DEBUG: Print state for the first bus to verify values
                        # if i == 0 and len(history) > 0:
                        #     current_obs = history[0]
                        #     obs: [line, bus, station, time, dir, fwd_h, bwd_h, wait, target, duration]
                        #     if len(current_obs) > 7:
                        #         print(f"DEBUG_BUS [{bus_id}]: Step {step}, Station {current_obs[2]}, Fwd_H {current_obs[5]:.2f}, Bwd_H {current_obs[6]:.2f}, Wait {current_obs[7]}, Current time {current_obs[-1]}")

                        if len(history) == 0:
                            continue
                        if len(history) == 1:
                            if action_dict[line_id][bus_id] is None:
                                state_vec = np.array(history[0])
                                if args.use_state_norm:
                                    state_vec = sac_trainer.state_norm(state_vec)
                                action = sac_trainer.policy_net.get_action(torch.from_numpy(state_vec).float(), deterministic=DETERMINISTIC)
                                action_dict[line_id][bus_id] = action
                                
                                # Action Logging (Case 1: First Stop)
                                action_val = action[0] if isinstance(action, (np.ndarray, list)) else action
                                stop_id = history[0][station_feature_idx]
                                sim_time = history[0][-1]
                                fwd_h = history[0][5]
                                bwd_h = history[0][6]
                                log_entry = f"{line_id},{bus_id},{step},{stop_id},{action_val:.4f},{fwd_h:.2f},{bwd_h:.2f}\n"
                                action_log_buffer.append(log_entry)
                                
                                if debug:
                                    # Format: From Algorithm, when no state, Bus id: ... , station id is: ... , current time is: ... , action is: ... , reward: ...
                                    print(f'From Algorithm, when no state, Bus id: {line_id}_{bus_id} , station id is: {int(stop_id)} , current time is: {sim_time:.1f} , action is: {action_val:.4f}, reward: {get_reward_value(reward_dict, line_id, bus_id):.4f}')
                                    print()

                        elif len(history) >= 2:
                            state_vec = np.array(history[0])
                            next_state_vec = np.array(history[1])
                            
                            # Check if station changed (valid transition)
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
                                stored_action = action_dict[line_id][bus_id] if action_dict[line_id][bus_id] is not None else 0.0
                                
                                replay_buffer.push(state_vec_norm, stored_action, scaled_reward, next_state_vec_norm, done)
                                
                                if debug:
                                    # Format: From Algorithm store, Bus id: ... , station id is: ... , current time is: ... , action is: ... , reward: ...
                                    # Note: using history[0] (previous state) for station id and time to match the 'store' context
                                    store_time = history[0][-1]
                                    store_stop = history[0][station_feature_idx]
                                    # print(f'From Algorithm store, Bus id: {line_id}_{bus_id} , station id is: {int(store_stop)} , current time is: {store_time:.1f} , action is: {stored_action}, reward: {current_reward:.4f}')
                                    pass
                                
                                # Trajectory Logging
                                raw_reward = current_reward
                                action_val = stored_action
                                if isinstance(action_val, (np.ndarray, list)):
                                    action_val = action_val[0]
                                s_stop = history[0][station_feature_idx]
                                ns_stop = history[1][station_feature_idx]
                                s_fwd_h = history[0][5]
                                s_bwd_h = history[0][6]
                                log_entry = f"{line_id},{bus_id},{step},{s_stop},{ns_stop},{action_val:.4f},{raw_reward:.4f},{s_fwd_h:.2f},{s_bwd_h:.2f}\n"
                                trajectory_log_buffer.append(log_entry)
                                # DEBUG: Check for large negative rewards
                                if raw_reward < -100:
                                    print(f"DEBUG_REWARD: Line {line_id}, Bus {bus_id}, Step {step}, Reward {raw_reward:.2f}, EpReward {episode_reward:.2f}, Fwd {history[0][5]:.2f}, Bwd {history[0][6]:.2f}")

                                episode_steps += 1
                                step += 1
                                episode_reward += current_reward

                            # Initializing Action for Next State
                            state_dict[line_id][bus_id] = history[1:] # Shift history
                            # Now history is history[1:] - [0] is the NEW state
                            
                            new_state_vec = np.array(state_dict[line_id][bus_id][0])
                            if args.use_state_norm:
                                new_state_vec_norm = sac_trainer.state_norm(new_state_vec)
                            else:
                                new_state_vec_norm = new_state_vec

                            action = sac_trainer.policy_net.get_action(torch.from_numpy(new_state_vec_norm).float(), deterministic=DETERMINISTIC)
                            action_dict[line_id][bus_id] = action

                            # Action Logging (Case 2: Subsequent Stops - Initial Action)
                            action_val = action[0] if isinstance(action, (np.ndarray, list)) else action
                            stop_id = state_dict[line_id][bus_id][0][station_feature_idx]
                            sim_time = state_dict[line_id][bus_id][0][-1]
                            fwd_h = state_dict[line_id][bus_id][0][5]
                            bwd_h = state_dict[line_id][bus_id][0][6]
                            log_entry = f"{line_id},{bus_id},{step},{stop_id},{action_val:.4f},{fwd_h:.2f},{bwd_h:.2f}\n"
                            action_log_buffer.append(log_entry)
                            
                            if debug:
                                # Format: From Algorithm run, Bus id: ... , station id is: ... , current time is: ... , action is: ... , reward: ...
                                # print(f'From Algorithm run, Bus id: {line_id}_{bus_id} , station id is: {int(stop_id)} , current time is: {sim_time:.1f} , action is: {action_val:.4f}, reward: {get_reward_value(reward_dict, line_id, bus_id):.4f}')
                                # print(f"DEBUG_SET: Bus {line_id}_{bus_id} key={line_id},{bus_id} Val={action} DictVal={action_dict.get(line_id, {}).get(bus_id)}")
                                pass
            
            # DEBUG: Trace action_dict before step
            if debug:
                # for line_id, buses in action_dict.items():
                #     for bus_id, action in buses.items():
                #         if action is not None:
                #             print(f"DEBUG_TRACE Pre-Step: Bus {line_id}_{bus_id} Action={action}")
                pass

            state_dict, reward_dict, done = safe_step(env, action_dict, render=render)

            # Flush logs periodically (e.g., every 1000 steps)
            if step % 1000 == 0:
                if action_log_buffer:
                    with open("action_log.csv", "a") as f:
                        f.writelines(action_log_buffer)
                    action_log_buffer = []
                if trajectory_log_buffer:
                    with open("trajectory_log.csv", "a") as f:
                        f.writelines(trajectory_log_buffer)
                    trajectory_log_buffer = []
                if len(replay_buffer) > args.batch_size and len(replay_buffer) % args.training_freq == 0 and step_trained != step:
                    step_trained = step
                    for i in range(update_itr):
                        _ = sac_trainer.update(args.batch_size, training_steps, reward_scale=10., auto_entropy=args.auto_entropy, target_entropy=-1. * action_dim)
                        training_steps += 1

                if done:
                    if args.use_reward_scaling:
                        sac_trainer.reward_scaling.reset()
                    replay_buffer.last_episode_step = episode_steps
                    break
            # 计算每个 episode 的平均 Q 值
            rewards.append(episode_reward)
            if training_steps > 0:
                q_values_episode.append(np.mean(q_values[-training_steps:]))
                reg_norms1_episode.append(np.mean(reg_norms1[-training_steps:]))
                reg_norms2_episode.append(np.mean(reg_norms2[-training_steps:]))
                log_probs_episode.append(np.mean(log_probs[-training_steps:]))
                alpha_values_episode.append(np.mean(alpha_values[-training_steps:]))
            else:
                q_values_episode.append(0.0)
                reg_norms1_episode.append(0.0)
                reg_norms2_episode.append(0.0)
                log_probs_episode.append(0.0)
                alpha_values_episode.append(0.0)

            # Print Episode Summary (Matching original format)
            replay_buffer_usage = len(replay_buffer) / replay_buffer_size * 100
            print(
                f"[SAC | max_alpha={args.maximum_alpha}] Episode: {eps} | Episode Reward: {episode_reward:.2f} "
                f"| CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | "
                f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | "
                f"Replay Buffer Usage: {replay_buffer_usage:.2f}% | "
                f"Avg Q-Value: {q_values_episode[-1]:.2f}")

            if eps % args.plot_freq == 0:  # plot and model saving interval
                if not os.path.exists(LOG_DIR):
                    os.makedirs(LOG_DIR, exist_ok=True)

                plot(rewards)
                np.save(os.path.join(LOG_DIR, 'rewards.npy'), rewards)
                np.save(os.path.join(LOG_DIR, 'q_values.npy'), q_values_episode)
                np.save(os.path.join(LOG_DIR, 'reg_norms1_episode.npy'), reg_norms1_episode)
                np.save(os.path.join(LOG_DIR, 'reg_norms2_episode.npy'), reg_norms2_episode)
                np.save(os.path.join(LOG_DIR, 'log_probs_episode.npy'), log_probs_episode)
                np.save(os.path.join(LOG_DIR, 'alpha_values_episode.npy'), alpha_values_episode)
                
                # 评估当前策略
                mean_reward, reward_std = evaluate_policy(sac_trainer, env, num_eval_episodes=0, deterministic=True)
                # print(f"评估结果 (Episode {eps}): 平均奖励 = {mean_reward:.2f}, 标准差 = {reward_std:.2f}")
                
                # 记录评估结果
                eval_episodes.append(eps)
                eval_mean_rewards.append(mean_reward)
                eval_reward_stds.append(reward_std)
                
                # 保存评估结果
                np.save(os.path.join(LOG_DIR, 'eval_episodes.npy'), eval_episodes)
                np.save(os.path.join(LOG_DIR, 'eval_mean_rewards.npy'), eval_mean_rewards)
                np.save(os.path.join(LOG_DIR, 'eval_reward_stds.npy'), eval_reward_stds)
                
                # 保存带有episode信息的模型
                model_name = f"{model_path}_episode_{eps}"
                sac_trainer.save_model(os.path.join(LOG_DIR, f'sac_v2_episode_{eps}'))
                sac_trainer.save_model(model_name)
                # snapshot = tracemalloc.take_snapshot()
                # for stat in snapshot.statistics('lineno')[:10]:
                #     print(stat)  # 显示内存占用最大的10行
            replay_buffer_usage = len(replay_buffer) / replay_buffer_size * 100
          # 在训练结束时保存完整模型（包括critic和actor）
        sac_trainer.save_model(model_path)
        
        # 评估最终策略
        mean_reward, reward_std = evaluate_policy(sac_trainer, env, num_eval_episodes=0, deterministic=True)
        # print(f"最终评估结果: 平均奖励 = {mean_reward:.2f}, 标准差 = {reward_std:.2f}")
        
        # 记录最终评估结果
        final_eval_episode = args.max_episodes - 1
        eval_episodes.append(final_eval_episode)
        eval_mean_rewards.append(mean_reward)
        eval_reward_stds.append(reward_std)
        
        # 保存最终评估结果
        final_log_dir = LOG_DIR
        if not os.path.exists(final_log_dir):
            os.makedirs(final_log_dir, exist_ok=True)

        np.save(os.path.join(final_log_dir, 'rewards.npy'), rewards)
        np.save(os.path.join(final_log_dir, 'q_values.npy'), q_values_episode)
        np.save(os.path.join(final_log_dir, 'reg_norms1_episode.npy'), reg_norms1_episode)
        np.save(os.path.join(final_log_dir, 'reg_norms2_episode.npy'), reg_norms2_episode)
        np.save(os.path.join(final_log_dir, 'log_probs_episode.npy'), log_probs_episode)
        np.save(os.path.join(final_log_dir, 'alpha_values_episode.npy'), alpha_values_episode)

        np.save(os.path.join(final_log_dir, 'eval_episodes.npy'), eval_episodes)
        np.save(os.path.join(final_log_dir, 'eval_mean_rewards.npy'), eval_mean_rewards)
        np.save(os.path.join(final_log_dir, 'eval_reward_stds.npy'), eval_reward_stds)
        
        # 保存带有最终episode信息的模型
        final_model_name = f"{model_path}_episode_final"
        sac_trainer.save_model(os.path.join(final_log_dir, 'sac_v2_episode_final'))
        sac_trainer.save_model(final_model_name)

    if args.test:
        sac_trainer.policy_net.load_state_dict(torch.load(model_path))
        for eps in range(10):

            done = False
            env.reset()
            episode_reward = 0
            action_dict = build_action_template(state_dict)
            station_feature_idx = sac_trainer.station_feature_idx if sac_trainer.station_feature_idx is not None else 1

            while not done:
                action_dict = build_action_template(state_dict, action_dict)
                for line_id, buses in state_dict.items():
                    for bus_id, history in buses.items():
                        if len(history) == 0:
                            continue
                        if len(history) == 1:
                            if action_dict[line_id][bus_id] is None:
                                state_input = np.array(history[0])
                                action = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                                action_dict[line_id][bus_id] = action
                        elif len(history) >= 2:
                            if history[0][station_feature_idx] != history[1][station_feature_idx]:
                                episode_reward += get_reward_value(reward_dict, line_id, bus_id)
                            state_dict[line_id][bus_id] = history[1:]
                            state_input = np.array(state_dict[line_id][bus_id][0])
                            action_dict[line_id][bus_id] = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)

                state_dict, reward_dict, done = safe_step(env, action_dict)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
