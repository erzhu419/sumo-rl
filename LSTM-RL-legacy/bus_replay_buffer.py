import numpy as np
import random

# 用字典作为replaybuffer的数据结构，提高性能
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
