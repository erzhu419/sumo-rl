# Project Roadmap: H2O+ Integration with Snapshot-based Buffer Reset
基于对 @[H2Oplus] 代码库、@[LSTM-RL/env/sim.py] 以及 @[SUMO_ruiguang] 的分析

**任务目标**: 实现一套基于 H2O+ 的 Sim-to-Real 强化学习框架，利用低保真仿真器 (`LSTM-RL/env/sim.py`) 和少量真实路网数据 (`SUMO_ruiguang`) 训练鲁棒策略。

**利用 `H2O+` 框架，结合：**
1.  **Offline Data**: 来自复杂的 `SUMO_ruiguang` (视为 "Real World")。
2.  **Online Sim**: 来自简化的 `LSTM-RL/env/sim.py` (视为 "Dynamics Gap Simulator")。
3.  **Algorithm**: 使用 H2O+ (或 CQL) 进行混合训练。

**核心机制**:
1.  **上帝模式重置 (God-Mode Reset)**: 利用真实数据的全局快照 (Snapshot) 强制重置仿真器状态，消除 `LSTM-RL/env`的时间漂移(随着仿真运行，`sim.py`产出的数据分布离`SUMO_ruiguang`产出的越来越远)。
2.  **上下文感知判别器 (Context-Aware Discriminator)**: 利用全局快照提取宏观交通特征 (Context)，帮助判别器精准识别 Sim 与 Real 的分布差异。

---

## Phase 0: 核心数据协议定义 (Core Data Protocols)

**执行优先**: 在编写任何环境代码前，必须先定义好共享的数据结构和工具函数。建议新建文件 `common/data_utils.py`。

### 0.1 快照数据结构 (Snapshot Schema)
`LSTM-RL/env/sim.py` 和 `SUMO_ruiguang` 必须生成**结构完全一致**的字典，用于描述某一时刻的全局系统状态,每一次 "公交到站决策事件"，产生一个snapshot，包括一个传统RL用的tuple，用于训练offline RL，以及包含其他所有车辆当前时刻信息的infors。
这个结构代表了我们对客观世界状态的最佳估计 (Best Estimate)。在 SUMO 实验中直接取真值；在实车部署中，这代表经过清洗和推断后的数据。

```python
# Type Definition in pseudo-code
Transition = {
    # --- Part A: 微观 RL 数据 (Standard Offline RL) ---
    # 描述 "Ego Bus" (当前触发决策的车辆) 的状态和行为
    "global_time":float,     # 以"事件"为核心的索引
    "obs": np.ndarray,       # Shape: (state_dim,)，若state已经为最后一站则删除该条数据
    "action": np.ndarray,    # Shape: (action_dim,), 执行的驻站时间，由算法返回得到，非控制状态为0
    "reward": float,         # 单步奖励，用LSTM-RL/env/bus.py中相似的计算方式计算得到
    "next_obs": np.ndarray,  # 需要暂时空置，待车辆到达下一站才获取到
    "terminal": bool,        # 是否结束

    # --- Part B: 宏观快照 (For Reset & Discriminator) ---
    # 描述决策时刻的全系统物理状态
    "infos": {
        "snapshot": {
            "global_time": float,   # 绝对仿真时间
            "ego_bus_id": str,      # 标记谁是当前决策的主角
            
            # 全网车辆列表 (包含 Ego Bus 和 背景车辆)
            "all_buses": [
                {
                    "id": str,
                    "pos": float,   # 距线路起点的绝对距离 (m) -> 核心对齐字段。"pos 必须是归一化到线路起点的线性距离，且需处理环线或折返线的坐标跳变。参考LSTM-RL/env/bus.py中的absolute_distance的定义方式"
                    "speed": float, # 当前速度 (m/s)
                    'last_stop_index': int, # 刚经过的站点索引 (0~N)
                    'ratio_to_next': float, # 距离下一站的进度百分比 (0.0~1.0)
                    "load": int,    # 车内人数
                    "is_ego": bool  # 方便快速索引
                },
                # ... 必须包含路网上所有活跃车辆
            ],
            
            # 全网站点列表(在真实中用推断值)
            "all_stations": [
                {"id": str, 
                "index": int,
                "waiting_count": int, # 站台滞留人数 (推断值)
                "arrival_rate":float # (可选) 该时段的历史平均到达率，辅助Sim生成新客流
                },
                # ...
            ]
        }
    }
}
```

### 0.2 上下文提取函数 (Context Extractor)
为了保留空间信息，我们将路网离散化为K个空间段 (Segments)，生成一个矩阵作为 Discriminator 的输入，而不是几个标量。

```python
def extract_structured_context(snapshot: dict, num_segments=10) -> np.ndarray:
    """
    将线性路网切分为 num_segments 个段，计算每一段的交通指纹。
    Input: SnapshotDict
    Output: Flat Vector, shape = (num_segments * 3, )
            包含 [速度分布, 车辆密度分布, 拥堵分布]
    """
    total_length = ROUTE_LENGTH # 假设已知线路总长
    segment_len = total_length / num_segments
    
    # 初始化 buckets
    seg_speeds = [[] for _ in range(num_segments)]
    seg_counts = [0] * num_segments
    seg_loads  = [0] * num_segments
    
    # 1. 将车辆映射到空间段中 (Spatial Binning)
    for bus in snapshot['all_buses']:
        # 计算该车在哪一段 (0 ~ K-1)
        idx = int(bus['pos'] / segment_len)
        idx = min(idx, num_segments - 1)
        
        seg_speeds[idx].append(bus['speed'])
        seg_counts[idx] += 1
        seg_loads[idx] += bus['load']
        
    # 2. 计算统计特征
    # 特征 1: 速度分布 (SimBus 通常均匀，Real 在红绿灯路段低)
    vec_speed = np.array([np.mean(s) if s else 30.0 for s in seg_speeds]) / 30.0
    
    # 特征 2: 密度分布 (哪里车多)
    vec_density = np.array(seg_counts) / 5.0 # 归一化
    
    # 特征 3: 站点拥挤度 (映射 waiting 到空间段)
    seg_waiting = [0] * num_segments
    for st in snapshot['all_stations']:
        # 假设我们知道站点的位置 station_pos
        # idx = int(st['pos'] / segment_len) ...
        # 这里简化处理，实际需查表
        pass 
    vec_waiting = np.array(seg_waiting) / 20.0
    
    # 3. 拼接并展平
    # 判别器将看到一个长度为 30 的向量，清晰地描述了整条路哪里快、哪里慢、哪里人多
    z = np.concatenate([vec_speed, vec_density, vec_waiting])
    return z.astype(np.float32)
```

**新方案** : 输入 30 个数（空间图谱）。判别器能判断 “为什么第 3 段（十字路口）没有降速？” 或者 “为什么第 5 段（学校）没有很多人？”
结果: 这极大地增强了 H2O+ 区分 Sim/Real 的能力，迫使 Policy 在那些 SimBus 模拟不准的“特征路段”（红绿灯、拥堵点）更加保守，而在路况简单的路段更加自信。

---

## Phase 1: 真实环境封装与数据采集 (Real World - /SUMO_ruiguang)

**目标**: 将 `SUMO_ruiguang` 封装为 Gym 环境，并采集包含 Snapshot 的全量数据。

### 1.1 `SumoGymWrapper` 实现

*   **文件**: `envs/sumo_wrapper.py`
*   **现状**: `rl_env.py` 是 Event-driven 的，返回字典结构状态。
*   **基类**: `gym.Env`
*   **策略**: 直接利用现有的高效 `rl_bridge.py` 逻辑。该 Bridge 已经实现了 `Libsumo` 加速和事件队列管理，性能优异。
*   **关键点**: Wrapper 内部维护一个 `while` 循环调用 `bridge.fetch_events()`，直到获得下一个 Action 请求，从而适配 Gym 的 `step()` 阻塞式接口。
*   **机制**: env.step(action) 不是前进固定的 1 秒，而是前进直到任意一辆公交车到站。
```python
def step(self, action):
    # 1. 对上一辆 Ego Bus 应用 Action (设定驻站)
    self._apply_hold(self.last_ego_id, action)
    
    # 2. 仿真循环，直到下一辆车需要决策
    while True:
        traci.simulationStep()
        arrived_bus = self._check_any_bus_arrival()
        if arrived_bus:
            self.last_ego_id = arrived_bus
            return self._get_state(arrived_bus), ...
```

### 1.2 `collect_data_sumo.py` 实现

### 1.2.1 异步数据采集与时序对齐 (Asynchronous Data Alignment)

**核心挑战**: 公交系统不同于标准 Gym 环境，动作 ($a_t$) 与下一状态 ($s_{t+1}$) 之间存在较长的物理时间延迟，且不同车辆是异步到达的。
**解决方案**: 采用 **"Pending Cache (挂起缓存)"** 机制，跨时间步拼接完整的 Transition 元组。可以参考`sac_v2_bus.py`或 `sac_v2_bus_SUMO.py`中使用state_dict和action_dict并在state_dict的长度凑足后才存入buffer的办法。

#### A. 时序逻辑定义
一条用于 Offline RL 和 H2O+ 的标准数据 `(s, a, r, s', snapshot)` 必须由两个时间点的事件共同构成：

*   **时刻 $T_1$ (上一站)**: 车辆到达站点 $k$。
    *   产生: `current_obs` ($s$), `action` ($a$), **`snapshot` ($Snapshot_{T1}$)**。
    *   动作: **暂存 (Cache)** 这些数据，因为此时不知道奖励和下一状态。
    *   *注意: Snapshot 必须捕获 $T_1$ 时刻的全网状态，用于后续 Reset 回到做出决策的那一刻。*
*   **时刻 $T_2$ (当前站)**: 车辆到达站点 $k+1$。
    *   产生: `next_obs` ($s'$), `reward` ($r$)。
    *   动作: **结算 (Settle)** 上一站的缓存，生成完整 Tuple 并存入 Buffer。

#### B. `collect_data_sumo.py` 核心实现逻辑

```python
# 缓存字典: Key=VehicleID, Value=Dict(上一次决策时的上下文)
pending_transitions = {} 

def on_simulation_step():
    # 获取当前仿真步内所有完成靠站、需要决策的车辆列表
    arrived_buses = sumo_env.get_arrived_buses()
    
    for bus_id in arrived_buses:
        # 1. 获取当前时刻(T2)的状态 -> 作为 s'
        current_obs = sumo_env.get_state(bus_id)
        current_time = sumo_env.get_time()
        
        # --- [结算逻辑] ---
        # 如果该车在上一站有未结单的决策
        if bus_id in pending_transitions:
            prev_data = pending_transitions.pop(bus_id)
            
            # 计算延迟奖励 r (根据 T1 的预测和 T2 的实际情况)
            # 例如: reward = -(current_headway_variance)
            reward = calculate_reward(prev_data['obs'], current_obs) #复用LSTM-RL/env/bus.py中的reward计算方式一样
            
            # 存入 Replay Buffer
            # 关键: info['snapshot'] 必须是 prev_data['snapshot'] (T1时刻的快照)
            replay_buffer.add(
                obs=prev_data['obs'],
                action=prev_data['action'],
                reward=reward,
                next_obs=current_obs,
                terminal=False,
                infos={'snapshot': prev_data['snapshot']} 
            )
            
        # --- [新开单逻辑] ---
        # 如果车辆未到达终点，需要进行下一次决策
        if not is_route_end(bus_id):
            # 1. 立即捕获当前时刻(T2)的全网快照 -> 作为下一次的 snapshot
            # 注意: 必须标记当前的 bus_id 为 ego_vehicle
            snapshot_now = capture_full_system_snapshot(ego_id=bus_id)
            
            # 2. 决策
            action = behavior_policy.get_action(current_obs)
            sumo_env.apply_action(bus_id, action)
            
            # 3. 存入缓存，等待下一次到达
            pending_transitions[bus_id] = {
                'obs': current_obs,
                'action': action,
                'snapshot': snapshot_now,
                'timestamp': current_time
            }
```

#### C. 对 H2O+ 的影响
这种采集方式确保了：
1.  **Reset 有效性**: 当我们用 Buffer 中的 `snapshot` 重置 SimBus 时，SimBus 会回到 $T_1$ 时刻。此时 Agent 看到的观测值正是 $s$，它可以尝试输出一个新的动作 $a'$，从而产生新的反事实轨迹。
2.  **Critic 训练正确性**: $r$ 和 $s'$ 真实反映了在真实动力学下，执行 $a$ 后的结果。

*   **输出**: `datasets/sumo_offline_full.hdf5` (或 pickle)。

#### 验证：通过训练offline RL,用collect_data_sumo.py和SumoGymWrapper采集的数据训练offline RL，并通过收敛性确认以上工作的完成情况。
---

## Phase 2: 仿真环境改造 (Sim World - /LSTM-RL/env)

**目标**: 改造 `LSTM-RL/env`中的`sim.py`/`bus.py`等依赖，使其支持多线路仿真，和并使其支持“写入快照(Step)”和“读取快照(Reset)”。

**动机**: 为了减小 Dynamics Gap，必须确保 `LSTM-RL/env` (Sim) 的基础数据和运行特性与 `SUMO_ruiguang` (Real) 高度一致。

#### 2.1 静态数据对齐 (Static Data Alignment)
-   **现状差异**: `SUMO` 使用 XML 定义路网和时刻表，`LSTM-RL` 使用 Excel (`stop_news.xlsx`, `time_table.xlsx` 等)。
-   **校准策略**: 编写数据转换脚本 `xml_to_excel_converter.py`。
    -   解析 `SUMO_ruiguang` 的 `save_obj_*.xml` 文件。
    -   生成 `LSTM-RL/env` 所需的 Excel 格式。
    -   **关键点**: 站点 ID、线路 ID、站点距离 (`distance`)、发车时间 (`launch_time`) 必须一一对应。确保 `LSTM-RL/env/sim.py` 读取的 `route_length` 和 `station_positions` 与 `SUMO_ruiguang`中的 XML 文件中的定义**误差 < 1%**。如果路网长度不对，位置映射就会失效。其余`LSTM-RL/env`中的客流/路况(速度)尽可能贴合`SUMO_ruiguang`中的.

#### 2.2 动力学参数 (Dynamics)
-   **现状差异**: `SUMO` 有完整的信号灯相位逻辑 (`Signal` 类)，`LSTM-RL` 仅有简单的 `Route` 和 `V_max`，无信号灯实体。
-   **校准策略 (简化版)**: 不在 `LSTM-RL` 中重写复杂的信号灯逻辑，而是通过**等效降速**来模拟。
    -   **统计**: 从 `SUMO` 历史运行数据中统计每条 Edge 的**平均通过时间** (包含红灯等待)。
    -   **应用**: 将该等效平均速度填入 `LSTM-RL` 的 `route_news.xlsx` 中的速度限制或参数中。
    -   **复用机制**: `LSTM-RL` 的 Bus 复用机制与 `SUMO` 的 Trip 机制不同，这影响不大，只要确保同一时刻在线车辆的行为符合物理规律即可。在生成 Gym Wrapper 时，根据 `trip_id` 唯一标识 Agent。


### 2.3 `BusSimEnv` 接口改造
*   **文件**: `LSTM-RL/env`
*   **需求 A: 输出快照 (Symmetry for Discriminator)**
    *   在 `step()` 返回的 `info` 字典中，必须调用 `self._build_snapshot()`，返回符合 **Phase 0 定义** 的 `SnapshotDict`。
    
*   **需求 B: 快照重置 (Reset Mechanism)**
    *   实现 `reset(snapshot=None)` 接口。
    ```python
    def reset(self, snapshot: dict = None):
        self.cleanup() # 清空环境
        
        if snapshot is None:
            return self._reset_standard() # t=0 发车
        else:
            # 时光倒流
            self.current_time = snapshot['global_time']
            
            # 重建物理实体
            for b_data in snapshot['all_buses']:
                new_bus = self.spawn_bus(
                    id=b_data['id'],
                    pos=b_data['pos'], # 映射回 SimBus 坐标
                    speed=b_data['speed'],
                    load=b_data['load']
                )
            
            # 恢复站点
            for s_data in snapshot['all_stations']:
                self.stations[s_data['id']].queue = s_data['waiting_count']
            self.set_passenger_arrival_rate(station_id, station_data['arrival_rate'])
            # 关键：寻找 snapshot['ego_bus_id'] 指向的那辆车
            # 并返回它的 State 作为初始 Obs
            return self._get_bus_state(snapshot['ego_bus_id'])
    ```
*   **验证**: 通过训练online RL,用之前已收敛的`sac_v2_bus.py`或`sac_v2_bus_SUMO.py`在改造后的`LSTM-RL/env`上训练,以验证该部分改造工作的正确性。
---

## Phase 3: H2O+ 算法深度集成 (H2O+ Integration)

**目标**: 修改 H2O+ 训练循环，实现混合重置与加权更新。

### 3.1 缓冲区管理 (Buffer Management)
*   `OfflineBuffer`: 加载 `datasets/sumo_offline_full.hdf5`。数据不可变。
    *   **预处理**: 在加载时，对所有数据调用 `extract_context(info['snapshot'])`，生成 `info` 向量并缓存。
*   `OnlineBuffer`: 这里的 Sim 数据是动态生成的。
    *   **运行时**: 每次 `env.step()` 后，调用 `extract_context(info['snapshot'])` 得到 `info`，存入 Buffer。

### 3.2 混合重置训练循环 (Training Loop)
*   **文件**: `algorithms/h2o_plus/train.py`
*   **伪代码逻辑**:
    ```python
    # 初始化
    offline_buffer.load("sumo_offline_full.pkl")
    online_buffer = ReplayBuffer()

    for episode in range(MAX_EPISODES):
        # --- A. 混合重置策略 ---
        if random.random() < P_RESET (e.g., 0.5):
            # Mode: Buffer Reset (解决 Drift)
            real_batch = offline_buffer.sample(1)
            snapshot = real_batch['infos']['snapshot'][0]
            obs = sim_env.reset(snapshot=snapshot)
            max_steps = H_ROLLOUT (e.g., 20) # 短程修补
        else:
            # Mode: Standard (全程规划)
            obs = sim_env.reset(snapshot=None)
            max_steps = FULL_LENGTH
            
        # --- B. 数据交互与收集 ---
        for t in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, info = sim_env.step(action)
            
            # 实时提取 Sim 的宏观特征
            z_sim = extract_context(info['snapshot'])
            
            # 存入 Online Buffer
            online_buffer.add(obs, action, reward, next_obs, done, z=z_sim)
            obs = next_obs
            
            # --- C. H2O+ 更新步骤 (每步或每N步) ---
            if ready_to_train:
                # 1. 采样
                batch_real = offline_buffer.sample(BATCH_SIZE) # 带 z_real
                batch_sim = online_buffer.sample(BATCH_SIZE)   # 带 z_sim
                
                # 2. 训练 Discriminator
                d_loss = update_discriminator(batch_real, batch_sim)
                
                # 3. 计算 Importance Weight
                # w = P_real / P_sim ≈ exp(logit_real - logit_sim) 
                # 具体公式参考 H2O 论文，通常使用 sigmoid 输出处理
                w = compute_weights(discriminator, batch_sim)
                
                # 4. 训练 Critic (带权重)
                # Loss = w * (Q - Target_Q)^2 + (1-w) * (Q - Target_Q_Conservative)^2 (可选保守项)
                # 简化版: Loss = w * (Q_sim - Target)^2 + (Q_real - Target)^2
                c_loss = update_critic(batch_real, batch_sim, w)
                
                # 5. 训练 Actor (标准 SAC)
                a_loss = update_actor(batch_sim) # 通常只在 Sim 数据上更新 Actor
    ```

### 3.3 判别器与 Critic 更新 (Update Logic)
*   **Discriminator**:
    *   输入: `state`, `action`, `context_z` (可能还需要 `next_state`)。
    *   Loss: 区分 `(s, a, z_real)` 和 `(s, a, z_sim)`。
    *   *预期*: 如果 LSTM-RL/env/sim.py 跑出的 `z_sim` (例如所有车都不堵) 与 `z_real` 差异大，判别器输出 `w -> 0`。
*   **Critic**:
    *   使用 $w \cdot (Q - \mathcal{T}Q)^2$ 进行加权更新。

---

## Phase 4: 验证清单 (Definition of Done)

在提交论文或大规模跑实验前，请按顺序验证以下三点：

1.  **接口一致性验证**: 打印 `SUMO_wrapper.observation_space` 和 `BusSimEnv.observation_space`，必须完全相同。
2.  **重置有效性验证**:
    *   从 Offline Data 拿一个 Snapshot（例如 t=3600s, 3辆车）。
    *   调用 `sim_env.reset(snapshot)`。
    *   检查 `sim_env` 内部是否真的生成了3辆车，且位置、速度、时间完全一致。
3.  **判别器敏感度验证**:
    *   手动构造一个“极度顺畅”的 Snapshot (Sim特征) 和一个“极度拥堵”的 Snapshot (Real特征)。
    *   检查 Discriminator 是否能给出显著不同的分数。
#### Structured Contex & Buffer Reset消融实验
    分别用简单的mean/var only contex对比structured contex，以及buffer reset vs 直接用sim的策略在SUMO跑做评价。
---
## Appendix A: 上下文感知判别器 (Context-Aware Discriminator)

**定位**: 替代 Phase 3.1 中简单的 MLP 判别器。
**核心思想**: 放弃手动提取均值/方差 ($z$)，改用 **Ego-Centric Attention (以自我为中心的注意力机制)** 自动学习 Ego Bus 与背景交通流的时空依赖关系。

### A.1 网络架构设计
该架构利用 Transformer Block 处理变长的车辆集合，具有**置换不变性 (Permutation Invariance)** 和 **距离敏感性**。

*   **Input**:
    *   `ego_feat`: 当前决策车辆特征 `(batch, state_dim)`
    *   `context_feat`: 背景车辆特征序列 `(batch, max_buses, state_dim)` (需 Padding)
    *   `mask`: 掩码矩阵，标记 Padding 的位置
*   **Architecture**:
    1.  **Embedding**: 将物理特征映射为高维隐向量。
    2.  **Cross-Attention**:
        *   **Query**: Ego Embedding
        *   **Key/Value**: Context Embedding (包含 Ego 自身)
        *   *物理含义*: 自动关注前车、后车以及拥堵路段的车辆，忽略远端无关车辆。
    3.  **Classifier**: 输出 Logit。

### A.2 PyTorch 参考实现

```python
import torch
import torch.nn as nn

class EgoCentricDiscriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        
        # 1. Feature Embedding (共享权重)
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Attention Block (核心)
        # batch_first=True: (Batch, Seq, Feature)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. Binary Classifier
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output Logit
        )

    def forward(self, ego_state, other_states, mask=None):
        """
        Args:
            ego_state: (B, Dim)
            other_states: (B, N_others, Dim)
            mask: (B, 1+N_others) Boolean mask, True for padding
        """
        # [B, 1, H]
        q_emb = self.embedding(ego_state).unsqueeze(1)
        
        # [B, N, H]
        k_v_emb = self.embedding(other_states)
        
        # 拼接 Ego 和 Others 作为 Key/Value 源 (Ego 也需要关注自己)
        # [B, 1+N, H]
        seq_emb = torch.cat([q_emb, k_v_emb], dim=1)
        
        # Attention Calculation
        # Query 只有 Ego，输出 Context Vector z
        # attn_output: [B, 1, H]
        z, _ = self.attention(
            query=q_emb, 
            key=seq_emb, 
            value=seq_emb, 
            key_padding_mask=mask
        )
        
        # Classification
        return self.head(z.squeeze(1))
```

---

## Appendix B: 递归状态估计器 (Recursive State Estimator)

**定位**: 解决 Phase 0 中真实世界数据缺失（Unknown Waiting/Load）的问题。
**核心思想**: 利用 **卡尔曼滤波 (Kalman Filter)** 的“预测-修正”思想，维护全网站点的人数信念状态 (Belief State)，而无需每次重新计算积分。

### B.1 数学模型
*   **状态向量 ($X_t$)**: 全网站点当前滞留人数向量。
*   **预测步 (Predict)**: $X_{t+\Delta t} = X_t + \lambda \cdot \Delta t$
    *   $\lambda$: 历史平均到达率向量 (Arrival Rate)。
*   **修正步 (Correct)**: $X_{new} = \max(0, X_{pred} - u_{board})$
    *   $u_{board}$: 推测上车人数。

### B.2 状态估计器实现 (Python)

此模块应集成在 `BusSimEnv` 和 `collect_data_sumo.py` 中，作为数据预处理层。

```python
import numpy as np

class TrafficStateEstimator:
    def __init__(self, station_ids, arrival_rates):
        """
        Args:
            station_ids: 站点ID列表
            arrival_rates: Dict {station_id: rate_per_second} (来自历史统计)
        """
        self.station_map = {sid: i for i, sid in enumerate(station_ids)}
        self.rates = np.array([arrival_rates[sid] for sid in station_ids])
        
        # 信念状态 (Belief State)
        self.queues = np.zeros(len(station_ids))
        self.last_update_time = 0.0

    def predict_until(self, current_time):
        """
        [积分步]：根据流逝时间推演队列增长
        """
        dt = current_time - self.last_update_time
        if dt > 0:
            # 简单泊松过程期望：Rate * Time
            # 进阶：可在此处加入高斯噪声模拟不确定性
            self.queues += self.rates * dt
            self.last_update_time = current_time

    def correct_on_arrival(self, station_id, estimated_board):
        """
        [修正步]：车辆到站带走乘客
        Args:
            estimated_board: 
                - SUMO中: 直接读取真值
                - Real World: (Dwell_Time - Dead_Time) / Time_Per_Person
        """
        idx = self.station_map[station_id]
        
        # 1. 先同步到当前时刻
        # self.predict_until(now) # 需传入当前时间
        
        # 2. 修正状态 (人数不能为负)
        self.queues[idx] = max(0.0, self.queues[idx] - estimated_board)
        
        return self.queues[idx]

    def get_snapshot_state(self):
        """返回用于构建 Snapshot 的数据"""
        return self.queues.copy()
```

### B.3 鲁棒性增强策略 (Robustness Strategy)

为了防止推测误差导致 H2O+ 训练崩塌，在 **Phase 3 (训练循环)** 中建议加入以下策略：

1.  **噪声重置 (Noisy Reset)**:
    在执行 `sim_env.reset(snapshot)` 时，不要精准还原推测出的 `waiting` 或 `load`，而是注入噪声：
    ```python
    # 在 BusSimEnv._reset_from_snapshot 中
    noise = np.random.normal(0, 0.15) # 15% 的估计误差
    bus.load = int(data['load'] * (1 + noise))
    station.queue = int(data['waiting'] * (1 + noise))
    ```
    *目的*: 迫使 Policy 学会在状态估计不准的情况下依然表现良好 (Domain Randomization 思想)。

2.  **特征模糊化**:
    Discriminator 训练时，可以对 Context Feature 中的 `waiting_count` 维度进行 Dropout 或添加噪声，降低判别器对这一“不可靠特征”的依赖权重。