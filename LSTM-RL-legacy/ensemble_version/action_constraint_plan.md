#### 方案 A：单标量映射（最简单，彻底杜绝冲突）
将 Actor 的输出从 2 维缩减为 1 维（标量 $a \in [-1, 1]$）。

*   **逻辑**：
    *   当 $a > 0$：代表“需要延迟”，映射到 $Holding = a \times MaxH$，$Speed = 1.0$。
    *   当 $a < 0$：代表“需要赶路”，映射到 $Holding = 0$，$Speed = 1.0 + |a| \times (MaxV - 1.0)$。
    *   当 $a \approx 0$：维持原状。
*   **代码实现思路**：
    ```python
    # 在 Actor 输出后进行转换
    raw_action = self.actor(state) # 输出范围 [-1, 1]
    if raw_action > 0:
        holding = raw_action * max_holding
        speed_factor = 1.0
    else:
        holding = 0
        speed_factor = 1.0 + abs(raw_action) * (max_speed - 1.0)
    ```
*   **效果**：这种方法让 Agent 只需要纠结“我要快还是慢”，而不需要纠结“我要用什么手段变快”。

### 2. 一个致命的陷阱：Action Correction（动作纠正）

在使用映射法时，有一个细节决定了你能不能跑赢 Baseline。

**问题**：SAC 是 Off-policy 算法。如果网络输出的是 $a_{raw}$（冲突动作），但你实际执行的是 $a_{mapped}$（修正动作），你应该存哪个到 Replay Buffer 里？

*   **正确做法**：存入 **$a_{mapped}$**。
*   **原因**：你要让 Critic 知道，当 Agent 在某个状态下给出那个“愚蠢”的原始输出时，最终环境反馈的 Reward 其实是来自于修正后的动作。这会引导 Actor 的输出梯度自动向修正后的区域靠拢。
*   **公式推导支持**：这在 RL 中被称为 **"Action Squashing"**。如果你存入错误的动作，会导致 **Policy Evaluation Error**，模型会发现“我明明输出了加速+Holding，为什么 Reward 还挺高？”，从而继续产生冲突动作。

### 4. 为什么映射法能跑赢 Hard Rule？

你的 Hard Rule 通常是**二值化**的（或者是分段线性的）。
而映射法下的 SAC 可以学习到：
*   “虽然晚点了，但我只需要增加 2% 的速度就能追上，不需要加到 MaxSpeed（节省能耗）。”
*   “虽然早到了，但我可以通过 Holding 5秒 + 减速 5% 的组合，让乘客体感最舒适。”

**建议尝试**：先用 **方案 A (1D 映射)** 跑一下。如果 1D 映射的性能能追平 Hard Rule，说明你的问题核心就在于动作冗余；如果 1D 映射还打不过，那可能问题出在 Reward 函数对 Headway 的敏感度不够上。