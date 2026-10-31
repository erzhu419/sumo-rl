# JTT-Inspired Targeted Snapshot Reset for H2O+ Bus Training

## Core Insight: JTT ↔ H2O+ Mapping

**JTT (supervised learning)**:
1. **Phase 1 (Identify)**: Train ERM model for T epochs → find **error set** E (samples misclassified)
2. **Phase 2 (Upweight)**: Retrain from scratch, upweight samples in E by λ

**H2O+ Bus (Sim2Real RL)**:
1. **Phase 1 (Identify)**: Train baseline policy → find **fragile transitions** F (high TD error | high Q-disagreement | high sim-drift)
2. **Phase 2 (Reset & Focus)**: Continue training with **targeted snapshot resets** to states in F, forcing the sim to generate more data from exactly those fragile regions

> [!IMPORTANT]
> The key innovation: In supervised learning (JTT), you can only **upweight** existing samples.
> In RL with snapshot reset, we can **generate entirely new counterfactual trajectories** by
> resetting the simulator to the exact fragile state and rolling out with the updated policy.
> This is strictly more powerful than JTT's upweighting.

## What Makes a Transition "Fragile"?

In JTT, the error set is simply `y_pred ≠ y_true`. In RL, we don't have labels.
We need proxy signals for **where the policy is most vulnerable**:

### Priority Score: `p(i) = α·TD_err(i) + β·Q_disagree(i) + γ·disc_drift(i)`

| Component | Formula | Intuition |
|-----------|---------|-----------|
| **TD Error** | `\|r + γQ(s',a') - Q(s,a)\|` | Where the value function is most inaccurate — "surprise" |
| **Q Disagreement** | `\|Q1(s,a) - Q2(s,a)\|` | Where twin Q-networks disagree — epistemic uncertainty |
| **Disc. Drift** | `1 - w(z_t, z_{t+1})` | Where sim dynamics diverge from real — dynamics gap |

> [!NOTE]
> All three signals are **already computed** during H2O+ training:
> - TD error: computed in `_compute_q_loss`
> - Q disagreement: `qf1(s,a) - qf2(s,a)` from the twin networks
> - Discriminator weight `w`: computed by `ZOnlyDiscriminator`

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    H2O+ Training Loop                              │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Phase 1: Warmup (epochs 0..K)                              │  │
│  │  • Standard training with uniform random resets              │  │
│  │  • Accumulate per-transition priority scores                 │  │
│  │  • Store scores in priority_buffer[i] for offline data       │  │
│  └──────────────────────┬──────────────────────────────────────┘  │
│                         │ after K epochs                          │
│                         ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Phase 2: Targeted Reset (epochs K..N)                      │  │
│  │                                                              │  │
│  │  ┌──────────────┐    priority      ┌───────────────────┐    │  │
│  │  │ PriorityIndex │───sampling──▶│ SnapshotStore     │    │  │
│  │  │ p(i) scores   │               │ lazy load 20KB    │    │  │
│  │  └──────────────┘               └────────┬──────────┘    │  │
│  │                                           │ snapshot       │  │
│  │                                           ▼                │  │
│  │                                  ┌────────────────┐       │  │
│  │                                  │  SimEnv.reset  │       │  │
│  │                                  │  (fragile s₀)  │       │  │
│  │                                  └────────┬───────┘       │  │
│  │                                           │ rollout        │  │
│  │                                           ▼                │  │
│  │                              Counterfactual trajectory     │  │
│  │                              from the exact state where    │  │
│  │                              the policy was weakest         │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Design

### 1. PriorityIndex (in replay buffer)

```python
class PriorityIndex:
    """Maintains per-transition priority scores for offline data."""
    
    def __init__(self, n_offline: int, alpha=0.5, beta=0.3, gamma=0.2):
        self.n = n_offline
        self.alpha = alpha  # TD error weight
        self.beta = beta    # Q-disagreement weight
        self.gamma = gamma  # discriminator drift weight
        
        # Running EMA of priority signals (per offline transition)
        self.td_error = np.zeros(n_offline, dtype=np.float32)
        self.q_disagree = np.zeros(n_offline, dtype=np.float32)
        self.disc_drift = np.zeros(n_offline, dtype=np.float32)
        self.update_count = np.zeros(n_offline, dtype=np.int32)
        
    def update(self, indices: np.ndarray, td_err, q_dis, d_drift, ema=0.1):
        """Update priority scores for sampled offline transitions.
        
        Called after every training batch — scores are EMA-smoothed
        to avoid noise from single-batch estimates.
        """
        for arr, new_val in [(self.td_error, td_err), 
                              (self.q_disagree, q_dis),
                              (self.disc_drift, d_drift)]:
            old = arr[indices]
            arr[indices] = (1 - ema) * old + ema * new_val
        self.update_count[indices] += 1
        
    def sample_reset_idx(self, temperature=1.0) -> int:
        """Sample one index with probability ∝ priority^(1/temperature).
        
        temperature=1.0: standard priority sampling
        temperature→∞: uniform random (recovers standard H2O+)
        temperature→0: always pick highest priority (greedy)
        """
        # Only consider transitions that have valid snapshots
        valid = self.update_count > 0
        if not valid.any():
            return np.random.randint(self.n)
            
        p = (self.alpha * self.td_error + 
             self.beta * self.q_disagree + 
             self.gamma * self.disc_drift)
        p = np.clip(p, 0, None)  # ensure non-negative
        p[~valid] = 0
        
        # Temperature-scaled softmax
        p = p ** (1.0 / temperature)
        p = p / (p.sum() + 1e-8)
        
        return np.random.choice(self.n, p=p)
```

### 2. Priority Score Collection (in h2oplus_bus.py)

During training, after computing Q-losses, extract per-sample signals:

```python
# Inside H2OPlusBus.train(), after computing real Q-values:
with torch.no_grad():
    # Already computed:
    real_q1 = self.qf1(real_observations, real_actions)
    real_q2 = self.qf2(real_observations, real_actions)
    
    # TD error (absolute Bellman residual)
    td_error = torch.abs(real_q1 - real_td_targets).cpu().numpy()
    
    # Q-disagreement (epistemic uncertainty)
    q_disagree = torch.abs(real_q1 - real_q2).cpu().numpy()
    
    # Discriminator drift (how un-real does the dynamics look?)
    # For offline data, this is computed from z_t/z_t1
    with torch.no_grad():
        w_real = compute_z_importance_weight(
            self.discriminator, real_z_t, real_z_t1
        )
    disc_drift = (1.0 - w_real.clamp(0, 10) / 10.0).cpu().numpy()
    
    # Update priority index
    self.priority_index.update(
        batch_indices,  # which offline transitions were in this batch
        td_error, q_disagree, disc_drift
    )
```

### 3. Modified BusStepSampler (targeted reset)

```python
# In BusStepSampler.sample():
if use_buffer_reset:
    if self._episode_count > self.warmup_episodes and self.priority_index is not None:
        # JTT-style: reset to fragile states
        idx = self.priority_index.sample_reset_idx(temperature=self.reset_temperature)
    else:
        # Standard: uniform random reset
        idx = np.random.choice(self.replay_buffer._valid_snap_indices)
    
    snapshot = self.snapshot_store.get_by_buffer_idx(self.replay_buffer, idx)
    self.env.reset(snapshot=snapshot)
    max_events = self.h_rollout
```

### 4. JTT Analogy: "Error Set" Definition

In JTT, the error set E is binary: wrong/right.  
We can define a continuous version with a hard threshold fallback:

```python
# JTT-style binary error set (optional, for logging/analysis)
threshold = np.percentile(priority_scores[priority_scores > 0], 80)
error_set = priority_scores > threshold  # top 20% = "fragile"

# Log: what fraction of resets target the error set?
pct_targeted = np.mean(error_set[reset_indices])
wandb.log({"pct_fragile_resets": pct_targeted})
```

## How This Maps to JTT's Theory

| JTT Concept | H2O+ Analog | Implementation |
|-------------|-------------|----------------|
| ERM model (Phase 1) | Warmup policy | First K epochs with uniform resets |
| Error set E | Fragile transitions F | `priority_score > threshold` |
| Upweight λ | Reset probability boost | `P(reset to i) ∝ priority(i)` |
| Train from scratch (Phase 2) | Continue training | Same networks, targeted resets |
| Group robustness | Sim-robustness in fragile states | Policy improves exactly where it struggles |

> [!TIP]
> **Why this is better than vanilla JTT**: 
> JTT upweights *existing* data points. We *generate new counterfactual data* by
> resetting the sim to the fragile state and rolling out with the current (improved) policy.
> This is an **active learning** version of JTT — each reset produces fresh, policy-relevant
> experience from the hardest regions.

## Training Protocol

```
Epoch 0..K (Warmup):
  ├── Uniform random snapshot resets
  ├── Accumulate priority scores from TD error / Q-disagreement
  ├── No JTT targeting yet (cold-start protection)
  └── Log: mean_priority, priority_std, error_set_size

Epoch K..N (Targeted):
  ├── Priority-weighted snapshot resets (focused on fragile states)
  ├── Continue updating priority scores (they evolve!)
  ├── Gradually anneal temperature: T = T_max * (1 - epoch/N)^2
  │   (start broad, progressively focus more tightly)
  ├── Monitor: does avg_priority decrease? (policy is learning!)
  └── Log: reset_target_distribution, priority_decay_rate

Temperature Schedule:
  T_max = 2.0 (broad exploration at start of Phase 2)
  T_min = 0.3 (tight focus at end of training)
  Anneal: T = T_max * max(T_min/T_max, (1 - (epoch-K)/(N-K))^2)
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_epochs` (K) | 20 | Uniform reset before JTT kicks in |
| `alpha` | 0.5 | TD error weight in priority |
| `beta` | 0.3 | Q-disagreement weight |
| `gamma` | 0.2 | Discriminator drift weight |
| `temperature_max` | 2.0 | Reset sampling broadness (start) |
| `temperature_min` | 0.3 | Reset sampling tightness (end) |
| `ema_decay` | 0.1 | Priority score smoothing |
| `snapshot_cache_size` | 256 | LRU cache for lazy loading |

## Open Questions

> [!IMPORTANT]
> 1. **Priority score scope**: Should we compute priorities for ALL offline data, or only transitions with valid snapshots? 
>    - Recommendation: only transitions with snapshots (saves compute).
>
> 2. **Temperature schedule**: Linear vs exponential vs cosine annealing?
>    - Start with quadratic decay; tune based on priority decay curves.
>
> 3. **Phase transition trigger**: Should we use a fixed K, or trigger Phase 2 when some metric stabilizes?
>    -ixed K is simpler; metric-based trigger is more adaptive but harder to debug.
