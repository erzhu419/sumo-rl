import sys
import os

# Fix for Intel MKL conflict with torch/libsumo (suggested by error logs)
os.environ["MKL_THREADING_LAYER"] = "GNU"

import h5py
import numpy as np
import torch
import argparse
import datetime

# Add project root needed for bridge
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from offline_sumo.envs.sumo_env import SumoBusHoldingEnv

# Add RORL to path
RORL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../RORL"))
if RORL_PATH not in sys.path:
    sys.path.append(RORL_PATH)

from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.trainers.q_learning.sac import SACTrainer
from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
from lifelong_rl.policies.base.base import MakeDeterministic
import lifelong_rl.util.pythonplusplus as ppp
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchOfflineRLAlgorithm
import lifelong_rl.torch.pytorch_util as ptu

# --- Embedding Layer (Aligned with CQL) ---
class EmbeddingLayer(torch.nn.Module):
    def __init__(self, cat_dims, embedding_dims=None, layer_norm=False, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.cat_dims = cat_dims
        self.cat_names = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
        
        # ALIGN WITH CQL: [4, 21, 1, 1, 2]
        self.manual_embed_dims = [4, 21, 1, 1, 2]
        self.embedding_dims = []
        modules = []
        for i, card in enumerate(self.cat_dims):
            dim = self.manual_embed_dims[i] if i < len(self.manual_embed_dims) else min(50, card // 2)
            self.embedding_dims.append(dim)
            modules.append(torch.nn.Embedding(card, dim))
        self.embeddings = torch.nn.ModuleList(modules)
        self.output_dim = sum(self.embedding_dims)
        self.layer_norm = torch.nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        embedded = []
        for i, emb in enumerate(self.embeddings):
            val = x[:, i].long()
            val = torch.clamp(val, 0, self.cat_dims[i]-1)
            embedded.append(emb(val))
        out = torch.cat(embedded, dim=1)
        if self.layer_norm:
            out = self.layer_norm(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class EmbeddingTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(self, cat_dims, num_cont, cont_mean, cont_std, **kwargs):
        # Compute input_dim which includes the sum of embedding dimensions and continuous features
        temp_emb = EmbeddingLayer(cat_dims)
        input_dim = temp_emb.output_dim + num_cont
        super().__init__(obs_dim=input_dim, **kwargs)
        self.embedding_layer = temp_emb
        self.num_cats = len(cat_dims)
        self.register_buffer('cont_mean', ptu.from_numpy(cont_mean))
        self.register_buffer('cont_std', ptu.from_numpy(cont_std))

    def forward(self, obs, **kwargs):
        cats = obs[:, :self.num_cats].long()
        conts = obs[:, self.num_cats:]
        conts_norm = (conts - self.cont_mean) / (self.cont_std + 1e-6)
        embeds = self.embedding_layer(cats)
        mlp_input = torch.cat([embeds, conts_norm], dim=-1)
        return super().forward(mlp_input, **kwargs)

class EmbeddingEnsembleFlattenMLP(ParallelizedEnsembleFlattenMLP):
    def __init__(self, cat_dims, num_cont, cont_mean, cont_std, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = EmbeddingLayer(cat_dims)
        self.num_cats = len(cat_dims)
        self.register_buffer('cont_mean', ptu.from_numpy(cont_mean))
        self.register_buffer('cont_std', ptu.from_numpy(cont_std))

    def forward(self, *inputs, **kwargs):
        obs, action = inputs[0], inputs[1]
        cats = obs[:, :self.num_cats].long()
        conts = obs[:, self.num_cats:]
        conts_norm = (conts - self.cont_mean) / (self.cont_std + 1e-6)
        embeds = self.embedding_layer(cats)
        mlp_input = torch.cat([embeds, conts_norm], dim=-1)
        return super().forward(mlp_input, action, **kwargs)

def train_rorl(args):
    ptu.set_gpu_mode(torch.cuda.is_available())
    
    # 1. Load Data
    print(f"Loading dataset from {args.dataset}")
    with h5py.File(args.dataset, 'r') as f:
        obs = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        next_obs = f['next_observations'][:]
        terminals = f['terminals'][:]
        
    num_cats = 5
    cat_dims = []
    for i in range(num_cats):
        card = int(np.max(obs[:, i])) + 1
        cat_dims.append(card)
    print(f"Inferred Categorical Dims: {cat_dims}")
    
    cont_data = obs[:, num_cats:]
    cont_mean = np.mean(cont_data, axis=0)
    cont_std = np.std(cont_data, axis=0) + 1e-6
    print(f"Cont Mean: {cont_mean}")

    print(f"Original Reward Mean: {np.mean(rewards)}")
    rewards = rewards * 0.01 # Hardcoded 0.01 to match CQL
    print(f"Scaled Reward Mean: {np.mean(rewards)}")

    # 2. Env
    base_env = SumoBusHoldingEnv(gui=False, max_steps=args.eval_sim_steps)
    # No wrapper needed if policy handles it
    env = base_env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Env Obs Dim: {obs_dim}, Action Dim: {action_dim}")

    # 3. Buffer (Memory Efficient: Storing raw integers and floats)
    print("Populating Replay Buffer...")
    replay_buffer = EnvReplayBuffer(
        max_replay_buffer_size=len(obs),
        env=env
    )
    replay_buffer._observations[:len(obs)] = obs
    replay_buffer._actions[:len(obs)] = actions
    replay_buffer._rewards[:len(obs)] = rewards.reshape(-1, 1)
    replay_buffer._next_obs[:len(obs)] = next_obs
    replay_buffer._terminals[:len(obs)] = terminals.reshape(-1, 1)
    replay_buffer._top = len(obs)
    replay_buffer._size = len(obs)
    print(f"Buffer Size: {replay_buffer._size}")

    # 4. Networks
    M = args.hidden_dim
    num_qs = args.num_qs
    
    num_cont = obs_dim - num_cats
    embed_dim = EmbeddingLayer(cat_dims).output_dim
    
    # Matching sac_v2_bus_ensemble.py: 4 hidden layers
    hidden_sizes = [M, M, M, M]
    
    # Obs Std Padding for Trainer OOD Logic
    # obs = [cat...cat, cont...cont]
    # trainer applies 2 * eps * obs_std * (rand-0.5)
    # We want 1.0 for cats (muted anyway) and cont_std for conts
    full_obs_std = np.ones(obs_dim, dtype=np.float32)
    full_obs_std[num_cats:] = cont_std

    qfs, target_qfs = ppp.group_init(
        2,
        EmbeddingEnsembleFlattenMLP,
        cat_dims=cat_dims,
        num_cont=num_cont,
        cont_mean=cont_mean,
        cont_std=cont_std,
        ensemble_size=num_qs,
        hidden_sizes=hidden_sizes,
        input_size=embed_dim + num_cont + action_dim,
        output_size=1,
    )
    
    policy = EmbeddingTanhGaussianPolicy(
        cat_dims=cat_dims,
        num_cont=num_cont,
        cont_mean=cont_mean,
        cont_std=cont_std,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    # Calculate steps per epoch for RORL BC Phase and Algorithm
    steps_per_epoch = len(obs) // args.batch_size
    print(f"Steps per Epoch (full pass): {steps_per_epoch}")

    # 5. Trainer
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qfs=qfs,
        target_qfs=target_qfs,
        replay_buffer=replay_buffer,
        
        # RORL / Robustness Params
        q_ood_reg=args.beta_ood, 
        q_ood_uncertainty_reg=args.beta_uncertainty,
        q_ood_eps=args.eps_ood,
        
        policy_smooth_reg=args.beta_smooth,
        policy_smooth_eps=args.eps_smooth,          # Essential for smoothing
        q_smooth_reg=args.beta_q_smooth,
        q_smooth_eps=args.eps_q_smooth,             # Essential for smoothing
        
        eta=args.eta,                               # Gradient diversity (PBRL)
        num_samples=args.num_samples,               # Monte Carlo samples for reg
        policy_eval_start=args.bc_epochs * steps_per_epoch, # BC Phase
        
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        
        # SAC / Optimization Params
        discount=args.gamma,
        soft_target_tau=args.tau,                   # ADDED: Tau control
        use_automatic_entropy_tuning=True,
        policy_lr=args.learning_rate,
        qf_lr=args.learning_rate,
        num_qs=num_qs,

        # ALIGNMENT FIXES:
        norm_input=True,
        obs_std=full_obs_std,
        num_cats=num_cats,
    )

    # 6. Logging Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(PROJECT_ROOT, "offline_sumo", "RORL", "logs", f"rorl_{timestamp}")
    from lifelong_rl.core import logger
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode('last')
    logger.add_tabular_output(os.path.join(log_dir, 'progress.csv'))
    logger.add_text_output(os.path.join(log_dir, 'debug.log'))
    print(f"Logging to {log_dir}")
    
    eval_path_collector = MdpPathCollector(
        env=env,
        policy=MakeDeterministic(policy),
    )

    # 8. Plotting Function (Integrated as post_epoch_func)
    def rorl_post_epoch_plot(algo, epoch):
        from offline_sumo.RORL.plot_rorl import plot_rorl
        # logger.get_snapshot_dir() is log_dir
        log_dir = logger.get_snapshot_dir()
        plot_rorl(log_dir)
        print(f"Epoch {epoch} plot updated in {log_dir}")

    algorithm = TorchOfflineRLAlgorithm(
        trainer=trainer,
        evaluation_policy=MakeDeterministic(policy),
        evaluation_env=env,
        replay_buffer=replay_buffer,
        evaluation_data_collector=eval_path_collector,
        max_path_length=args.eval_sim_steps,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_eval_steps_per_epoch=args.eval_sim_steps if args.eval_freq > 0 else 0,
        num_trains_per_train_loop=steps_per_epoch,
    )
    algorithm.post_epoch_funcs.append(rorl_post_epoch_plot)
    
    algorithm.to(ptu.device)
    
    try:
        algorithm.train()
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        save_path = os.path.join(log_dir, "rorl_policy_final.pth")
        torch.save(policy.state_dict(), save_path)
        print(f"Final policy saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Core Args (Shared with CQL)
    parser.add_argument("--dataset", type=str, default="offline_sumo/data/buffer_expert_parallel.hdf5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64) # ALIGNED: hidden_dim=64
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)    # ALIGNED: tau=0.01
    
    # RORL Robustness Suite
    parser.add_argument("--beta_ood", type=float, default=0.01) # ALIGNED: beta_ood=0.01
    parser.add_argument("--beta_uncertainty", type=float, default=2.0) # ALIGNED: beta=2.0
    parser.add_argument("--eps_ood", type=float, default=0.1)
    
    parser.add_argument("--beta_smooth", type=float, default=1.0) # Policy smoothing weight
    parser.add_argument("--eps_smooth", type=float, default=0.01) # Policy smoothing noise
    parser.add_argument("--beta_q_smooth", type=float, default=0.0) # Q smoothing weight
    parser.add_argument("--eps_q_smooth", type=float, default=0.0) # Q smoothing noise
    
    parser.add_argument("--eta", type=float, default=-1.0) # Gradient diversity (-1 to disable)
    parser.add_argument("--num_samples", type=int, default=20) # MC Samples for noise
    parser.add_argument("--bc_epochs", type=int, default=0) # Epochs of initial BC phase
    
    parser.add_argument("--max_q_backup", action="store_true", default=False)
    parser.add_argument("--deterministic_backup", action="store_true", default=False)
    
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--eval_sim_steps", type=int, default=18000)
    
    args = parser.parse_args()
    train_rorl(args)
