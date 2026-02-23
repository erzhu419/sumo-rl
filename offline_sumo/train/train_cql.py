import os
import sys
import datetime
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Embedding Layer from Reference ---
class EmbeddingLayer(nn.Module):
    def __init__(self, cat_dims, embedding_dims=None, layer_norm=False, dropout=0.0):
        super(EmbeddingLayer, self).__init__()
        self.cat_dims = cat_dims # List of cardinalities
        
        self.embedding_dims = []
        modules = []
        for i, card in enumerate(self.cat_dims):
            dim = min(32, max(2, int(round(card ** 0.5)) + 1)) # Reference heuristic
            self.embedding_dims.append(dim)
            modules.append(nn.Embedding(card, dim))
            
        self.embeddings = nn.ModuleList(modules)
        self.output_dim = sum(self.embedding_dims)
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        # x: (batch, num_cats)
        embedded = []
        for i, emb in enumerate(self.embeddings):
            val = x[:, i].long()
            # Safety clamp
            val = torch.clamp(val, 0, self.cat_dims[i]-1)
            embedded.append(emb(val))
        
        out = torch.cat(embedded, dim=1)
        if self.layer_norm:
            out = self.layer_norm(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class Actor(nn.Module):
    def __init__(self, cat_dims, num_cont, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.embedding = EmbeddingLayer(cat_dims)
        input_dim = self.embedding.output_dim + num_cont
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Added 3rd layer as per reference
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        # State: [cat_1, ..., cat_k, cont_1, ..., cont_m]
        num_cats = len(self.embedding.cat_dims)
        cats = state[:, :num_cats]
        conts = state[:, num_cats:]
        
        embeds = self.embedding(cats)
        x = torch.cat([embeds, conts], dim=1)
        mean = self.net(x)
        log_std = torch.clamp(self.log_std, -20, 2)
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()  # Reparameterization trick
            
        action = torch.tanh(x_t)
        
        # Calculate log_prob
        log_prob = normal.log_prob(x_t)
        # Enforce Action Bound correction for Tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, cat_dims, num_cont, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.embedding = EmbeddingLayer(cat_dims)
        input_dim = self.embedding.output_dim + num_cont + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state, action):
        num_cats = len(self.embedding.cat_dims)
        cats = state[:, :num_cats]
        conts = state[:, num_cats:]
        
        embeds = self.embedding(cats)
        x = torch.cat([embeds, conts, action], dim=1)
        return self.net(x)

def train_cql(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    print(f"Loading dataset from {args.dataset}")
    with h5py.File(args.dataset, 'r') as f:
        obs = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        next_obs = f['next_observations'][:]
        terminals = f['terminals'][:]
    
    # Normalize Actions (0-60 -> -1 to 1)
    action_min = 0.0
    action_max = 60.0
    actions_norm = 2 * (actions - action_min) / (action_max - action_min) - 1
    
    # Feature Engineering (Split Cat/Cont)
    # Cat Indices: 0, 1, 2, 3, 4
    # Cont Indices: 5 to 10
    num_cats = 5
    cat_data = obs[:, :num_cats]
    cont_data = obs[:, num_cats:]
    next_cat_data = next_obs[:, :num_cats]
    next_cont_data = next_obs[:, num_cats:]
    
    # Infer Cardinalities
    cat_dims = []
    for i in range(num_cats):
        card = int(np.max(cat_data[:, i])) + 1
        cat_dims.append(card)
    print(f"Inferred Categorical Dims: {cat_dims}")
    
    # Normalize Continuous Features ONLY
    cont_mean = np.mean(cont_data, axis=0)
    cont_std = np.std(cont_data, axis=0) + 1e-6
    
    cont_norm = (cont_data - cont_mean) / cont_std
    next_cont_norm = (next_cont_data - cont_mean) / cont_std
    
    # Reassemble Obs
    obs_processed = np.concatenate([cat_data, cont_norm], axis=1)
    next_obs_processed = np.concatenate([next_cat_data, next_cont_norm], axis=1)
    
    # Scale Rewards (Simple Fixed Scaling to Match Reference Magnitude)
    # Z-score shifting can be unstable for CQL (creating artificial positive rewards).
    # We use fixed scaling to keep rewards negative (Max 0).
    rewards_scaled = rewards * 0.01 # Map -75 to -0.75, -953 to -9.5. This keeps gradients stable.
    # Actually, let's use 0.1 if range is [-953, 0], giving [-95, 0].
    # Or 1/100 to get unit variance roughly? Std is 84.
    # Let's use 1/100 (0.01).
    rewards_scaled = rewards * 0.01
    
    # Update Stats Print
    print(f"Cont Mean: {cont_mean}")
    print(f"Cont Std: {cont_std}")
    print(f"Rewards Scaled (Factor 0.01): Mean={np.mean(rewards_scaled):.4f}, Std={np.std(rewards_scaled):.4f}, Max={np.max(rewards_scaled)}")

    dataset = TensorDataset(
        torch.FloatTensor(obs_processed),
        torch.FloatTensor(actions_norm),
        torch.FloatTensor(rewards_scaled),
        torch.FloatTensor(next_obs_processed),
        torch.FloatTensor(terminals).float()
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    num_cont = cont_data.shape[1]
    action_dim = actions.shape[1]
    
    # Initialize with Embeddings
    actor = Actor(cat_dims, num_cont, action_dim, hidden_dim=args.hidden_dim).to(device)
    critic1 = Critic(cat_dims, num_cont, action_dim, hidden_dim=args.hidden_dim).to(device)
    critic2 = Critic(cat_dims, num_cont, action_dim, hidden_dim=args.hidden_dim).to(device)
    
    # Save Scaler (Only for continuous parts)
    np.savez(os.path.join(os.path.dirname(args.dataset), "scaler_v2.npz"), 
             mean=cont_mean, std=cont_std, cat_dims=cat_dims)
             
    target_critic1 = Critic(cat_dims, num_cont, action_dim, hidden_dim=args.hidden_dim).to(device)
    target_critic2 = Critic(cat_dims, num_cont, action_dim, hidden_dim=args.hidden_dim).to(device)
    
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    
    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=args.learning_rate)
    
    # Auto-Alpha Tuning
    target_entropy = -action_dim # Heuristic: -dim(A)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = optim.Adam([log_alpha], lr=args.learning_rate)
    alpha = log_alpha.exp().item()
    
    # WandB Init
    try:
        import wandb
        wandb.init(project="offline_sumo_cql", config=args)
    except ImportError:
        print("WandB not installed. Skipping logging.")
        wandb = None

    # Logging history
    history = {
        "loss": [],
        "q_value": [],
        "return": [],
        "epochs": []
    }
    
    # Eval Env
    try:
        from offline_sumo.envs.sumo_env import SumoBusHoldingEnv
        # Use longer evaluation episodes to match reference magnitude
        eval_env = SumoBusHoldingEnv(gui=False, max_steps=args.eval_sim_steps) 
        has_eval_env = True
    except Exception as e:
        print(f"Warning: Could not enable evaluation env: {e}")
        has_eval_env = False

    # Create Log Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(PROJECT_ROOT, "offline_sumo", "logs", f"cql_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")
    
    # CSV Logger
    csv_file = os.path.join(log_dir, "progress.csv")
    with open(csv_file, "w") as f:
        f.write("epoch,loss,q_value,return,episode_length\n")

    # Infinite DataLoader
    def infinite_dataloader(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = infinite_dataloader(dataloader)

    pbar = tqdm(range(args.epochs))
    try:
        for epoch in pbar:
            epoch_loss = 0
            epoch_q1 = 0
            steps = 0
            
            # Fixed steps per epoch
            for _ in range(args.steps_per_epoch):
                batch = next(train_iter)
                b_s, b_a, b_r, b_ns, b_d = [t.to(device) for t in batch]
                b_r = b_r.unsqueeze(1)
                b_d = b_d.unsqueeze(1)
                
                # --- Critic Update (CQL) ---
                with torch.no_grad():
                    next_action, next_log_prob = actor.get_action(b_ns)
                    
                    target_q1 = target_critic1(b_ns, next_action)
                    target_q2 = target_critic2(b_ns, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    
                    # SAC Target: r + gamma * (Q - alpha * log_pi)
                    target_val = b_r + (1 - b_d) * args.gamma * (target_q - alpha * next_log_prob)
                
                current_q1 = critic1(b_s, b_a)
                current_q2 = critic2(b_s, b_a)
                
                mse_loss = nn.MSELoss()(current_q1, target_val) + nn.MSELoss()(current_q2, target_val)
                
                # CQL Regularization
                random_actions = torch.rand_like(b_a) * 2 - 1
                cql_loss1 = torch.logsumexp(critic1(b_s, random_actions), dim=0).mean() - current_q1.mean()
                cql_loss2 = torch.logsumexp(critic2(b_s, random_actions), dim=0).mean() - current_q2.mean()
                
                critic_loss = mse_loss + args.cql_weight * (cql_loss1 + cql_loss2)
                
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()
                
                # --- Actor Update ---
                pred_action, pred_log_prob = actor.get_action(b_s)
                q1_pred = critic1(b_s, pred_action)
                q2_pred = critic2(b_s, pred_action)
                q_pred = torch.min(q1_pred, q2_pred)
                
                actor_loss = (alpha * pred_log_prob - q_pred).mean()
                
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                
                # --- Alpha Update ---
                alpha_loss = -(log_alpha * (pred_log_prob + target_entropy).detach()).mean()
                
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()
                
                alpha = log_alpha.exp().item()
                
                # Soft update
                for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
                    target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
                for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
                    target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
                    
                epoch_loss += critic_loss.item()
                epoch_q1 += current_q1.mean().item()
                steps += 1
                
            avg_loss = epoch_loss/steps
            avg_q1 = epoch_q1/steps
            
            # --- Evaluation ---
            eval_return = 0.0
            eval_steps = 0
            if has_eval_env and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
                eval_obs = eval_env.reset()
                done = False
                total_r = 0
                eval_steps = 0
                while not done:
                    # Normalize Obs (Split Cat/Cont)
                    num_cats = 5
                    cat_part = eval_obs[:num_cats]
                    cont_part = eval_obs[num_cats:]
                    cont_part_norm = (cont_part - cont_mean) / cont_std
                    processed_obs = np.concatenate([cat_part, cont_part_norm])
                    
                    s_t = torch.FloatTensor(processed_obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        a_t, _ = actor.get_action(s_t, deterministic=True) 
                        a_t = a_t.cpu().numpy()[0]
                    
                    real_action = (a_t + 1) / 2 * 60.0 # Map -1,1 to 0,60
                    eval_obs, r, done, _ = eval_env.step([real_action])
                    total_r += r
                    eval_steps += 1
                eval_return = total_r
            else:
                eval_return = history["return"][-1] if history["return"] else 0.0
                eval_steps = history["episode_length"][-1] if history.get("episode_length") else 1

            history["loss"].append(avg_loss)
            history["q_value"].append(avg_q1)
            history["return"].append(eval_return)
            history["epochs"].append(epoch)
            if "episode_length" not in history: history["episode_length"] = []
            history["episode_length"].append(eval_steps)
            
            pbar.set_description(f"Epoch {epoch} Loss: {avg_loss:.2f} Q: {avg_q1:.2f} Ret: {eval_return:.2f}")
            if wandb:
                wandb.log({"loss": avg_loss, "q_value": avg_q1, "return": eval_return, "epoch": epoch})       
            
            # CSV Log
            with open(csv_file, "a") as f:
                f.write(f"{epoch},{avg_loss:.4f},{avg_q1:.4f},{eval_return:.4f},{eval_steps}\n")

            # Real-time Plotting (Every epoch)
            try:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                axs[0].plot(history["epochs"], history["loss"])
                axs[0].set_title("Critic Loss")
                axs[0].set_xlabel("Epoch")
                
                axs[1].plot(history["epochs"], history["q_value"])
                axs[1].set_title("Average Q Value")
                axs[1].set_xlabel("Epoch")
                
                axs[2].plot(history["epochs"], history["return"])
                axs[2].set_title("Evaluation Return")
                axs[2].set_xlabel("Epoch")
                
                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, "training_results.png"))
                plt.close(fig)
            except ImportError:
                pass

            # Checkpoint (Save every 10 epochs or updated best)
            if epoch % 10 == 0:
                torch.save(actor.state_dict(), os.path.join(log_dir, f"cql_actor_epoch_{epoch}.pth"))

    except KeyboardInterrupt:
        print("Training interrupted! Saving current state...")
    finally:
        # Final Save
        torch.save(actor.state_dict(), os.path.join(log_dir, "cql_actor_final.pth"))
        print(f"Model saved to {log_dir}")
        if has_eval_env:
            eval_env.close()
    
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="offline_sumo/data/buffer.hdf5") 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=1000) # Decoupled from dataset size
    parser.add_argument("--batch_size", type=int, default=2048) # Updated to Match SAC
    parser.add_argument("--learning_rate", type=float, default=1e-5) # Match Reference (Very Low!)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden_dim", type=int, default=32) # Match Reference (Tiny!)
    parser.add_argument("--cql_weight", type=float, default=0.2) # Reduced from 1.0 for stochastic env
    parser.add_argument("--sac_alpha", type=float, default=0.2)   # New entropy temp
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--eval_sim_steps", type=int, default=18000) # 5 hours
    args = parser.parse_args()
    train_cql(args)
