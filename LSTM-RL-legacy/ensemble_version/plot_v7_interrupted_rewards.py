import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
log_dir = "logs/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long"
reward_file = os.path.join(log_dir, "rewards.npy")
output_file = "pic/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/rewards_ep1_to_48.png"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

try:
    # Load rewards
    rewards = np.load(reward_file)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, marker='o', linestyle='-', color='b', alpha=0.7)
    
    # Add labels and formatting
    plt.title("Reward per Episode (Production V7 Long) - Epochs 0-48", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate max and min points for context
    max_idx = np.argmax(rewards)
    min_idx = np.argmin(rewards)
    plt.annotate(f"Max: {rewards[max_idx]:.0f}", xy=(max_idx, rewards[max_idx]), xytext=(max_idx, rewards[max_idx]+5000),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=5))
    plt.annotate(f"Min: {rewards[min_idx]:.0f}", xy=(min_idx, rewards[min_idx]), xytext=(min_idx, rewards[min_idx]-25000),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Successfully generated plot at: {output_file}")
    
except Exception as e:
    print(f"Error generating plot: {e}")
