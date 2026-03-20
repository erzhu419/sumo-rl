import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
log_dir = "logs/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long"
reward_file = os.path.join(log_dir, "rewards.npy")
output_file = "pic/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/full_rewards_150ep.png"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

try:
    # Load rewards
    rewards = np.load(reward_file)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(rewards)), rewards, marker='.', linestyle='-', color='#1f77b4', alpha=0.6, label='Episode Reward')
    
    # Calculate moving average for smoother visualization
    window_size = 5
    if len(rewards) >= window_size:
        sma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), sma, color='#ff7f0e', linewidth=2, label=f'SMA {window_size}')

    # Highlight the best point
    best_idx = np.argmax(rewards)
    plt.scatter(best_idx, rewards[best_idx], color='red', s=100, zorder=5, label=f'Best: Ep {best_idx}')
    plt.annotate(f"Best: {rewards[best_idx]:.0f} (Ep {best_idx})", 
                 xy=(best_idx, rewards[best_idx]), 
                 xytext=(best_idx, rewards[best_idx] + 50000 if rewards[best_idx] < -1000000 else rewards[best_idx] + 20000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10, fontweight='bold')

    # Baseline reference (-0.71M)
    plt.axhline(y=-710000, color='green', linestyle='--', alpha=0.5, label='Heuristic Baseline (-0.71M)')

    # Add labels and formatting
    plt.title("Full 150-Episode Learning Curve (Production V7 Long)", fontsize=16)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    
    # Set y-axis limits to focus on the -1M to -0.6M range primarily
    plt.ylim(max(rewards.min(), -1500000), rewards.max() + 50000)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    print(f"Successfully generated full plot at: {output_file}")
    
except Exception as e:
    print(f"Error generating plot: {e}")
