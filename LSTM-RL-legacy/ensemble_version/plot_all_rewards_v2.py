import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_rewards():
    logs_dir = "/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/logs"
    reward_files = glob.glob(os.path.join(logs_dir, "*/rewards.npy"))
    
    # Filter out short or extremely bad runs that squash the graph
    valid_runs = []
    for r_file in reward_files:
        try:
            data = np.load(r_file)
            if data.ndim == 0 or len(data) < 5:
                continue
                
            mean_val = np.mean(data)
            # Filter extremely bad outliers (e.g., below -3M) to keep the scale meaningful
            if mean_val < -3000000:
                print(f"Skipping {os.path.basename(os.path.dirname(r_file))} (Mean={mean_val:.2f}) - Outlier")
                continue
                
            valid_runs.append((r_file, data))
        except Exception as e:
            print(f"Error loading {r_file}: {e}")
            
    # Sort for consistent coloring
    valid_runs.sort(key=lambda x: os.path.basename(os.path.dirname(x[0])))
    
    plt.figure(figsize=(15, 8))
    
    # Use a high-distinction color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_runs)))
    
    for i, (r_file, data) in enumerate(valid_runs):
        dir_name = os.path.basename(os.path.dirname(r_file))
        
        # Simplify label
        label = dir_name
        if label.startswith("sac_ensemble_SUMO_linear_penalty_"):
            label = "[L] " + label[len("sac_ensemble_SUMO_linear_penalty_"):]
        elif label.startswith("sac_ensemble_SUMO_heuristic_guidance_"):
            label = "[G] " + label[len("sac_ensemble_SUMO_heuristic_guidance_"):]
            
        color = colors[i]
        episodes = np.arange(len(data))
        
        # Raw data (very light)
        plt.plot(episodes, data, color=color, alpha=0.15)
        
        # Smoothed version (solid)
        window = max(1, len(data) // 10)
        if len(data) > window:
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            plt.plot(episodes[window-1:], smoothed, label=label, color=color, linewidth=2)
        else:
            plt.plot(episodes, data, label=label, color=color, linewidth=2)

    plt.title("Reward Comparison (Filtered Outliers < -3M)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Limit Y axis to meaningful range if things are still squashed
    # plt.ylim(-2000000, -500000) 
    
    plt.tight_layout()
    
    output_path = "/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/all_rewards_comparison_filtered.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_rewards()
