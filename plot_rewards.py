import numpy as np
import matplotlib.pyplot as plt
import os

LOG_DIR = '/home/erzhu419/mine_code/sumo-rl/logs/sac_v2_bus_SUMO_gpt_version'
REWARDS_FILE = os.path.join(LOG_DIR, 'rewards.npy')
OUTPUT_IMAGE = os.path.join(LOG_DIR, 'cumulative_rewards_gpt.png')

def plot_rewards():
    if not os.path.exists(REWARDS_FILE):
        print(f"Error: {REWARDS_FILE} not found.")
        return

    rewards = np.load(REWARDS_FILE)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='o', linestyle='-', color='g', label='Episode Reward') # Green for success
    plt.title('Cumulative Reward per Episode (sac_v2_bus_SUMO_gpt_version)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(OUTPUT_IMAGE)
    print(f"Plot saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_rewards()
