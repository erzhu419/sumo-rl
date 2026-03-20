import os
import numpy as np
import glob

def check_rewards():
    logs_dir = "/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/logs"
    reward_files = glob.glob(os.path.join(logs_dir, "*/rewards.npy"))
    
    print(f"{'Run Name':<60} | {'Shape':<10} | {'Mean':<12} | {'Max':<12}")
    print("-" * 100)
    
    for r_file in sorted(reward_files):
        try:
            dir_name = os.path.basename(os.path.dirname(r_file))
            data = np.load(r_file)
            if data.ndim == 0 or len(data) == 0:
                print(f"{dir_name:<60} | EMPTY")
                continue
                
            mean_val = np.mean(data)
            max_val = np.max(data)
            print(f"{dir_name:<60} | {str(data.shape):<10} | {mean_val:>12.2f} | {max_val:>12.2f}")
        except Exception as e:
            print(f"{r_file}: Error {e}")

if __name__ == "__main__":
    check_rewards()
