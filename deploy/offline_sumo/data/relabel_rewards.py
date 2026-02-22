
import h5py
import numpy as np
import argparse

def relabel_rewards(file_path):
    print(f"Relabeling Rewards for {file_path} to limit Max=0...")
    try:
        with h5py.File(file_path, 'r+') as f:
            obs = f['observations'][:]
            rewards = f['rewards'][:]
            
            # Logic: Reward = -0.1 * abs(current - target)
            # Obs Structure: [line, bus, station, time, dir, fwd_h, bwd_h, wait, target, dur, time]
            # Index 5: fwd_h
            # Index 8: target
            
            fwd_h = obs[:, 5]
            target = obs[:, 8]
            
            # Match baseline: Reward = -1.0 * abs(current - target)
            new_rewards = -1.0 * np.abs(fwd_h - target)
            
            # Sanity Check
            print(f"Old Rewards: Mean={np.mean(rewards):.4f}, Max={np.max(rewards):.4f}")
            print(f"New Rewards: Mean={np.mean(new_rewards):.4f}, Max={np.max(new_rewards):.4f}, Min={np.min(new_rewards):.4f}")
            
            # Overwrite
            del f['rewards']
            f.create_dataset('rewards', data=new_rewards)
            print("Rewards updated successfully.")

    except Exception as e:
        print(f"Error updating file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()
    relabel_rewards(args.file)
