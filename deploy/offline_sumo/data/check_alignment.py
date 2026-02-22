
import h5py
import numpy as np
import argparse

def check_alignment(file_path):
    print(f"Checking alignment for {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            obs = f['observations'][:]
            next_obs = f['next_observations'][:]
            terminals = f['terminals'][:]
            rewards = f['rewards'][:]
            actions = f['actions'][:]
            
        n = len(obs)
        print(f"Total transitions: {n}")
        
        mismatches = 0
        checks = 0
        
        # Check obs[t+1] == next_obs[t] unless done[t] is True
        # Note: This assumes data is stored sequentially by episode
        for i in range(n - 1):
            if not terminals[i]:
                # Next step of current transition should match current step of next transition
                # Allow small float tolerance
                diff = np.abs(obs[i+1] - next_obs[i])
                if np.max(diff) > 1e-4:
                    print(f"Mismatch at index {i}")
                    print(f"Next Obs [t]: {next_obs[i][:5]}...")
                    print(f"Obs [t+1]   : {obs[i+1][:5]}...")
                    mismatches += 1
                    if mismatches > 10:
                        break
                checks += 1
                
        print(f"Checked {checks} sequential transitions.")
        if mismatches == 0:
            print("SUCCESS: All sequential transitions align perfectly (obs[t+1] == next_obs[t]).")
        else:
            print(f"FAILURE: Found {mismatches} mismatches.")

        # Basic Stats
        print("\nStatistics:")
        print(f"Rewards: Mean={np.mean(rewards):.4f}, Std={np.std(rewards):.4f}, Min={np.min(rewards)}, Max={np.max(rewards)}")
        print(f"Actions: Mean={np.mean(actions):.4f}, Min={np.min(actions)}, Max={np.max(actions)}")
        print(f"Terminals: {np.sum(terminals)} episodes found.")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()
    check_alignment(args.file)
