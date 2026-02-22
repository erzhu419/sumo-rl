
import h5py
import numpy as np
import argparse

def check_returns(file_path):
    print(f"Analyzing Returns for {file_path}...")
    try:
        with h5py.File(file_path, 'r') as f:
            rewards = f['rewards'][:]
            terminals = f['terminals'][:]
            
        n = len(rewards)
        episode_returns = []
        current_return = 0
        
        for i in range(n):
            current_return += rewards[i]
            if terminals[i]:
                episode_returns.append(current_return)
                current_return = 0
                
        if len(episode_returns) == 0:
            print("No completed episodes found.")
        else:
            mean_ret = np.mean(episode_returns)
            std_ret = np.std(episode_returns)
            max_ret = np.max(episode_returns)
            min_ret = np.min(episode_returns)
            print(f"Analysis of {len(episode_returns)} Episodes:")
            print(f"  Mean Return: {mean_ret:.2f}")
            print(f"  Std Return : {std_ret:.2f}")
            print(f"  Max Return : {max_ret:.2f}")
            print(f"  Min Return : {min_ret:.2f}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    args = parser.parse_args()
    check_returns(args.file)
