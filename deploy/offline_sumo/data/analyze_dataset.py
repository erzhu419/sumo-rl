import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        obs = f['observations'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
    
    print(f"Dataset Size: {len(obs)}")
    
    print("\n--- Observation Stats ---")
    print(f"Shape: {obs.shape}")
    print(f"Mean: {np.mean(obs, axis=0)}")
    print(f"Std:  {np.std(obs, axis=0)}")
    print(f"Min:  {np.min(obs, axis=0)}")
    print(f"Max:  {np.max(obs, axis=0)}")
    # Obs[3] is time_idx, Obs[-1] is sim_time. Check magnitudes.
    
    print("\n--- Action Stats ---")
    print(f"Mean: {np.mean(actions)}")
    print(f"Std:  {np.std(actions)}")
    print(f"Min:  {np.min(actions)}")
    print(f"Max:  {np.max(actions)}")
    
    print("\n--- Reward Stats ---")
    print(f"Mean: {np.mean(rewards)}")
    print(f"Std:  {np.std(rewards)}")
    print(f"Min:  {np.min(rewards)}")
    print(f"Max:  {np.max(rewards)}")
    
    # Check for sparse rewards
    print(f"Zero Rewards: {np.sum(rewards == 0)}")

if __name__ == "__main__":
    analyze("offline_sumo/data/buffer.hdf5")
