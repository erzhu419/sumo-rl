
import numpy as np
import sys

files = [
    "logs/sac_v2_bus_SUMO_best_result/rewards.npy",
]

for f in files:
    try:
        data = np.load(f)
        print(f"File: {f}")
        print(f"  Shape: {data.shape}")
        print(f"  Mean: {np.mean(data)}")
        print(f"  Max: {np.max(data)}")
        print(f"  Min: {np.min(data)}")
        print(f"  Last 5: {data[-5:]}")
    except Exception as e:
        print(f"Could not load {f}: {e}")
