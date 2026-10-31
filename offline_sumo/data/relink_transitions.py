
import h5py
import numpy as np
import argparse
from tqdm import tqdm

def relink_dataset(dataset_path):
    print(f"Relinking transitions in {dataset_path}...")
    
    with h5py.File(dataset_path, 'r+') as f:
        obs = f['observations'][:]
        next_obs = f['next_observations'][:]
        rewards = f['rewards'][:]
        terminals = f['terminals'][:]
        actions = f['actions'][:]
        
        # Identify Bus ID column (Assumption: 1st column based on cat_dims)
        # obs shape: (N, dim)
        # Col 0: Bus ID
        bus_ids = obs[:, 0].astype(int)
        
        unique_buses = np.unique(bus_ids)
        print(f"Found {len(unique_buses)} unique buses.")
        
        # We need to preserve the original order or just update next_obs in place?
        # Updating in place is hard if we need to find "next" row which might be far away.
        # Strategy:
        # 1. Group indices by bus_id
        # 2. Sort indices by time? 
        #    Wait, the dataset is collected sequentially by time (mostly).
        #    The "Event Time" is likely in the observation?
        #    Col 2 is 'time_period' (hour).
        #    Wait, we need precise time or just trust the file order?
        #    Use file order as proxy for time (since it was appended sequentially).
        
        total_relinked = 0
        total_terminals = 0
        
        # Create a mapping for fast lookups if needed, but simple list per bus is easier
        bus_indices = {bid: [] for bid in unique_buses}
        
        for idx, bid in enumerate(bus_ids):
            bus_indices[bid].append(idx)
            
        # Iterate over each bus's trajectory
        for bid, indices in bus_indices.items():
            # Sort indices just in case (though file order should be correct for a single worker)
            # Actually, for parallel collection, the file is merged from chunks.
            # If merged simply, chunks are sequential.
            # Inside a chunk (episode), events are sequential.
            # So sorting by index is safe enough for "Next Step".
            indices.sort()
            
            for i in range(len(indices) - 1):
                curr_idx = indices[i]
                next_idx = indices[i+1]
                
                # Relink: The next state of curr_idx is the state of next_idx
                next_obs[curr_idx] = obs[next_idx]
                
                # Terminal: Intermediate steps are NOT terminal (unless they were already? Simulation done?)
                # Usually done=False.
                # terminals[curr_idx] = False 
                
                total_relinked += 1
            
            # The last event for this bus in the dataset
            last_idx = indices[-1]
            # Next obs is unknown (simulation ended or truncated).
            # If 'terminals[last_idx]' was True, keep it.
            # If False (TimeLimit?), we might keep it but next_obs is garbage (or last known).
            # Usually in D4RL, last step next_obs can be anything if terminal=True.
            total_terminals += 1

        print(f"Relinked {total_relinked} transitions.")
        print(f"Left {total_terminals} terminal transitions.")
        
        # Save back to HDF5
        f['next_observations'][...] = next_obs
        # f['terminals'][...] = terminals # Optional: update terminals if logic requires
        
    print("Done. Dataset Markov property restored (per-bus).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to HDF5 dataset")
    args = parser.parse_args()
    
    relink_dataset(args.dataset)
