import h5py
import numpy as np
import os
import argparse

def merge_buffers(output_file, input_files):
    print(f"Merging {len(input_files)} files into {output_file}...")
    
    # Initialize lists to hold data
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_terminals = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue
            
        print(f"Reading {file_path}...")
        try:
            with h5py.File(file_path, 'r') as f:
                all_obs.append(f['observations'][:])
                all_actions.append(f['actions'][:])
                all_rewards.append(f['rewards'][:])
                all_next_obs.append(f['next_observations'][:])
                all_terminals.append(f['terminals'][:])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not all_obs:
        print("No data loaded. Exiting.")
        return

    # Concatenate
    combined_obs = np.concatenate(all_obs, axis=0)
    combined_actions = np.concatenate(all_actions, axis=0)
    combined_rewards = np.concatenate(all_rewards, axis=0)
    combined_next_obs = np.concatenate(all_next_obs, axis=0)
    combined_terminals = np.concatenate(all_terminals, axis=0)

    print(f"Total samples: {len(combined_obs)}")

    # Save
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('observations', data=combined_obs, compression="gzip")
        f.create_dataset('actions', data=combined_actions, compression="gzip")
        f.create_dataset('rewards', data=combined_rewards, compression="gzip")
        f.create_dataset('next_observations', data=combined_next_obs, compression="gzip")
        f.create_dataset('terminals', data=combined_terminals, compression="gzip")
        
    print(f"Successfully saved merged buffer to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="offline_sumo/data/merged_buffer_all.hdf5")
    args = parser.parse_args()

    # Define the specific files to merge
    base_dir = "offline_sumo/data"
    files_to_merge = [
        "buffer_zero_parallel.hdf5",
        "buffer_expert_15_parallel.hdf5",
        "buffer_expert_18_parallel.hdf5",
        "buffer_expert_20_parallel.hdf5",
        "buffer_expert_25_parallel.hdf5",
        "buffer_expert_30_parallel.hdf5"
    ]
    
    # Resolve absolute paths assuming script is run from project root, or handle relative
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check if files exist relative to where script is running vs absolute
    # The prompt implies we are in project root often, but let's be robust
    input_paths = []
    for fname in files_to_merge:
        # Try relative to script location if base_dir relative fails
        p = os.path.join(base_dir, fname)
        if not os.path.exists(p):
             # Try absolute relative to project root constructed above
             p_abs = os.path.join(project_root, "offline_sumo", "data", fname)
             if os.path.exists(p_abs):
                 p = p_abs
        input_paths.append(p)
        
    merge_buffers(args.output, input_paths)
