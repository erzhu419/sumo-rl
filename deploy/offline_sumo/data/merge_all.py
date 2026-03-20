"""
merge_all.py
Merges all per-policy HDF5 files in a directory into a single combined buffer.

Usage:
    python3 offline_sumo/data/merge_all.py \
        --input_dir offline_sumo/data \
        --output offline_sumo/data/buffer_combined.hdf5 \
        --pattern "buffer_*.hdf5"
"""

import argparse
import glob
import os
import numpy as np
import h5py


def merge_all(input_dir: str, output_path: str, pattern: str = "buffer_*.hdf5"):
    # Collect all matching HDF5 files, excluding the output file itself
    search = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search))
    output_abs = os.path.abspath(output_path)
    files = [f for f in files if os.path.abspath(f) != output_abs]

    if not files:
        print(f"[merge_all] No files matched '{search}'. Nothing to merge.")
        return

    print(f"[merge_all] Found {len(files)} file(s) to merge:")
    for f in files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")

    total_obs = []
    total_actions = []
    total_rewards = []
    total_next_obs = []
    total_terminals = []
    total_policy_types = []
    total_rows = 0

    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            obs = f['observations'][:]
            total_obs.append(obs)
            total_actions.append(f['actions'][:])
            total_rewards.append(f['rewards'][:])
            total_next_obs.append(f['next_observations'][:])
            total_terminals.append(f['terminals'][:])
            if 'policy_types' in f:
                total_policy_types.append(f['policy_types'][:])
            else:
                total_policy_types.append(np.zeros((len(obs), 1), dtype=np.float32))
            total_rows += len(obs)

    print(f"\n[merge_all] Total transitions to write: {total_rows:,}")

    all_obs         = np.concatenate(total_obs, axis=0)
    all_actions     = np.concatenate(total_actions, axis=0)
    all_rewards     = np.concatenate(total_rewards, axis=0)
    all_next_obs    = np.concatenate(total_next_obs, axis=0)
    all_terminals   = np.concatenate(total_terminals, axis=0)
    all_policy_types = np.concatenate(total_policy_types, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('observations',      data=all_obs,           compression='gzip', compression_opts=4)
        f.create_dataset('actions',           data=all_actions,       compression='gzip', compression_opts=4)
        f.create_dataset('rewards',           data=all_rewards,       compression='gzip', compression_opts=4)
        f.create_dataset('next_observations', data=all_next_obs,      compression='gzip', compression_opts=4)
        f.create_dataset('terminals',         data=all_terminals,     compression='gzip', compression_opts=4)
        f.create_dataset('policy_types',      data=all_policy_types,  compression='gzip', compression_opts=4)

    out_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[merge_all] Done! Written to: {output_path}  ({out_mb:.1f} MB)")
    print(f"  observations:      {all_obs.shape}")
    print(f"  actions:           {all_actions.shape}")
    print(f"  rewards:           {all_rewards.shape}")
    print(f"  policy_types:      {all_policy_types.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='offline_sumo/data',
                        help='Directory containing per-policy HDF5 files')
    parser.add_argument('--output', type=str, default='offline_sumo/data/buffer_combined.hdf5',
                        help='Output path for the merged HDF5')
    parser.add_argument('--pattern', type=str, default='buffer_*.hdf5',
                        help='Glob pattern to match input files')
    args = parser.parse_args()
    merge_all(args.input_dir, args.output, args.pattern)
