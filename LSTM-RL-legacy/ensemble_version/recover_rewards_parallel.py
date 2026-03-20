"""
Parallel batch evaluation of checkpoints 0-48 using 16 concurrent workers.
Each worker runs an independent SUMO simulation for one checkpoint.
Results are collected and patched back into rewards.npy.
"""

import os
import sys
import re
import time
import shutil
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = "Production_Augmented_BangBang_V7_Long"
EXPERIMENT_ID = f"sac_ensemble_SUMO_linear_penalty_{RUN_NAME}"
MODEL_DIR = os.path.join(BASE_DIR, "model", EXPERIMENT_ID)
LOG_DIR = os.path.join(BASE_DIR, "logs", EXPERIMENT_ID)
REWARDS_PATH = os.path.join(LOG_DIR, "rewards.npy")
RECOVERED_PATH = os.path.join(LOG_DIR, "rewards_recovered_parallel.npy")

PYTHON = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python3"
SOURCE = os.path.join(BASE_DIR, "sac_ensemble_SUMO_linear_penalty.py")

NUM_WORKERS = 16


def build_and_run_eval(episode: int):
    """Build a deterministic eval script for a specific episode and run it."""
    eval_script = os.path.join(BASE_DIR, f"_tmp_eval_ep{episode}.py")
    eval_log = os.path.join(LOG_DIR, f"recover_ep{episode}.log")
    run_name_ep = f"{RUN_NAME}_RECOVER_EP{episode}"

    # Build the eval script from the source
    with open(SOURCE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "for eps in range(start_episode, args.max_episodes):" in line:
            line = line.replace("for eps in range(start_episode, args.max_episodes):", "for eps in range(1):")

        if "episode_start_time = time.time()" in line:
            indent = line[:len(line) - len(line.lstrip())]
            ckpt = os.path.join(MODEL_DIR, f"checkpoint_episode_{episode}")
            inject = f"{indent}sac_trainer.load_model(r'{ckpt}')\n"
            new_lines.append(inject)

        if "DETERMINISTIC = False" in line:
            line = line.replace("DETERMINISTIC = False", "DETERMINISTIC = True")

        if "if len(replay_buffer) > args.batch_size and" in line:
            line = line.replace("if len(replay_buffer) >", "if False and len(replay_buffer) >")

        if "sac_trainer.save_model" in line:
            line = line.replace("sac_trainer.save_model", "pass # sac_trainer.save_model")

        new_lines.append(line)

    with open(eval_script, 'w') as f:
        f.writelines(new_lines)

    # Run the eval
    cmd = [
        PYTHON, eval_script,
        "--train",
        "--use_sumo_env",
        "--bang_bang",
        "--run_name", run_name_ep
    ]
    with open(eval_log, 'w') as logf:
        subprocess.run(cmd, stdout=logf, stderr=logf, cwd=BASE_DIR)

    # Parse reward
    reward = np.nan
    try:
        with open(eval_log, 'r') as logf:
            content = logf.read()
        match = re.search(r'Episode Reward:\s*([-\d.]+)', content)
        if match:
            reward = float(match.group(1))
    except Exception as e:
        print(f"[ep{episode}] ERROR reading log: {e}", flush=True)

    # Cleanup temp script
    try:
        os.remove(eval_script)
    except:
        pass

    print(f"[ep{episode}] -> Reward: {reward:.2f}", flush=True)
    return episode, reward


if __name__ == "__main__":
    # Find available checkpoints
    available = []
    for ep in range(49):
        policy_file = os.path.join(MODEL_DIR, f"checkpoint_episode_{ep}_policy")
        if os.path.exists(policy_file):
            available.append(ep)
        else:
            print(f"WARNING: Checkpoint ep{ep} not found, skipping.", flush=True)

    print(f"Found {len(available)} checkpoints: episodes {available[0]}-{available[-1]}", flush=True)
    print(f"Launching {NUM_WORKERS} parallel workers...", flush=True)

    results = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(build_and_run_eval, ep): ep for ep in available}
        for future in as_completed(futures):
            ep = futures[future]
            try:
                ep_result, reward = future.result()
                results[ep_result] = reward
            except Exception as e:
                print(f"[ep{ep}] EXCEPTION: {e}", flush=True)
                results[ep] = np.nan

    elapsed = time.time() - start_time
    print(f"\nAll done in {elapsed/60:.1f} minutes.", flush=True)

    # Build ordered array
    recovered_arr = np.array([results.get(ep, np.nan) for ep in range(49)])
    np.save(RECOVERED_PATH, recovered_arr)
    print(f"Saved recovered rewards to: {RECOVERED_PATH}")
    print(recovered_arr)

    # Patch rewards.npy
    if os.path.exists(REWARDS_PATH):
        full_rewards = np.load(REWARDS_PATH)
        print(f"\nOriginal rewards.npy: {len(full_rewards)} entries")
        if len(full_rewards) >= 49:
            full_rewards[:49] = recovered_arr
            np.save(REWARDS_PATH, full_rewards)
            print("Patched rewards.npy first 49 entries.")
        else:
            np.save(REWARDS_PATH, recovered_arr)
            print("rewards.npy had <49 entries; replaced entirely.")
    else:
        np.save(REWARDS_PATH, recovered_arr)
    print("Done.")
