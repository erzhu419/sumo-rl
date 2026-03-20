"""
Batch deterministic evaluation of checkpoints 0-48 from Production_Augmented_BangBang_V7_Long.
Reconstructs the true rewards for the first 49 episodes and injects them back into rewards.npy.

Strategy:
- Evaluate each checkpoint (episode 0 to 48) using deterministic inference
- Collect the rewards in order
- Patch the first 49 entries of the existing rewards.npy with the recovered values
"""

import os
import sys
import subprocess
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = "Production_Augmented_BangBang_V7_Long"
EXPERIMENT_ID = f"sac_ensemble_SUMO_linear_penalty_{RUN_NAME}"
MODEL_DIR = os.path.join(BASE_DIR, "model", EXPERIMENT_ID)
LOG_DIR = os.path.join(BASE_DIR, "logs", EXPERIMENT_ID)
REWARDS_PATH = os.path.join(LOG_DIR, "rewards.npy")
RECOVERED_PATH = os.path.join(LOG_DIR, "rewards_recovered.npy")

PYTHON = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python3"
EVAL_SCRIPT = os.path.join(BASE_DIR, "eval_batch_ep.py")

# Build the parameterized eval script (once)
def build_eval_script(episode: int):
    """Generate the deterministic eval script targeting a specific episode checkpoint."""
    source = os.path.join(BASE_DIR, "sac_ensemble_SUMO_linear_penalty.py")
    with open(source, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Run only 1 episode
        if "for eps in range(start_episode, args.max_episodes):" in line:
            line = line.replace("for eps in range(start_episode, args.max_episodes):", "for eps in range(1):")

        # Inject load_model
        if "episode_start_time = time.time()" in line:
            indent = line[:len(line) - len(line.lstrip())]
            ckpt = os.path.join(MODEL_DIR, f"checkpoint_episode_{episode}")
            inject = f"{indent}sac_trainer.load_model(r'{ckpt}')\n"
            new_lines.append(inject)

        # Deterministic mode
        if "DETERMINISTIC = False" in line:
            line = line.replace("DETERMINISTIC = False", "DETERMINISTIC = True")

        # No gradient updates
        if "if len(replay_buffer) > args.batch_size and" in line:
            line = line.replace("if len(replay_buffer) >", "if False and len(replay_buffer) >")

        # No checkpoint saving
        if "sac_trainer.save_model" in line:
            line = line.replace("sac_trainer.save_model", "pass # sac_trainer.save_model")

        new_lines.append(line)

    with open(EVAL_SCRIPT, 'w') as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    # Detect which checkpoints we have
    available = []
    for ep in range(49):  # episodes 0-48
        policy_file = os.path.join(MODEL_DIR, f"checkpoint_episode_{ep}_policy")
        if os.path.exists(policy_file):
            available.append(ep)
        else:
            print(f"WARNING: Checkpoint for episode {ep} not found, skipping.")

    print(f"Found {len(available)} checkpoints: {available}")

    recovered = {}
    for ep in available:
        print(f"\n=== Evaluating episode {ep} ===")
        build_eval_script(ep)

        eval_log = os.path.join(LOG_DIR, f"recover_ep{ep}.log")
        cmd = [
            PYTHON, EVAL_SCRIPT,
            "--train",
            "--use_sumo_env",
            "--bang_bang",
            "--run_name", f"{RUN_NAME}_RECOVER_EP{ep}"
        ]
        with open(eval_log, 'w') as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=logf, cwd=BASE_DIR)

        # Parse reward from log
        try:
            with open(eval_log, 'r') as logf:
                content = logf.read()
            match = re.search(r'Episode Reward:\s*([-\d.]+)', content)
            if match:
                reward = float(match.group(1))
                recovered[ep] = reward
                print(f"  -> Reward: {reward:.2f}")
            else:
                print(f"  -> WARNING: Could not parse reward for ep {ep}")
        except Exception as e:
            print(f"  -> ERROR reading log: {e}")

    # Save recovered rewards
    recovered_arr = np.array([recovered.get(ep, np.nan) for ep in range(49)])
    np.save(RECOVERED_PATH, recovered_arr)
    print(f"\n=== Recovery complete! Saved to {RECOVERED_PATH} ===")
    print(recovered_arr)

    # Patch the first 49 entries of the main rewards.npy
    if os.path.exists(REWARDS_PATH):
        full_rewards = np.load(REWARDS_PATH)
        print(f"\nOriginal rewards.npy has {len(full_rewards)} entries")
        if len(full_rewards) >= 49:
            full_rewards[:49] = recovered_arr
            np.save(REWARDS_PATH, full_rewards)
            print(f"Patched rewards.npy: first 49 entries updated.")
        else:
            print("rewards.npy has fewer than 49 entries, saving recovered array as the main file")
            np.save(REWARDS_PATH, recovered_arr)
    else:
        np.save(REWARDS_PATH, recovered_arr)
    print("Done.")
