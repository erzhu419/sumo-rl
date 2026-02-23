import sys
import numpy as np

def extract_rewards(log_file):
    with open(log_file, "r") as f:
        return [float(line.strip().split("Reward: ")[1].split(" |")[0]) for line in f if "Episode Reward" in line]

import subprocess
import os

print("Running Vanilla 0-action test...")
subprocess.run("python sac_zero_vanilla.py --train --use_sumo_env --max_episodes 1 > log_v.txt 2>&1", shell=True)
print("Running Ensemble 0-action test...")
subprocess.run("python sac_zero_ensemble.py --train --use_sumo_env --max_episodes 1 > log_e.txt 2>&1", shell=True)

v_reward = extract_rewards("log_v.txt")
e_reward = extract_rewards("log_e.txt")

print(f"Vanilla: {v_reward}")
print(f"Ensemble: {e_reward}")

# Check action logs for trajectory comparisons
import pandas as pd
df_v = pd.read_csv('action_log.csv')

# Backup the vanilla action log
subprocess.run("cp action_log.csv action_log_vanilla.csv", shell=True)

# Run Ensemble and capture the action log
subprocess.run("python sac_zero_ensemble.py --train --use_sumo_env --max_episodes 1 > log_e_again.txt 2>&1", shell=True)
subprocess.run("cp action_log.csv action_log_ensemble.csv", shell=True)

df_e = pd.read_csv('action_log_ensemble.csv')
print("Vanilla action rows:", len(df_v))
print("Ensemble action rows:", len(df_e))
