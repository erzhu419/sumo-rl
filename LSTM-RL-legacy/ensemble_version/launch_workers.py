import os
import subprocess

episodes = list(range(49))
num_workers = 14
chunks = [episodes[i::num_workers] for i in range(num_workers)]

for w in range(num_workers):
    eps_to_run = chunks[w]
    if not eps_to_run: continue
    
    ep_str = ",".join(map(str, eps_to_run))
    cmd = f"EPS_TO_RUN='{ep_str}' WORKER_ID={w} nohup python3 recover_rewards_standalone.py > worker_{w}.log 2>&1 &"
    os.system(cmd)
    
print("Launched 14 standalone evaluation workers.")
