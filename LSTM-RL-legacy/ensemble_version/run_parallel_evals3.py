import os
import sys
import subprocess

run_name = "Production_Augmented_BangBang_V7_Long"
model_dir = f"model/sac_ensemble_SUMO_linear_penalty_{run_name}"
log_dir = f"logs/sac_ensemble_SUMO_linear_penalty_{run_name}"

episodes = list(range(49))
num_workers = 14
chunks = [episodes[i::num_workers] for i in range(num_workers)]

# Create wrapper scripts that call the main script
for w in range(num_workers):
    eps_to_run = chunks[w]
    if not eps_to_run: continue
    
    script_content = f"""
import os
import sys
import numpy as np
import torch

# Import everything from the main script!
from sac_ensemble_SUMO_linear_penalty import *

# We already have `env` and `sac_trainer` from the main script's global scope!
# But we need to make sure it doesn't run the training loop.
"""
    # Wait, if we import *, the main script runs its training loop because it's not guarded by if __name__ == '__main__' perfectly maybe.
