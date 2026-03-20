import torch

path = '/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/best model/checkpoint_episode_39_policy'
state_dict = torch.load(path, map_location='cpu')

print("Loaded policy state_dict. Keys & Shapes:")
for k, v in state_dict.items():
    print(f"{k}: {v.shape}")

import numpy as np
path_rewards = '/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/logs/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/rewards.npy'
data = np.load(path_rewards)
print("\nLatest rewards (35-45):")
for i in range(35, min(48, len(data))):
    print(f"Episode {i}: {data[i]}")
