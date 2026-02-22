import torch
import os

model_path = "logs/sac_v2_bus_SUMO_best_result/sac_v2_episode_18_policy"
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found")
    exit(1)

state_dict = torch.load(model_path, map_location='cpu')

print("--- Model State Dict Shapes ---")
for k, v in state_dict.items():
    print(f"{k}: {v.shape}")
