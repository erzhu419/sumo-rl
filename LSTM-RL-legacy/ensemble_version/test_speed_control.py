import sys
import os
import torch
import numpy as np

os.environ['SUMO_HOME'] = '/usr/share/sumo'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from SUMO_ruiguang.online_control.rl_bridge import build_bridge
bridge_dict = build_bridge(root_dir=os.path.join(project_root, "SUMO_ruiguang", "online_control"))

decision_provider = bridge_dict['decision_provider']
action_executor = bridge_dict['action_executor']

from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv
env = SumoBusHoldingEnv(
    root_dir=os.path.join(project_root, "SUMO_ruiguang", "online_control"),
    schedule_file=os.path.join(project_root, 'SUMO_ruiguang', 'd_bus_rou', '2_bus_timetable.rou.xml'),
    decision_provider=decision_provider,
    action_executor=action_executor,
    debug=True,
    reward_type="linear_penalty"
)

print("Environment built, resetting...")
obs = env.reset()

print("Initial observation keys:", obs.keys())

# Step the environment with dummy 2D actions
action_dict = {}
for line_id, buses in obs.items():
    action_dict[line_id] = {}
    for bus_id in buses.keys():
        action_dict[line_id][bus_id] = np.array([30.0, 1.1]) # 30s holding, +10% speed

print("Stepping with 2D actions:")
print(action_dict)

next_obs, rewards, done, info = env.step(action_dict)

print("Step completed. Rewards keys:", rewards.keys())

# Optional: read max speed to verify if it's 1.1 * 13.89
import traci
try:
    if len(action_dict) > 0:
        first_line = list(action_dict.keys())[0]
        first_bus = list(action_dict[first_line].keys())[0]
        max_speed = traci.vehicle.getMaxSpeed(first_bus)
        print(f"Max speed of {first_bus}: {max_speed:.2f} m/s (Expected: {13.89*1.1:.2f})")
except Exception as e:
    print("Could not query traci directly:", e)

env.close()
print("Test successful!")
