import os, sys

if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    sys.path.append("/usr/share/sumo/tools") # guess fallback

sys.path.append(os.path.abspath("/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang"))
from online_control.rl_env import SumoBusHoldingEnv
from online_control.rl_bridge import SumoRLBridge, build_bridge

bridge_funcs = build_bridge(root_dir="/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/online_control", sumo_cfg="control_sim_traci_period.sumocfg")
env = SumoBusHoldingEnv(
    root_dir="/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/online_control",
    schedule_file="/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/d_bus_rou/2_bus_timetable.rou.xml",
    decision_provider=bridge_funcs['decision_provider'],
    action_executor=bridge_funcs['action_executor'],
    reset_callback=bridge_funcs['reset_callback'],
    close_callback=bridge_funcs['close_callback'],
)

obs, rew, done = env.reset()
print(f"Initial Obs Keys: {obs.keys()}")
specs = env.get_feature_spec()
print(f"Feature spec: {specs}")

step = 0
while step < 10 and not done:
    actions = {}
    for line_id, buses in obs.items():
        actions[line_id] = {}
        for bus_id, history in buses.items():
            latest_obs = history[-1]
            print(f"Step {step} - {line_id}_{bus_id}: Length {len(latest_obs)}, Co_fwd={latest_obs[-3]:.1f}, Co_bwd={latest_obs[-2]:.1f}, Speed={latest_obs[-1]:.1f}")
            actions[line_id][bus_id] = [0.0, 1.0] # hold_value, speed_ratio
            
    obs, rew, done, _ = env.step(actions)
    step += 1

env.close()
print("Test completed successfully.")
