import os, sys, time

if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)

sys.path.append(os.path.abspath("/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang"))
from online_control.rl_env import SumoBusHoldingEnv
from online_control.rl_bridge import SumoRLBridge, build_bridge

# Use gui=True for visualization
bridge_funcs = build_bridge(
    root_dir="/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/online_control", 
    sumo_cfg="control_sim_traci_period.sumocfg",
    gui=True
)

env = SumoBusHoldingEnv(
    root_dir="/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/online_control",
    schedule_file="/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/d_bus_rou/2_bus_timetable.rou.xml",
    decision_provider=bridge_funcs['decision_provider'],
    action_executor=bridge_funcs['action_executor'],
    reset_callback=bridge_funcs['reset_callback'],
    close_callback=bridge_funcs['close_callback'],
)

obs, rew, done = env.reset()
print(f"Simulation started with GUI. Looking for capacity evictions...")

step = 0
max_steps = 1000 # Run longer to increase chance of seeing full stations

while step < max_steps and not done:
    actions = {}
    for line_id, buses in obs.items():
        actions[line_id] = {}
        for bus_id, history in buses.items():
            # Force a VERY LONG holding time (120s) to trigger capacity bottlenecks intentionally
            actions[line_id][bus_id] = [1.0, 1.0] 
            
    obs, rew, done, _ = env.step(actions)
    
    # Small sleep to make it easier to follow in GUI if needed
    # time.sleep(0.01)
    
    step += 1

env.close()
print("Visualization test completed.")
