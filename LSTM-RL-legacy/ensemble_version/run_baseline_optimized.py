import os
import argparse
import numpy as np
import sys

# Match sac_ensemble import style to enable libsumo if available
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    import traci

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SUMO_ruiguang.online_control.rl_bridge import build_bridge
from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumo_gui', action='store_true', help='Run with SUMO GUI')
    parser.add_argument('--traffic_scale', type=float, default=1.0, help='Scale of social vehicles')
    parser.add_argument('--passenger_update_freq', type=int, default=1, help='Frequency of passenger state updates (1=every step, high precision)')
    args = parser.parse_args()

    # Build the bridge components using the highly optimized configuration
    # update_freq=1 ensures all passengers are captured and berth logic is ultra-precise
    bridge = build_bridge(
        root_dir=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'),
        gui=args.sumo_gui,
        update_freq=args.passenger_update_freq,
        scale=args.traffic_scale
    )
    
    decision_provider = bridge.get('decision_provider')
    action_executor = bridge.get('action_executor')
    reset_cb = bridge.get('reset_callback')
    close_cb = bridge.get('close_callback')

    # Initialize Environment wrapper
    env = SumoBusHoldingEnv(
        root_dir=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'),
        schedule_file='initialize_obj/save_obj_bus.add.xml',
        decision_provider=decision_provider,
        action_executor=action_executor,
        reset_callback=reset_cb,
        close_callback=close_cb,
        reward_type="linear_penalty"
    )

    print(f"Starting Optimized Baseline Simulation (Update Freq: {args.passenger_update_freq}, Traffic Scale: {args.traffic_scale})...")
    state_dict, reward_dict, done = env.reset()
    
    episode_reward = 0.0
    steps = 0
    action_count = 0

    while not done:
        action_dict = {}
        for line_id, buses in state_dict.items():
            action_dict[line_id] = {}
            for bus_id, history in buses.items():
                if len(history) > 0:
                    # Retrieve the underlying decision event from the environment 
                    event = env._pending_events.get((line_id, bus_id))
                    
                    if event and event.forward_bus_present and event.target_forward_headway > 0:
                        fwd_h = event.forward_headway
                        tgt_h = event.target_forward_headway
                        
                        # Optimized Hard Rules:
                        if fwd_h > tgt_h:
                            # Behind schedule: Speed up
                            hold_time = 0.0
                            speed_ratio = np.random.uniform(1.0, 1.2)
                        else:
                            # Catching up: Hold at station
                            # Stochastic holding to prevent platoon formation
                            hold_time = np.random.uniform(0.0, 60.0)
                            speed_ratio = 1.0
                    else:
                        hold_time = 0.0
                        speed_ratio = 1.0

                    action_dict[line_id][bus_id] = [hold_time, speed_ratio]
                    action_count += 1
                else:
                    action_dict[line_id][bus_id] = None

        state_dict, reward_dict, done, _ = env.step(action_dict)
        
        # Accumulate reward
        step_r = sum(sum(buses.values()) for buses in reward_dict.values() if isinstance(buses, dict))
        episode_reward += step_r
        steps += 1

    print("========================================")
    print("Optimized Baseline Simulation Finished!")
    print(f"Total Simulation Actions Taken: {action_count}")
    print(f"Total Cumulative Reward: {episode_reward:.2f}")
    print("========================================")

if __name__ == '__main__':
    main()
