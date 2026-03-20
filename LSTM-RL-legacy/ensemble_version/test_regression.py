import os
import argparse
import numpy as np
import sys

# This script attempts to replicate the "Best Heuristic" behavior by:
# 1. Forcing target headway to 360.0 (the bug-compatible value)
# 2. Setting update_freq to 10 (the previous default)
# 3. Disabling Berth Control (to match previous environment)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SUMO_ruiguang.online_control.rl_bridge import build_bridge
from SUMO_ruiguang.online_control.rl_env import SumoBusHoldingEnv

def main():
    # Build a "Legacy" bridge
    # Note: I'll have to manually override the headway behavior if I want it's bug-compatible
    # But for now, let's just see what happens if we use freq=10
    bridge = build_bridge(
        root_dir=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'),
        gui=False,
        update_freq=10, # Legacy frequency
        scale=1.0
    )
    
    # MANUALLY RE-INTRODUCE THE BUG (Lock all line headways to 360.0)
    # The bridge.get('decision_provider') uses the bridge's internal line_headways
    # I need to reach into the bridge object.
    # Fortunately, the 'bridge' returned by build_bridge is the object itself (wrapped in a dict/tuple)
    # or I can just re-read how build_bridge is implemented.
    
    # Actually, I'll just look at rl_bridge.py logic. 
    # The DecisionProvider is a closure.
    
    env = SumoBusHoldingEnv(
        root_dir=os.path.join(PROJECT_ROOT, 'SUMO_ruiguang/online_control'),
        schedule_file='initialize_obj/save_obj_bus.add.xml',
        decision_provider=bridge.get('decision_provider'),
        action_executor=bridge.get('action_executor'),
        reset_callback=bridge.get('reset_callback'),
        close_callback=bridge.get('close_callback'),
        reward_type="linear_penalty"
    )

    print("Starting Legacy Regression Test (Freq=10, Target=???)...")
    state_dict, reward_dict, done = env.reset()
    
    # We still need to force the 360s target in the hard rule logic below
    # because the DecisionEvent.target_forward_headway comes from the bridge.
    
    episode_reward = 0.0
    action_count = 0

    while not done:
        action_dict = {}
        for line_id, buses in state_dict.items():
            action_dict[line_id] = {}
            for bus_id, history in buses.items():
                if len(history) > 0:
                    event = env._pending_events.get((line_id, bus_id))
                    
                    # Manual override of target_forward_headway to 360.0 for regression
                    tgt_h = 360.0 
                    fwd_h = event.forward_headway
                    
                    if event and event.forward_bus_present:
                        if fwd_h > tgt_h:
                            hold_time = 0.0
                            speed_ratio = np.random.uniform(1.0, 1.2)
                        else:
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
        step_r = sum(sum(buses.values()) for buses in reward_dict.values() if isinstance(buses, dict))
        episode_reward += step_r

    print("========================================")
    print("Legacy Regression Test Finished!")
    print(f"Total Cumulative Reward: {episode_reward:.2f}")
    print("========================================")

if __name__ == '__main__':
    main()
