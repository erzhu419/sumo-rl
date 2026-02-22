
import sys
import os
import argparse

# Add RORL to path
RORL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../RORL"))
if RORL_PATH not in sys.path:
    sys.path.append(RORL_PATH)

# Add Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from offline_sumo.envs.sumo_env import SumoBusHoldingEnv

def test_env():
    print("Testing Env...")
    env = SumoBusHoldingEnv(gui=False, max_steps=100)
    print("Env created.")
    
    # Pre-check traci
    try:
        import traci
        print(f"Traci imported from: {traci.__file__}")
        print(f"Traci dir: {dir(traci)}")
    except ImportError:
        print("Traci import failed directly.")

    try:
        obs = env.reset()
        print("Env reset successful.")
        print(f"Obs shape: {obs.shape}")
        
        # Check traci
        from offline_sumo.envs.bridge_v2 import traci
        print(f"Traci module: {traci}")
        if hasattr(traci, 'start'):
            print("Traci has start()")
        else:
            print("Traci DOES NOT have start()")
            print(f"Traci dir: {dir(traci)}")
            
    except Exception as e:
        print(f"Env reset failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env()
