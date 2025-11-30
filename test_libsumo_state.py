import libsumo as traci
import os

# Create a dummy config if needed, or use existing one
# We'll use the user's existing config
SUMO_CFG = "/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/online_control/control_sim_traci_period.sumocfg"

try:
    print("Starting SUMO...")
    traci.start(["sumo", "-c", SUMO_CFG, "--no-warnings", "--log", "sumo_log.txt"])
    print("SUMO started.")
    
    print("Saving state to 'state.sbx'...")
    traci.simulation.saveState("state.sbx")
    print("State saved.")
    
    print("Advancing 10 steps...")
    for _ in range(10):
        traci.simulationStep()
    print(f"Time: {traci.simulation.getTime()}")
    
    print("Loading state 'state.sbx'...")
    traci.simulation.loadState("state.sbx")
    print("State loaded.")
    
    print(f"Time after load: {traci.simulation.getTime()}")
    if traci.simulation.getTime() == 0.0:
        print("SUCCESS: Time reset to 0.")
    else:
        print("FAILURE: Time not reset.")
        
    traci.close()
except Exception as e:
    print(f"ERROR: {e}")
