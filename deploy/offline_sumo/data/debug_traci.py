import os
import sys

print(f"SUMO_HOME: {os.environ.get('SUMO_HOME')}")
try:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import traci
    print(f"TraCI file: {traci.__file__}")
    print(f"TraCI dir: {dir(traci)}")
    if hasattr(traci, 'start'):
        print("traci.start exists")
    else:
        print("traci.start MISSING")
except Exception as e:
    print(f"Error importing traci: {e}")

try:
    import libsumo
    print(f"Libsumo imported: {libsumo}")
    print(f"Libsumo dir: {dir(libsumo)}")
except ImportError:
    print("Libsumo not found")
