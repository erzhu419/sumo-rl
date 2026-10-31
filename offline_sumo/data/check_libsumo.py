import os
import sys

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)

print("Checking libsumo...")
try:
    import libsumo
    print(f"Libsumo imported: {libsumo}")
    print(f"Has start? {hasattr(libsumo, 'start')}")
    if hasattr(libsumo, 'start'):
        print(f"libsumo.start doc: {libsumo.start.__doc__}")
except ImportError as e:
    print(f"Libsumo import failed: {e}")

print("\nChecking traci...")
try:
    import traci
    print(f"Traci imported: {traci}")
    print(f"Has start? {hasattr(traci, 'start')}")
except ImportError as e:
    print(f"Traci import failed: {e}")
