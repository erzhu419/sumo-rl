
import os
import sys

def test_imports():
    print("Testing import order...")
    
    # Ensure tools in path
    if "SUMO_HOME" in os.environ:
         tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
         if tools_path not in sys.path:
             sys.path.append(tools_path)

    try:
        import libsumo
        print("PASS: libsumo imported cleanly at start.")
    except ImportError as e:
        print(f"FAIL: libsumo failed at start: {e}")
        return

def test_order(first, second, desc):
    print(f"\nTesting Order: {first} -> {second} ({desc})...")
    import subprocess
    cmd = [
        sys.executable, "-c",
        f"import os, sys; "
        f"sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools')); "
        f"print('Importing {first}...'); "
        f"import {first}; "
        f"print('Importing {second}...'); "
        f"import {second}; "
        f"print('SUCCESS: Both imported.')"
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print(f"FAIL: {first} -> {second} crashed.")

if __name__ == "__main__":
    test_imports()
    test_order("gym", "libsumo", "Checking if Gym breaks Libsumo")
    test_order("libsumo", "torch", "Should work if libsumo first?")
    test_order("torch", "libsumo", "Known to fail")
