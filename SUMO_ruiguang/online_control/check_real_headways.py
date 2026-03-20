import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SUMO_ruiguang.online_control.rl_bridge import build_bridge

def check_headways():
    bridge_info = build_bridge(
        root_dir=os.path.abspath('..'),
        gui=False,
    )
    # The bridge logic calculates headways in _load_objects which is called during build_bridge or similar
    # Wait, build_bridge creates a bridge, but objects are loaded during reset or manually.
    
    # Let's peek into the internal dictionary if possible
    # We need to simulate the load_objects logic
    import xml.etree.ElementTree as ET
    from SUMO_ruiguang.online_control.case import f_8_create_obj
    
    result = f_8_create_obj.create_obj_fun()
    bus_obj_dic = result[4]
    
    line_headways = {}
    line_bus_map = {}
    for bus_id, bus in bus_obj_dic.items():
        if bus.belong_line_id_s not in line_bus_map:
            line_bus_map[bus.belong_line_id_s] = []
        line_bus_map[bus.belong_line_id_s].append(bus)
        
    for line_id, buses in line_bus_map.items():
        if len(buses) > 1:
            buses.sort(key=lambda b: b.start_time_n)
            start_times = [b.start_time_n for b in buses]
            diffs = [j - i for i, j in zip(start_times[:-1], start_times[1:])]
            if diffs:
                line_headways[line_id] = float(np.median(diffs))
            else:
                line_headways[line_id] = 360.0
        else:
            line_headways[line_id] = 360.0
            
    print("Calculated Median Headways:")
    for l, h in line_headways.items():
        print(f"Line {l}: {h}s")
    
    print(f"\nAverage Median Headway: {np.mean(list(line_headways.values())):.2f}s")

if __name__ == "__main__":
    check_headways()
