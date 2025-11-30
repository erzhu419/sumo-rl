import xml.etree.ElementTree as ET
import os

def verify_mapping():
    route_file = "SUMO_ruiguang/d_bus_rou/2_bus_timetable.rou.xml"
    if not os.path.exists(route_file):
        print(f"Error: {route_file} not found.")
        return

    tree = ET.parse(route_file)
    root = tree.getroot()

    # Find 311S_1
    vehicle = None
    for veh in root.findall('vehicle'):
        if veh.get('id') == '311S_1':
            vehicle = veh
            break
    
    if vehicle is None:
        print("Vehicle 311S_1 not found.")
        return

    stops = vehicle.findall('stop')
    print(f"Stops for 311S_1 (Total: {len(stops)}):")
    for i, stop in enumerate(stops):
        print(f"Index {i}: {stop.get('busStop')} (duration={stop.get('duration')})")

if __name__ == "__main__":
    verify_mapping()
