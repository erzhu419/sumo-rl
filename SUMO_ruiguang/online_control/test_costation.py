import sys
import os
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath("/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang"))
from online_control.case import f_8_create_obj

result = f_8_create_obj.create_obj_fun()
stop_obj_dic = result[1]
line_obj_dic = result[3]

for line_id, line in line_obj_dic.items():
    dist = 0.0
    print(f"Line: {line_id}")
    for stop_id in line.stop_id_l:
        dist += line.distance_between_stop_d.get(stop_id, 0.0)
        print(f"  Stop: {stop_id}, Distance from start = {dist}")
    break
