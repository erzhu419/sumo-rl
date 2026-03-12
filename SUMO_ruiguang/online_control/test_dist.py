import sys, os
import unittest.mock as mock
sys.modules['traci'] = mock.Mock()
sys.path.append(os.path.abspath("/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang"))
from online_control.case import f_8_create_obj

result = f_8_create_obj.create_obj_fun()
line_obj_dic = result[3]

for line_id, line in line_obj_dic.items():
    print(f"Line {line_id}:")
    dist = 0.0
    for stop_id in line.stop_id_l:
        val = line.distance_between_stop_d.get(stop_id, 0.0)
        dist += val
        print(f"  {stop_id}: interval={val}, accum={dist}")
    break
