# 功能：与SUMO连接
# 时间：2025.02.24

# "../intersection_delay/a_sorted_busline_edge.xml"中的路段距离，不包括连接部分，以及交叉口部分，故计算到站时间存在一定的误差

import sys
import os

# 添加当前目录到Python路径以支持模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import f_8_create_obj
import e_8_gurobi_test_considerbusnum_V3
import d_8_compute_running_time
import h_8_save_data
from sumolib import checkBinary
import xml.etree.ElementTree as ET
import traci
import time
import math
import pickle
import tkinter as tk  # Tkinter 需要在主线程初始化


# 解决 Tkinter 线程错误
try:
    tk_root = tk.Tk()
    tk_root.withdraw()  # 隐藏窗口，防止 GUI 影响
except Exception as e:
    print("Tkinter 初始化失败:", e)


start = time.perf_counter()

"""配置接口"""
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'!")

if_show_gui = True
if not if_show_gui:
    sumoBinary = checkBinary("sumo")
    print(sumoBinary)
else:
    sumoBinary = checkBinary("sumo-gui")
    print(sumoBinary)

# 使用绝对路径避免工作目录问题
online_control_dir = os.path.dirname(current_dir)
sumo_cfg_file = os.path.join(online_control_dir, "control_sim_traci_period.sumocfg")
traci.start([sumoBinary, "-c", sumo_cfg_file])

"""准备工作"""
# 创建仿真各元素对象并字典存储
result = f_8_create_obj.create_obj_fun()
lane_obj_dic = result[0]
stop_obj_dic = result[1]
signal_obj_dic = result[2]
line_obj_dic = result[3]
bus_obj_dic = result[4]
passenger_obj_dic = result[5]

"""获取静态信息"""
bus_line_list, station_dict, station_interval_dis_dict, scaled_line_station_od_otd_dict, line_station_od_otd_dict, bus_arrstation_od_otd_dict, \
    od_otd_arr_rate_dict, BusCap, AveAlightingTime, AveBoardingTime = e_8_gurobi_test_considerbusnum_V3.get_static_info(line_obj_dic)

"""获取各公交线路排好顺序的edge字典"""
edge_file_path = os.path.join(online_control_dir, "intersection_delay", "a_sorted_busline_edge.xml")
edge_file = ET.parse(edge_file_path)
sorted_busline_edge_d, involved_tl_ID_l, busline_stop_edge_d, busline_tl_time_d = d_8_compute_running_time.get_sorted_busline_edge(edge_file)

involved_signal_d = {}

"""触发时间"""
trigger_time = 900
first_stop_num = 30  # 暂时考虑所有车站

"""补充仿真各元素对象的静态变量"""
for stop_id in stop_obj_dic.keys():
    stop_obj_dic[stop_id].get_accessible_stop(line_obj_dic)
    stop_obj_dic[stop_id].get_passenger_arriver_rate(passenger_obj_dic)
    stop_obj_dic[stop_id].get_initial_just_leave_data(line_station_od_otd_dict, trigger_time)
for signal_id in signal_obj_dic.keys():
    signal_obj_dic[signal_id].get_attribute_by_traci()
    signal_obj_dic[signal_id].get_pass_line(line_obj_dic)
for bus_id in bus_obj_dic.keys():
    bus_obj_dic[bus_id].get_arriver_timetable(line_obj_dic[bus_obj_dic[bus_id].belong_line_id_s])

"""交通仿真"""
print("开始仿真！")

# 初始化各车站已经进行控制的公交车ID
stop_controled_bus_d = {}
# 为周期乘客到达率之前的触发车站，虽然这个触发车站在当前周期性乘客分布下，不一定是车头时距大的，但是不影响，结果还可以，所以证明与车站无关，与车站数量，即触发频率有关，对应数据“Paper2_Code_2-3 - 20250312\i_with_control_delete_partial_OD_VC\trajectory\control_sim_traci_period_0413PeriodVC”
trigger_stop_l = ["102X17_122S19_311S25_406X17", "311X31", "122X21_311X21_406S28", "406X35", "102X16_122S18_311S24_406X16", "102S23_122X18_311X18_406S25", "122X22_311X22_406S29"]  # 串车headway为150，排队长度为4

for trigger_stop in trigger_stop_l:
    stop_controled_bus_d[trigger_stop] = []

bus_speed_curve = {}
GurobimodelNoSolutionFlag_d = {}
over_control_flag = 0
Variable_Num = []  # 用于在公交站考虑不同公交车数量问题下，保留不同数量下模型的变量数量信息
for step in range(0, 18000):
    """获取当前的仿真时间"""
    simulation_current_time = traci.simulation.getTime()
    """更新仿真各元素状态"""
    # 更新车道的状态
    # for land_id in lane_obj_dic.keys():
    #     lane_obj_dic[land_id].update_lane_state(simulation_current_time)
    # 更新车站的状态
    for stop_id in stop_obj_dic.keys():
        stop_obj_dic[stop_id].update_stop_state()
    # 更新信号灯状态
    # for signal_id in signal_obj_dic.keys():
    #     signal_obj_dic[signal_id].update_queue_number(simulation_current_time)
    #     if traci.trafficlight.getParameter(signal_id, "cycleSecond") == "0.00":
    #         signal_obj_dic[signal_id].update_signal_state(simulation_current_time, bus_obj_dic)
    # 更新线路的状态
    # for line_id in line_obj_dic.keys():
    #     line_obj_dic[line_id].update_line_state(simulation_current_time)
    # 更新公交车状态
    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        if traci.vehicle.getTypeID(vehicle_id) == "Bus":
            if bus_obj_dic[vehicle_id].bus_state_s == "No":
                bus_obj_dic[vehicle_id].bus_activate(line_obj_dic[bus_obj_dic[vehicle_id].belong_line_id_s],
                                                     stop_obj_dic, signal_obj_dic, simulation_current_time)
            else:
                BusCap = 50
                AveAlightingTime = 1.5
                AveBoardingTime = 2.5

                bus_obj_dic[vehicle_id].bus_running(line_obj_dic[bus_obj_dic[vehicle_id].belong_line_id_s],
                                                    stop_obj_dic, signal_obj_dic, passenger_obj_dic,
                                                    simulation_current_time, BusCap, AveAlightingTime,
                                                    AveBoardingTime, bus_arrstation_od_otd_dict, bus_obj_dic,
                                                    involved_tl_ID_l, sorted_busline_edge_d)




    # 更新乘客的状态
    passenger_id_list = traci.person.getIDList()
    for passenger_id in passenger_id_list:
        if passenger_obj_dic[passenger_id].passenger_state_s == "No":
            passenger_obj_dic[passenger_id].passenger_activate(simulation_current_time, line_obj_dic)
        else:
            passenger_obj_dic[passenger_id].passenger_run(simulation_current_time, line_obj_dic)

    # 如果触发过控制模型，则按照上次模型结果进行车辆控制
    if over_control_flag == 1:
        for bus_id in bus_speed_curve:
            if bus_id in vehicle_id_list:
                bus_current_edge = traci.vehicle.getRoadID(bus_id)
                if bus_current_edge in bus_speed_curve[bus_id]:
                    traci.vehicle.setSpeed(bus_id, 10)
                else:  # 将速度控制权返回给SUMO
                    traci.vehicle.setSpeed(bus_id, -1)

    """执行下一步仿真动作"""
    traci.simulationStep()

    """每1000s自动保存一次数据"""
    if simulation_current_time > 0 and simulation_current_time % 1000 == 0:
        h_8_save_data.save_lane_data(lane_obj_dic)
        h_8_save_data.save_stop_data(stop_obj_dic)
        h_8_save_data.save_signal_data(signal_obj_dic)
        h_8_save_data.save_line_data(line_obj_dic)
        h_8_save_data.save_bus_data(bus_obj_dic)
        h_8_save_data.save_passenger_data(passenger_obj_dic)


        output_file_path = os.path.join(online_control_dir, "output_file", "VariableNumInfo.pkl")
        with open(output_file_path, "wb") as f:
            pickle.dump(Variable_Num, f)


vehicle_id_list = traci.vehicle.getIDList()
for bus_id in bus_obj_dic.keys():
    if bus_id not in vehicle_id_list:
        bus_obj_dic[bus_id].bus_end(line_obj_dic[bus_obj_dic[bus_id].belong_line_id_s])
passenger_id_list = traci.person.getIDList()
for passenger_id in passenger_obj_dic.keys():
    if passenger_id not in passenger_id_list:
        passenger_obj_dic[passenger_id].passenger_end()
traci.close()
print("结束仿真")

"""数据保存"""
h_8_save_data.save_lane_data(lane_obj_dic)
h_8_save_data.save_stop_data(stop_obj_dic)
h_8_save_data.save_signal_data(signal_obj_dic)
h_8_save_data.save_line_data(line_obj_dic)
h_8_save_data.save_bus_data(bus_obj_dic)
h_8_save_data.save_passenger_data(passenger_obj_dic)

end = time.perf_counter()
print("运行时间：", end - start)

# 保存变量数量信息
output_file_path = os.path.join(online_control_dir, "output_file", "VariableNumInfo.pkl")
with open(output_file_path, "wb") as f:
    pickle.dump(Variable_Num, f)

debug = 0
