import sys
import os
import time

# 确保脚本从正确的目录运行
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 添加项目路径以支持模块导入
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入模块
try:
    from g_no_control import create_obj
    from g_no_control import save_data
except ImportError:
    import create_obj
    import save_data

from sumolib import checkBinary
import traci


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

sumo_cfg_file = "no_control_sim_traci.sumocfg"
traci.start([sumoBinary, "-c", sumo_cfg_file])

"""准备工作"""
# 创建仿真各元素对象并字典存储
result = create_obj.create_obj_fun()
lane_obj_dic = result[0]
stop_obj_dic = result[1]
signal_obj_dic = result[2]
line_obj_dic = result[3]
bus_obj_dic = result[4]
passenger_obj_dic = result[5]
# 补充仿真各元素对象的静态变量
for stop_id in stop_obj_dic.keys():
    stop_obj_dic[stop_id].get_accessible_stop(line_obj_dic)
    stop_obj_dic[stop_id].get_passenger_arriver_rate(passenger_obj_dic)
for signal_id in signal_obj_dic.keys():
    signal_obj_dic[signal_id].get_attribute_by_traci()
    signal_obj_dic[signal_id].get_pass_line(line_obj_dic)
for bus_id in bus_obj_dic.keys():
    bus_obj_dic[bus_id].get_arriver_timetable(line_obj_dic[bus_obj_dic[bus_id].belong_line_id_s])

"""交通仿真"""
print("开始仿真！")
for step in range(0, 18000):
    """获取当前的仿真时间"""
    simulation_current_time = traci.simulation.getTime()
    """更新仿真各元素状态"""
    # 更新车道的状态
    for land_id in lane_obj_dic.keys():
        lane_obj_dic[land_id].update_lane_state(simulation_current_time)
    # 更新车站的状态
    for stop_id in stop_obj_dic.keys():
        stop_obj_dic[stop_id].update_stop_state()
    # 更新信号灯状态
    for signal_id in signal_obj_dic.keys():
        signal_obj_dic[signal_id].update_queue_number(simulation_current_time)
        if traci.trafficlight.getParameter(signal_id, "cycleSecond") == "0.00":
            signal_obj_dic[signal_id].update_signal_state(simulation_current_time, bus_obj_dic)
    # 更新线路的状态
    for line_id in line_obj_dic.keys():
        line_obj_dic[line_id].update_line_state(simulation_current_time)
    # 更新公交车状态
    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        if traci.vehicle.getTypeID(vehicle_id) == "Bus":
            if bus_obj_dic[vehicle_id].bus_state_s == "No":
                bus_obj_dic[vehicle_id].bus_activate(line_obj_dic[bus_obj_dic[vehicle_id].belong_line_id_s],
                                                     stop_obj_dic, signal_obj_dic, simulation_current_time)
            else:
                bus_obj_dic[vehicle_id].bus_running(line_obj_dic[bus_obj_dic[vehicle_id].belong_line_id_s],
                                                    stop_obj_dic, signal_obj_dic, passenger_obj_dic,
                                                    simulation_current_time)
    # 更新乘客的状态
    passenger_id_list = traci.person.getIDList()
    for passenger_id in passenger_id_list:
        if passenger_obj_dic[passenger_id].passenger_state_s == "No":
            passenger_obj_dic[passenger_id].passenger_activate(simulation_current_time, line_obj_dic)
        else:
            passenger_obj_dic[passenger_id].passenger_run(simulation_current_time, line_obj_dic)
    """执行下一步仿真动作"""
    traci.simulationStep()

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
# save_data.save_lane_data(lane_obj_dic)
save_data.save_stop_data(stop_obj_dic)
save_data.save_signal_data(signal_obj_dic)
# save_data.save_line_data(line_obj_dic)
save_data.save_bus_data(bus_obj_dic)
save_data.save_passenger_data(passenger_obj_dic)

end = time.perf_counter()
print("运行时间：", end - start)
