import sys
import os
import time
QUEUE_UPDATE_INTERVAL = 20
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
import sumo_adapter as sumo


start = time.perf_counter()

"""配置接口"""
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'!")

if_show_gui = False
if not if_show_gui:
    sumoBinary = checkBinary("sumo")
    print(sumoBinary)
else:
    sumoBinary = checkBinary("sumo-gui")
    print(sumoBinary)

sumo_cfg_file = "no_control_sim_traci.sumocfg"
sumo.start([sumoBinary, "-c", sumo_cfg_file, "--threads", "16"])

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
# 1. 站点与线路的关系建立
for stop_id in stop_obj_dic.keys():
    stop_obj_dic[stop_id].get_accessible_stop(line_obj_dic)
    stop_obj_dic[stop_id].get_passenger_arriver_rate(passenger_obj_dic)
# 2. 信号灯与线路的关系建立  
for signal_id in signal_obj_dic.keys():
    signal_obj_dic[signal_id].get_attribute_by_traci()
    signal_obj_dic[signal_id].get_pass_line(line_obj_dic)
# 3. 公交车与线路的关系建立
for bus_id in bus_obj_dic.keys():
    bus_obj_dic[bus_id].get_arriver_timetable(line_obj_dic[bus_obj_dic[bus_id].belong_line_id_s])

"""交通仿真"""
print("开始仿真！")
for step in range(0, 18000):
    """获取当前的仿真时间"""
    simulation_current_time = sumo.simulation.getTime()
    """更新仿真各元素状态"""
    # # 更新车道的状态：统计车道上的车辆数量、平均速度、交通流量等实时交通参数
    # for land_id in lane_obj_dic.keys():
    #     lane_obj_dic[land_id].update_lane_state(simulation_current_time)
    # # 更新车站的状态：统计站台乘客数量、等车公交车数量，记录乘客上下车信息
    # for stop_id in stop_obj_dic.keys():
    #     stop_obj_dic[stop_id].update_stop_state()
    # # 1. 信号灯状态更新（每个周期）：统计各进口道排队车辆数，如果信号灯处于周期开始时刻则更新信号状态
    for signal_id in signal_obj_dic.keys():
        if simulation_current_time % QUEUE_UPDATE_INTERVAL == 0:
            signal_obj_dic[signal_id].update_queue_number(simulation_current_time)  # 更新排队车辆数统计
        if sumo.trafficlight.getParameter(signal_id, "cycleSecond") == "0.00":  # 判断是否为周期开始
            signal_obj_dic[signal_id].update_signal_state(simulation_current_time, bus_obj_dic)  # 更新信号灯控制策略
    # # 更新线路的状态：统计线路各路段的车辆数、流量、速度、密度等交通运行指标
    # for line_id in line_obj_dic.keys():
    #     line_obj_dic[line_id].update_line_state(simulation_current_time)
    # 更新公交车状态：处理公交车的激活、运行过程，包括站点服务、信号灯交互等
    # 优化：只获取一次车辆列表，然后只处理已知的公交车
    vehicle_id_list = sumo.vehicle.getIDList() # SUMO自动管理车辆激活
    vehicle_id_set = set(vehicle_id_list)  # 转为集合以提高查找效率
    # 2. 公交车状态管理（时间驱动激活）
    for bus_id in bus_obj_dic.keys():
        # 只处理仍在仿真中的公交车
        if bus_id in vehicle_id_set:
            if bus_obj_dic[bus_id].bus_state_s == "No":  # 首次激活,在bus_activate里会设置bus_state_s为其他
                bus_obj_dic[bus_id].bus_activate(line_obj_dic[bus_obj_dic[bus_id].belong_line_id_s],
                                                 stop_obj_dic, signal_obj_dic, simulation_current_time)  # 激活公交车，初始化运行状态
            else:  # 公交车已激活，正在运行
                bus_obj_dic[bus_id].bus_running(line_obj_dic[bus_obj_dic[bus_id].belong_line_id_s],
                                                stop_obj_dic, signal_obj_dic, passenger_obj_dic,
                                                simulation_current_time)  # 更新公交车运行状态，处理到站、离站、载客等
    # 3. 乘客状态管理（时间驱动激活）
    passenger_id_list = sumo.person.getIDList()
    for passenger_id in passenger_id_list:
        if passenger_obj_dic[passenger_id].passenger_state_s == "No":  # 乘客尚未激活
            passenger_obj_dic[passenger_id].passenger_activate(simulation_current_time, line_obj_dic)  # 激活乘客，开始出行
        else:  # 乘客已激活，正在出行中
            passenger_obj_dic[passenger_id].passenger_run(simulation_current_time, line_obj_dic)  # 更新乘客状态，处理等车、上车、下车、换乘等
    """执行下一步仿真动作：让SUMO仿真前进一个时间步长（1秒）"""
    sumo.simulationStep()  # SUMO仿真引擎执行一个仿真步长，更新所有车辆、行人的位置和状态

vehicle_id_list = sumo.vehicle.getIDList()
for bus_id in bus_obj_dic.keys():
    if bus_id not in vehicle_id_list:
        bus_obj_dic[bus_id].bus_end(line_obj_dic[bus_obj_dic[bus_id].belong_line_id_s])
passenger_id_list = sumo.person.getIDList()
for passenger_id in passenger_obj_dic.keys():
    if passenger_id not in passenger_id_list:
        passenger_obj_dic[passenger_id].passenger_end()
sumo.close()
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

