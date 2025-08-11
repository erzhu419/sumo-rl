import sys
import os
# 添加父目录到路径以便导入sumo_adapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sumo_adapter as sumo


class Bus:    # 创建一个公交车类,用于描述每一个公交车的属性和行为

    def __init__(self, bus_id_s, start_time_n, belong_line_id_s):
        """初始化信号灯的函数"""
        """静态属性"""
        self.bus_id_s = bus_id_s                      # 公交车标签，字符串类型
        self.start_time_n = start_time_n              # 公交车发车时间，数值类型
        self.belong_line_id_s = belong_line_id_s      # 公交车所属线路标签，字符串类型
        self.arriver_timetable_d = {}                 # 公交车的到站时刻表，字典类型
        """动态属性"""
        self.bus_state_s = "No"                       # 公交车状态，字符串类型，No还没有产生；Edge在路段上；Stop在公交站；Signal在信号交叉口（触发更新）
        self.bus_speed_n = 0                          # 公交车速度，数值类型（每秒更新）
        self.distance_n = 0                           # 公交车行程，数值类型（每秒更新）
        self.timetable_deviation_n = 0                # 公交时刻表偏差，数值类型（到站更新）
        self.passenger_num_n = 0                      # 公交车乘客数，数值类型（离站更新）
        self.next_stop_id_s = ""                      # 下一个公交站标签，字符串类型，无下一公交站为“”（离站更新）
        self.next_stop_length_n = 0                   # 距下一个公交站停止线的长度，数值类型（每秒更新）
        self.next_signal_id_s = ""                    # 下一个信号灯标签，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_link_s = ""                  # 下一个信号灯所处连接，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_lane_s = ""                  # 下一个信号灯所处车道，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_phase_s = ""                 # 下一个信号灯所处相位，字符串类型，无下一信号灯为“”（离信号灯更新）
        self.next_signal_length_n = 0                 # 距下一个信号灯停止线的长度，数值类型（每秒更新）
        """容器属性"""
        self.bus_speed_l = []                         # 公交车的速度，列表类型（每秒更新） [速度值,速度值,...]
        self.distance_l = []                          # 公交车的行程，列表类型（每秒更新） [行驶距离,行驶距离,...]
        self.arriver_stop_time_d = {}                 # 公交车的到站时间，字典类型（到站更新） 公交站:时间
        self.depart_stop_time_d = {}                  # 公交车的离站时间，字典类型（离站更新） 公交站:时间
        self.alight_num_d = {}                        # 公交车的下车人数，字典类型（离站更新） 公交站:人数
        self.want_board_num_d = {}                    # 公交车的想要上车人数，字典类型（离站更新） 公交站:人数
        self.board_num_d = {}                         # 公交车的上车人数，字典类型（离站更新） 公交站:人数
        self.strand_num_d = {}                        # 公交车的滞留人数，字典类型（离站更新） 公交站:人数
        self.arriver_signal_time_d = {}               # 公交车信号灯排队开始时间，字典类型（到信号灯并进行排队才更新） 信号灯:时间
        self.depart_signal_time_d = {}                # 公交车信号灯排队结束时间，字典类型（离信号灯并存在排队才更新） 信号灯:时间
        self.no_stop_signal_id_list_l = []            # 没在信号灯排队的信号灯标签，列表类型（离信号灯排队更新） [信号灯标签,信号灯标签,...]
        self.stop_signal_id_list_l = []               # 有在信号灯排队的信号灯标签，列表类型（离信号灯排队更新） [信号灯标签,信号灯标签,...]

    def get_arriver_timetable(self, line_obj_ex):
        stop_id_list = line_obj_ex.stop_id_l
        edge_length_dic = line_obj_ex.edge_length_d
        edge_average_speed_dic = line_obj_ex.edge_average_speed_d
        edge_between_stop_dic = line_obj_ex.edge_between_stop_d
        total_time = self.start_time_n
        for stop_id in stop_id_list:
            for edge_id in edge_between_stop_dic[stop_id]:
                total_time = total_time + edge_length_dic[edge_id] / (edge_average_speed_dic[edge_id][int(self.start_time_n // 3600)] * 0.8)
            self.arriver_timetable_d[stop_id] = total_time

    def bus_activate(self, line_obj_ex, stop_obj_dic_ex, signal_obj_dic_ex, time_ex):
        self.bus_state_s = "Edge"
        self.bus_speed_n = 0
        self.distance_n = 0
        self.timetable_deviation_n = 0
        self.passenger_num_n = 0
        self.next_stop_id_s = line_obj_ex.stop_id_l[0]
        next_stop_length_n = 0
        if sumo.vehicle.getLaneID(self.bus_id_s) == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s:
            next_stop_length_n += (sumo.lane.getLength(stop_obj_dic_ex[self.next_stop_id_s].at_lane_s) -
                                   sumo.vehicle.getLanePosition(self.bus_id_s))
        else:
            next_stop_length_n += (sumo.lane.getLength(sumo.vehicle.getLaneID(self.bus_id_s)) -
                                   sumo.vehicle.getLanePosition(self.bus_id_s))
            for traci_lane_obj in sumo.vehicle.getNextLinks(self.bus_id_s):
                next_stop_length_n += sumo.lane.getLength(traci_lane_obj[0]) + traci_lane_obj[-1]
                if traci_lane_obj[0] == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s:
                    break
        self.next_stop_length_n = next_stop_length_n
        next_signal_list = sumo.vehicle.getNextTLS(self.bus_id_s)
        self.next_signal_id_s = next_signal_list[0][0]
        self.next_signal_link_s = str(next_signal_list[0][1])
        self.next_signal_lane_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][0]
        self.next_signal_phase_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][3]
        next_signal_length_n = 0
        if sumo.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
            next_signal_length_n += (sumo.lane.getLength(self.next_signal_lane_s) -
                                     sumo.vehicle.getLanePosition(self.bus_id_s))
        else:
            next_signal_length_n += (sumo.lane.getLength(sumo.vehicle.getLaneID(self.bus_id_s)) -
                                     sumo.vehicle.getLanePosition(self.bus_id_s))
            for traci_lane_obj in sumo.vehicle.getNextLinks(self.bus_id_s):
                next_signal_length_n += sumo.lane.getLength(traci_lane_obj[0]) + traci_lane_obj[-1]
                if traci_lane_obj[0] == self.next_signal_lane_s:
                    break
        self.next_signal_length_n = next_signal_length_n
        self.bus_speed_l.append(self.bus_speed_n)
        self.distance_l.append(self.distance_n)
        self.alight_num_d[self.next_stop_id_s] = 0

    def bus_running(self, line_obj_ex, stop_obj_dic_ex, signal_obj_dic_ex, passenger_obj_dic_ex, time_ex):
        # 如果上一步仿真，公交车在路段上
        if self.bus_state_s == "Edge":
            # 判断公交车是否到达公交站
            if sumo.vehicle.isAtBusStop(self.bus_id_s):
                self.bus_state_s = "Stop"
                self.timetable_deviation_n = time_ex - self.arriver_timetable_d[self.next_stop_id_s]
                self.arriver_stop_time_d[self.next_stop_id_s] = time_ex
            else:
                # 判断公交车是否到达信号灯（信号灯进口车道）
                if sumo.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
                    self.bus_state_s = "Signal"
        # 如果上一步仿真，公交车在公交站
        if self.bus_state_s == "Stop":
            # 判断公交车是否离开公交站
            if not sumo.vehicle.isAtBusStop(self.bus_id_s):
                # 判断公交车是否到达信号灯（信号灯进口车道）
                if sumo.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
                    self.bus_state_s = "Signal"
                else:
                    self.bus_state_s = "Edge"
                board_num = sumo.vehicle.getPersonNumber(self.bus_id_s) - (self.passenger_num_n - self.alight_num_d[self.next_stop_id_s])
                self.board_num_d[self.next_stop_id_s] = board_num
                strand_num = 0
                stop_passenger_list = sumo.busstop.getPersonIDs(self.next_stop_id_s)
                for passenger_id in stop_passenger_list:
                    if self.belong_line_id_s in passenger_obj_dic_ex[passenger_id].passable_line_l:
                        strand_num += 1
                self.strand_num_d[self.next_stop_id_s] = strand_num
                want_board_num = board_num + strand_num
                self.want_board_num_d[self.next_stop_id_s] = want_board_num
                self.depart_stop_time_d[self.next_stop_id_s] = time_ex
                stop_obj_dic_ex[self.next_stop_id_s].update_service_data(self)    # 更新公交站的属性
                self.passenger_num_n = sumo.vehicle.getPersonNumber(self.bus_id_s)
                self.next_stop_id_s = line_obj_ex.get_next_stop_id_by_this_stop_id(self.next_stop_id_s)
                if self.next_stop_id_s != "":
                    alight_num = 0
                    bus_passenger_list = sumo.vehicle.getPersonIDList(self.bus_id_s)
                    for passenger_id in bus_passenger_list:
                        if passenger_obj_dic_ex[passenger_id].transfer_stop_id_s == self.next_stop_id_s or passenger_obj_dic_ex[passenger_id].end_stop_id_s == self.next_stop_id_s:
                            alight_num += 1
                    self.alight_num_d[self.next_stop_id_s] = alight_num
        # 如果上一步仿真，公交车在信号灯（信号灯进口车道）
        if self.bus_state_s == "Signal":
            # 判断公交车是否离开信号灯（信号灯进口车道）
            next_signal_list = sumo.vehicle.getNextTLS(self.bus_id_s)
            if len(next_signal_list) == 0 or next_signal_list[0][0] != self.next_signal_id_s:
                # 记录排队结束时间
                if self.next_signal_id_s in self.arriver_signal_time_d.keys() and self.next_signal_id_s not in self.depart_signal_time_d.keys():
                    self.depart_signal_time_d[self.next_signal_id_s] = time_ex
                signal_obj_dic_ex[self.next_signal_id_s].update_service_data(self)    # 更新信号灯的属性
                if len(next_signal_list) == 0:
                    self.bus_state_s = "Edge"
                    self.next_signal_id_s = ""
                    self.next_signal_link_s = ""
                    self.next_signal_lane_s = ""
                    self.next_signal_phase_s = ""
                else:
                    self.next_signal_id_s = next_signal_list[0][0]
                    self.next_signal_link_s = str(next_signal_list[0][1])
                    self.next_signal_lane_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][0]
                    self.next_signal_phase_s = signal_obj_dic_ex[self.next_signal_id_s].connection_d[self.next_signal_link_s][3]
                    if sumo.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
                        self.bus_state_s = "Signal"
                    else:
                        self.bus_state_s = "Edge"
            else:
                # 判断公交车是否在排队
                if sumo.vehicle.getSpeed(self.bus_id_s) <= 0.1:
                    # 如果有排队记录还存在<0.1的情况，说明上一次不是最后一次最接近信号灯的排队，需要删除更新排队
                    if self.next_signal_id_s in self.arriver_signal_time_d.keys() and self.next_signal_id_s in self.depart_signal_time_d.keys():
                        del self.arriver_signal_time_d[self.next_signal_id_s]
                        del self.depart_signal_time_d[self.next_signal_id_s]
                    # 记录排队开始时间
                    if self.next_signal_id_s not in self.arriver_signal_time_d.keys():
                        self.arriver_signal_time_d[self.next_signal_id_s] = time_ex
                else:
                    # 记录排队结束时间
                    if self.next_signal_id_s in self.arriver_signal_time_d.keys() and self.next_signal_id_s not in self.depart_signal_time_d.keys():
                        self.depart_signal_time_d[self.next_signal_id_s] = time_ex
        # 更新每秒都需更新的状态
        self.bus_speed_n = sumo.vehicle.getSpeed(self.bus_id_s)
        self.distance_n = sumo.vehicle.getDistance(self.bus_id_s)
        if self.next_stop_id_s != "":
            next_stop_length_n = 0
            if sumo.vehicle.getLaneID(self.bus_id_s) == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s:
                next_stop_length_n += (sumo.lane.getLength(stop_obj_dic_ex[self.next_stop_id_s].at_lane_s) -
                                       sumo.vehicle.getLanePosition(self.bus_id_s))
            else:
                next_stop_length_n += (sumo.lane.getLength(sumo.vehicle.getLaneID(self.bus_id_s)) -
                                       sumo.vehicle.getLanePosition(self.bus_id_s))
                for traci_lane_obj in sumo.vehicle.getNextLinks(self.bus_id_s):
                    next_stop_length_n += sumo.lane.getLength(traci_lane_obj[0]) + traci_lane_obj[-1]
                    if traci_lane_obj[0] == stop_obj_dic_ex[self.next_stop_id_s].at_lane_s:
                        break
            self.next_stop_length_n = next_stop_length_n
        else:
            self.next_stop_length_n = -1
        if self.next_signal_id_s != "":
            next_signal_length_n = 0
            if sumo.vehicle.getLaneID(self.bus_id_s) == self.next_signal_lane_s:
                next_signal_length_n += (sumo.lane.getLength(self.next_signal_lane_s) -
                                         sumo.vehicle.getLanePosition(self.bus_id_s))
            else:
                next_signal_length_n += (sumo.lane.getLength(sumo.vehicle.getLaneID(self.bus_id_s)) -
                                         sumo.vehicle.getLanePosition(self.bus_id_s))
                for traci_lane_obj in sumo.vehicle.getNextLinks(self.bus_id_s):
                    next_signal_length_n += sumo.lane.getLength(traci_lane_obj[0]) + traci_lane_obj[-1]
                    if traci_lane_obj[0] == self.next_signal_lane_s:
                        break
            self.next_signal_length_n = next_signal_length_n
        else:
            self.next_signal_length_n = -1
        self.bus_speed_l.append(self.bus_speed_n)
        self.distance_l.append(self.distance_n)

    def bus_end(self, line_obj_ex):
        self.stop_signal_id_list_l = self.arriver_signal_time_d.keys()
        for signal_id in line_obj_ex.signal_id_l:
            if signal_id not in self.stop_signal_id_list_l:
                self.no_stop_signal_id_list_l.append(signal_id)
