import traci


class Stop:    # 创建一个公交站类,用于描述每一个公交站的属性和行为

    def __init__(self, stop_id_s, at_edge_s, at_lane_s, service_line_l):
        """初始化公交站的函数"""
        """静态属性"""
        self.stop_id_s = stop_id_s               # 公交站标签，字符串类型
        self.at_edge_s = at_edge_s               # 公交站所在的路段，字符串类型
        self.at_lane_s = at_lane_s               # 公交站所在的车道，字符串类型
        self.service_line_l = service_line_l     # 公交站可以服务的线路，列表类型 [线路标签,线路标签,...]
        self.accessible_stop_d = {}              # 不需要换乘可直接到达的公交站，字典类型 下游公交站:[线路标签,线路标签,...]
        self.passenger_arriver_rate_d = {}       # 公交站的乘客需求（每小时去往每个下游站的到达率），字典类型 下游公交站:[第1小时到达率,第2小时到达率,...]
        """动态属性"""
        self.passenger_num_n = 0                 # 公交站的站内乘客数量，数值类型（触发更新）
        self.passenger_id_l = []                 # 公交站的站内乘客标签，列表类型（触发更新） [乘客标签,乘客标签,...]
        self.bus_num_n = 0                       # 公交站的站内公交车数量，数值类型（触发更新）
        self.bus_id_l = []                       # 公交站的站内公交车标签，列表类型（触发更新） [公交车标签,公交车标签,...]
        self.just_leave_data_d = {}              # 每条线路刚刚服务公交车的信息，字典类型（离站更新） 线路标签:[公交车标签,线路标签,到达时间,离开时间]
        """容器属性"""
        self.service_data_l = []                 # 服务公交车的信息，列表类型（离站更新） [公交车标签,线路标签,到达时间,离开时间,下车乘客数,上车乘客数,滞留乘客数]

    def get_accessible_stop(self, line_obj_dic_ex):
        accessible_stop_list = []
        for line_id in line_obj_dic_ex.keys():
            if self.stop_id_s in line_obj_dic_ex[line_id].stop_id_l:
                this_stop_id = self.stop_id_s
                while line_obj_dic_ex[line_id].get_next_stop_id_by_this_stop_id(this_stop_id) != "":
                    this_stop_id = line_obj_dic_ex[line_id].get_next_stop_id_by_this_stop_id(this_stop_id)
                    if this_stop_id not in accessible_stop_list:
                        accessible_stop_list.append(this_stop_id)
        for stop_id in accessible_stop_list:
            available_line_list = []
            for line_id in line_obj_dic_ex.keys():
                if self.stop_id_s in line_obj_dic_ex[line_id].stop_id_l and stop_id in line_obj_dic_ex[line_id].stop_id_l:
                    available_line_list.append(line_id)
            self.accessible_stop_d[stop_id] = available_line_list

    def get_passenger_arriver_rate(self, passenger_obj_dic_ex):
        for stop_id in self.accessible_stop_d.keys():
            self.passenger_arriver_rate_d[stop_id] = [0, 0, 0, 0, 0]
        for passenger_id in passenger_obj_dic_ex.keys():
            if passenger_obj_dic_ex[passenger_id].start_stop_id_s == self.stop_id_s:
                start_time = passenger_obj_dic_ex[passenger_id].start_time_n
                if passenger_obj_dic_ex[passenger_id].transfer_stop_id_s == "":
                    alight_stop = passenger_obj_dic_ex[passenger_id].end_stop_id_s
                else:
                    alight_stop = passenger_obj_dic_ex[passenger_id].transfer_stop_id_s
                self.passenger_arriver_rate_d[alight_stop][int(start_time // 3600)] += 1
            if passenger_obj_dic_ex[passenger_id].transfer_stop_id_s == self.stop_id_s:
                start_time = passenger_obj_dic_ex[passenger_id].start_time_n
                alight_stop = passenger_obj_dic_ex[passenger_id].end_stop_id_s
                self.passenger_arriver_rate_d[alight_stop][int(start_time // 3600)] += 1
        for stop_id in self.accessible_stop_d.keys():
            self.passenger_arriver_rate_d[stop_id] = [n / 3600 for n in self.passenger_arriver_rate_d[stop_id]]

    def update_stop_state(self):    # 控制需要更新
        self.passenger_id_l = traci.busstop.getPersonIDs(self.stop_id_s)
        self.passenger_num_n = len(self.passenger_id_l)
        self.bus_id_l = traci.busstop.getVehicleIDs(self.stop_id_s)
        self.bus_num_n = len(self.bus_id_l)

    def update_service_data(self, bus_obj_ex):    # 公交离站更新
        just_leave_data = [bus_obj_ex.bus_id_s, bus_obj_ex.belong_line_id_s, bus_obj_ex.arriver_stop_time_d[self.stop_id_s], bus_obj_ex.depart_stop_time_d[self.stop_id_s]]
        self.just_leave_data_d[bus_obj_ex.belong_line_id_s] = just_leave_data
        # service_data = just_leave_data + [bus_obj_ex.alight_num_d[self.stop_id_s], bus_obj_ex.board_num_d[self.stop_id_s], bus_obj_ex.strand_num_d[self.stop_id_s]]
        service_data = just_leave_data
        self.service_data_l.append(service_data)
