from g_no_control.sim_obj import signal
import xml.etree.ElementTree as ET


def get_lane_length_by_lane_id(net_root_ex, lane_id_ex):
    """根据车道标签寻找该车道的长度"""
    lane_length = 0
    for child in net_root_ex:
        if child.tag == "edge" and child.get("id") == lane_id_ex[0:-2]:
            for sub_child in child:
                if sub_child.get("id") == lane_id_ex:
                    lane_length = float(sub_child.get("length"))
                    break
            break
    if lane_length == 0:
        print("警告，initialize_signal的get_lane_length_by_lane_id没找到该车道！")
    return lane_length


def get_last_lane_by_lane_id(net_root_ex, lane_id_ex):
    """根据车道标签寻找上一个以直行连接到达该车道的车道，并且两个车道之间的连接不受信号灯控制"""
    # 判断该车道的起始点是否被信号灯控制
    from_junction = ""
    for child in net_root_ex:
        if child.tag == "edge" and child.get("id") == lane_id_ex[0:-2]:
            from_junction = child.get("from")
    if from_junction == "":
        print("警告，initialize_signal的get_last_lane_by_lane_id没找到起始节点！")
    from_junction_if_signal = "no"
    for child in net_root_ex:
        if child.tag == "junction" and child.get("id") == from_junction and child.get("type") == "traffic_light":
            from_junction_if_signal = "yes"
    if from_junction_if_signal == "yes":
        return []
    # 寻找上一个以直行连接到达该车道的车道
    last_lane_list = []
    for child in net_root_ex:
        if child.tag == "connection" and child.get("to") == lane_id_ex[0:-2] and child.get("toLane") == lane_id_ex[-1] and child.get("dir") == "s" and child.get("state") == "M" and child.get("from")[0] != ":":
            last_lane_list.append(child.get("from") + "_" + child.get("fromLane"))
    return last_lane_list


def get_signal_structure_by_signal_id(net_root_ex, signal_id_ex):
    in_edge_list = []             # 带逆时针顺序的进口道列表
    out_edge_list = []            # 带逆时针顺序的出口道列表
    in_add_out_edge_list = []     # 不带顺序的交叉口进口道和出口道列表
    # 通过xml文件寻找in_edge_list和in_add_out_edge_list
    for child in net_root_ex:
        if child.tag == "junction" and child.get("id") == signal_id_ex:
            in_lane_list = child.get("incLanes").split(" ")
            for in_lane in in_lane_list:
                if in_lane[:-2] not in in_edge_list:
                    in_edge_list.append(in_lane[:-2])
        if child.tag == "connection" and child.get("tl") == signal_id_ex:
            if child.get("from") not in in_add_out_edge_list:
                in_add_out_edge_list.append(child.get("from"))
            if child.get("to") not in in_add_out_edge_list:
                in_add_out_edge_list.append(child.get("to"))
    # 通过xml文件找出进口道in_edge_list对应的出口道out_edge_list
    for in_edge in in_edge_list:
        can_to_edge_list = []
        for child in net_root_ex:
            if child.tag == "connection" and child.get("from") == in_edge and child.get("dir") != "t":
                if child.get("to") not in can_to_edge_list:
                    can_to_edge_list.append(child.get("to"))
        no_to_edge_list = []
        for edge in in_add_out_edge_list:
            if edge not in in_edge_list and edge not in can_to_edge_list:
                no_to_edge_list.append(edge)
        if len(no_to_edge_list) == 1:
            out_edge_list.append(no_to_edge_list[0])
        else:
            print(signal_id_ex, in_edge, no_to_edge_list)
    # 准备好双环结构相位与右转相位 "north": 1, "east": 2, "south": 3, "west": 4
    double_ring_phase = [[2, 3], [4, 2], [3, 4], [1, 3], [4, 1], [2, 4], [1, 2], [3, 1]]
    right_phase = [[1, 4], [2, 1], [3, 2], [4, 3]]
    # 通过xml文件提取相关的connection信息
    connection_xml_list = []
    for child in net_root_ex:
        if child.tag == "connection" and child.get("tl") == signal_id_ex:
            connection_xml_list.append(child)
    # 构造所需连接信息 数据形式为，连接编号：[进入车道、驶出车道、连接编号、相位编号]
    connection_dic = {}
    for child in connection_xml_list:
        enter_lane = child.get("from") + "_" + child.get("fromLane")
        exit_lane = child.get("to") + "_" + child.get("toLane")
        from_orientation = in_edge_list.index(child.get("from")) + 1
        to_orientation = out_edge_list.index(child.get("to")) + 1
        phase_index = 0
        if [from_orientation, to_orientation] in double_ring_phase:
            phase_index = double_ring_phase.index([from_orientation, to_orientation]) + 1
        elif [from_orientation, to_orientation] in right_phase:
            phase_index = 9
        else:
            print("该连接找不到相位：", child.get("from"), child.get("to"), child.get("fromLane"), child.get("toLane"))
        connection_dic[child.get("linkIndex")] = [enter_lane, exit_lane, child.get("linkIndex"), str(phase_index)]
    # 构造所需相位信息 数据形式为，相位编号：[[进入车道、驶出车道、连接编号、相位编号],[进入车道、驶出车道、连接编号、相位编号]，...]
    phase_dic = {}
    for key in connection_dic.keys():
        if connection_dic[key][3] not in phase_dic.keys():
            phase_dic[connection_dic[key][3]] = [connection_dic[key]]
        else:
            phase_dic[connection_dic[key][3]].append(connection_dic[key])
    # 构造进入的车道（延申到上一交叉口或大于500米小于1000米）字典 数据形式为，车道1：[车道1，车道2，...]
    in_lane_id_dic = {}
    for key in connection_dic.keys():
        if connection_dic[key][0] not in in_lane_id_dic.keys():
            in_lane_id = [connection_dic[key][0]]
            lane_total_length = get_lane_length_by_lane_id(net_root_ex, connection_dic[key][0])
            this_lane_id = [connection_dic[key][0]]
            while lane_total_length < 500:
                last_lane_id = []
                for lane_id in this_lane_id:
                    last_lane_id = last_lane_id + get_last_lane_by_lane_id(net_root_ex, lane_id)
                if len(last_lane_id) == 0:
                    break
                else:
                    in_lane_id = in_lane_id + last_lane_id
                    for lane_id in last_lane_id:
                        lane_total_length = lane_total_length + get_lane_length_by_lane_id(net_root_ex, lane_id)
                    this_lane_id = last_lane_id
            in_lane_id_dic[connection_dic[key][0]] = in_lane_id
    # 构造驶出的车道字典 数据形式为，车道1：[车道1]
    out_lane_id_dic = {}
    for key in connection_dic.keys():
        if connection_dic[key][0] not in out_lane_id_dic.keys():
            out_lane_id_dic[connection_dic[key][1]] = [connection_dic[key][1]]
    return in_lane_id_dic, out_lane_id_dic, connection_dic, phase_dic


def initialize_signal_obj(net_path_ex, detector_path_ex):
    net_file = ET.parse(net_path_ex)
    net_root = net_file.getroot()
    detector_data = []
    with open(detector_path_ex, "r") as file:
        for row in file:
            detector = list(map(str, row.strip().split('\t')))
            detector_data.append(detector)
    signal_detector_dic = {}
    for detector in detector_data:
        if detector[0] not in signal_detector_dic.keys():
            signal_detector_dic[detector[0]] = {}
        signal_detector_dic[detector[0]][detector[1]] = detector[2]
    signal_obj_dic = {}
    for child in net_root:
        if child.tag == "tlLogic":
            signal_structure = get_signal_structure_by_signal_id(net_root, child.get("id"))
            signal_obj = signal.Signal(child.get("id"), signal_structure[0], signal_structure[1],
                                       signal_detector_dic[child.get("id")], signal_structure[2], signal_structure[3])
            signal_obj_dic[child.get("id")] = signal_obj
    return signal_obj_dic


signal_obj_dic = initialize_signal_obj("../../b_network/1_changsha_bus_network_with_signal_d.net.xml",
                                       "../../b_network/5_E1_signal_detector.txt")
# 创建XML根元素
root = ET.Element("signal_obj_xml")
# 添加换行和空格
root.text = "\n\t"
for signal_id in signal_obj_dic.keys():
    signal_obj = ET.SubElement(root, "signal_obj")
    signal_obj.set("signal_id_s", signal_obj_dic[signal_id].signal_id_s)
    signal_obj.text = "\n\t\t"
    in_lane = ET.SubElement(signal_obj, "in_lane")
    for in_lane_id in signal_obj_dic[signal_id].in_lane_d.keys():
        in_lane.set(in_lane_id, ' '.join(signal_obj_dic[signal_id].in_lane_d[in_lane_id]))
    in_lane.tail = "\n\t\t"
    out_lane = ET.SubElement(signal_obj, "out_lane")
    for out_lane_id in signal_obj_dic[signal_id].out_lane_d.keys():
        out_lane.set(out_lane_id, ' '.join(signal_obj_dic[signal_id].out_lane_d[out_lane_id]))
    out_lane.tail = "\n\t\t"
    detector = ET.SubElement(signal_obj, "detector")
    for detector_id in signal_obj_dic[signal_id].detector_d.keys():
        detector.set(detector_id, signal_obj_dic[signal_id].detector_d[detector_id])
    detector.tail = "\n\t\t"
    connection = ET.SubElement(signal_obj, "connection")
    for connection_id in signal_obj_dic[signal_id].connection_d.keys():
        connection.set("c_" + connection_id, ' '.join(signal_obj_dic[signal_id].connection_d[connection_id]))
    connection.tail = "\n\t\t"
    phase = ET.SubElement(signal_obj, "phase")
    for phase_id in signal_obj_dic[signal_id].phase_d:
        phase.set("p_" + phase_id, ' '.join([';'.join(n) for n in signal_obj_dic[signal_id].phase_d[phase_id]]))
    phase.tail = "\n\t"
    signal_obj.tail = "\n\t"
# 创建XML文件并写入数据
xml_tree = ET.ElementTree(root)
xml_file = "save_obj_signal.add.xml"
xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
print(f"信号灯对象 XML 文件已生成。")
