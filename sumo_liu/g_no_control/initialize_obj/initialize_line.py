from g_no_control.sim_obj import line
import xml.etree.ElementTree as ET


def get_edge_length_by_edge_id(net_root_ex, edge_id_ex):
    """根据路段标签寻找该路段的长度"""
    edge_length = 0
    for child in net_root_ex:
        if child.tag == "edge" and child.get("id") == edge_id_ex:
            for sub_child in child:
                if sub_child.tag == "lane":
                    edge_length = float(sub_child.get("length"))
                    break
            break
    if edge_length == 0:
        print("警告，initialize_line的get_edge_length_by_edge_id没找到该路段！")
    return edge_length


def get_edge_id_by_stop_id(stop_root_ex, stop_id_ex):
    """根据公交站标签寻找该公交站所在路段"""
    edge_id = " "
    for child in stop_root_ex:
        if child.tag == "busStop" and child.get("id") == stop_id_ex:
            edge_id = child.get("lane")[0:-2]
            break
    if edge_id == " ":
        print("警告，initialize_line的get_edge_id_by_stop_id没找到路段！")
    return edge_id


def get_distance_between_edge(edge_id_list_ex, edge_length_dic_ex, start_edge_ex, end_edge_ex):
    """根据线路上两个路段标签获得两个路段之间的距离，开始的路段不记录距离，结束的路段记录距离"""
    total_length = 0
    edge_list = []
    if_add_length = "no"
    for edge_id in edge_id_list_ex:
        if if_add_length == "yes":
            total_length = total_length + edge_length_dic_ex[edge_id]
            edge_list.append(edge_id)
        if edge_id == start_edge_ex:
            if_add_length = "yes"
        if edge_id == end_edge_ex:
            break
    if total_length == 0:
        print("警告，initialize_line的get_distance_between_stop没获得距离！")
    return total_length, edge_list


def initialize_line_obj(net_path_ex, stop_path_ex, bus_rou_path_ex, detector_output_path_1_ex, detector_output_path_2_ex):
    net_file = ET.parse(net_path_ex)
    net_root = net_file.getroot()
    stop_file = ET.parse(stop_path_ex)
    stop_root = stop_file.getroot()
    bus_rou_file = ET.parse(bus_rou_path_ex)
    bus_rou_root = bus_rou_file.getroot()
    detector_output_1_file = ET.parse(detector_output_path_1_ex)
    detector_output_1_root = detector_output_1_file.getroot()
    detector_output_2_file = ET.parse(detector_output_path_2_ex)
    detector_output_2_root = detector_output_2_file.getroot()
    """构建所有线路对象"""
    line_obj_dic = {}
    bus_line_list = []
    for child in bus_rou_root:
        if child.tag == "vehicle":
            for sub_child in child:
                if sub_child.tag == "stop" and sub_child.get("line") not in bus_line_list:
                    bus_line_list.append(sub_child.get("line"))
    for bus_line in bus_line_list:
        for child in bus_rou_root:
            if_jump_loop = "no"
            for sub_child in child:
                if sub_child.tag == "stop" and sub_child.get("line") == bus_line:
                    edge_id_list = child[0].get("edges").split(" ")    # -----
                    edge_length_dic = {}    # -----
                    for edge_id in edge_id_list:
                        edge_length_dic[edge_id] = get_edge_length_by_edge_id(net_root, edge_id)
                    stop_id_list = []    # -----
                    for child_stop in child:
                        if child_stop.tag == "stop":
                            stop_id_list.append(child_stop.get("busStop"))
                    distance_between_stop_dic = {}    # -----
                    edge_between_stop_dic = {}    # -----
                    for n in range(0, len(stop_id_list)):
                        if n == 0:
                            start_edge = edge_id_list[0]
                            end_edge = get_edge_id_by_stop_id(stop_root, stop_id_list[n])
                            result_split_by_stop = get_distance_between_edge(edge_id_list, edge_length_dic, start_edge, end_edge)
                            distance_between_stop_dic[stop_id_list[n]] = edge_length_dic[start_edge] + result_split_by_stop[0]
                            edge_between_stop_dic[stop_id_list[n]] = [start_edge] + result_split_by_stop[1]
                        else:
                            start_edge = get_edge_id_by_stop_id(stop_root, stop_id_list[n-1])
                            end_edge = get_edge_id_by_stop_id(stop_root, stop_id_list[n])
                            result_split_by_stop = get_distance_between_edge(edge_id_list, edge_length_dic, start_edge, end_edge)
                            distance_between_stop_dic[stop_id_list[n]] = result_split_by_stop[0]
                            edge_between_stop_dic[stop_id_list[n]] = result_split_by_stop[1]
                    junction_id_list = []
                    junction_last_edge_id_list = []
                    for edge_id in edge_id_list:
                        for child_edge in net_root:
                            if child_edge.tag == "edge" and child_edge.get("id") == edge_id:
                                junction_id_list.append(child_edge.get("to"))
                                junction_last_edge_id_list.append([child_edge.get("to"), edge_id])
                                break
                    signal_id_list = []    # -----
                    signal_last_edge_id_list = []
                    for n in range(0, len(junction_id_list)):
                        for child_junction in net_root:
                            if child_junction.tag == "junction" and child_junction.get("id") == junction_id_list[n] and child_junction.get("type") == "traffic_light":
                                signal_id_list.append(junction_id_list[n])
                                signal_last_edge_id_list.append(junction_last_edge_id_list[n][1])
                    distance_between_signal_dic = {}    # -----
                    edge_between_signal_dic = {}    # -----
                    for n in range(0, len(signal_id_list)):
                        if n == 0:
                            start_edge = edge_id_list[0]
                            end_edge = signal_last_edge_id_list[n]
                            result_split_by_signal = get_distance_between_edge(edge_id_list, edge_length_dic, start_edge, end_edge)
                            distance_between_signal_dic[signal_id_list[n]] = edge_length_dic[start_edge] + result_split_by_signal[0]
                            edge_between_signal_dic[signal_id_list[n]] = [start_edge] + result_split_by_signal[1]
                        else:
                            start_edge = signal_last_edge_id_list[n-1]
                            end_edge = signal_last_edge_id_list[n]
                            result_split_by_signal = get_distance_between_edge(edge_id_list, edge_length_dic, start_edge, end_edge)
                            distance_between_signal_dic[signal_id_list[n]] = result_split_by_signal[0]
                            edge_between_signal_dic[signal_id_list[n]] = result_split_by_signal[1]
                    edge_average_speed_dic = {}    # -----
                    for edge_id in edge_id_list:
                        speed_list_every_hour = [[], [], [], [], []]
                        for child_interval in detector_output_1_root:
                            if child_interval.tag == "interval" and child_interval.get("id")[3:-2] == edge_id and float(child_interval.get("speed")) != -1:
                                speed_list_every_hour[int(float(child_interval.get("begin")) // 3600)].append(float(child_interval.get("speed")))
                        for child_interval in detector_output_2_root:
                            if child_interval.tag == "interval" and child_interval.get("id")[3:-2] == edge_id and float(child_interval.get("speed")) != -1:
                                speed_list_every_hour[int(float(child_interval.get("begin")) // 3600)].append(float(child_interval.get("speed")))
                        edge_average_speed_dic[edge_id] = []
                        for n in speed_list_every_hour:
                            if len(n) == 0:
                                edge_average_speed_dic[edge_id].append(13.89)
                                print(edge_id, speed_list_every_hour)
                            else:
                                edge_average_speed_dic[edge_id].append(sum(n)/len(n))
                    line_obj = line.Line(bus_line, edge_id_list, stop_id_list, signal_id_list, edge_length_dic, edge_average_speed_dic, distance_between_stop_dic, distance_between_signal_dic, edge_between_stop_dic, edge_between_signal_dic)
                    line_obj_dic[bus_line] = line_obj
                    if_jump_loop = "yes"
                    break
            if if_jump_loop == "yes":
                break
    return line_obj_dic


line_obj_dic = initialize_line_obj("../../b_network/1_changsha_bus_network_with_signal_d.net.xml",
                                   "../../b_network/3_bus_station.add.xml",
                                   "../../d_bus_rou/2_bus_timetable.rou.xml",
                                   "../../f_pre_sim/output_file/detectors_signal_E1_out.xml",
                                   "../../f_pre_sim/output_file/detectors_edge_E1_out.xml")
# 创建XML根元素
root = ET.Element("line_obj_xml")
# 添加换行和空格
root.text = "\n\t"
for line_id in line_obj_dic.keys():
    line_obj = ET.SubElement(root, "line_obj")
    line_obj.set("line_id_s", line_obj_dic[line_id].line_id_s)
    line_obj.set("edge_id_l", ' '.join(line_obj_dic[line_id].edge_id_l))
    line_obj.set("stop_id_l", ' '.join(line_obj_dic[line_id].stop_id_l))
    line_obj.set("signal_id_l", ' '.join(line_obj_dic[line_id].signal_id_l))
    line_obj.text = "\n\t\t"
    edge_length = ET.SubElement(line_obj, "edge_length")
    for edge_id in line_obj_dic[line_id].edge_length_d:
        edge_length.set(edge_id, str(line_obj_dic[line_id].edge_length_d[edge_id]))
    edge_length.tail = "\n\t\t"
    edge_average_speed = ET.SubElement(line_obj, "edge_average_speed")
    for edge_id in line_obj_dic[line_id].edge_average_speed_d:
        edge_average_speed.set(edge_id, ' '.join(str(n) for n in line_obj_dic[line_id].edge_average_speed_d[edge_id]))
    edge_average_speed.tail = "\n\t\t"
    distance_between_stop = ET.SubElement(line_obj, "distance_between_stop")
    for stop_id in line_obj_dic[line_id].distance_between_stop_d:
        distance_between_stop.set("s_" + stop_id, str(line_obj_dic[line_id].distance_between_stop_d[stop_id]))
    distance_between_stop.tail = "\n\t\t"
    distance_between_signal = ET.SubElement(line_obj, "distance_between_signal")
    for signal_id in line_obj_dic[line_id].distance_between_signal_d:
        distance_between_signal.set(signal_id, str(line_obj_dic[line_id].distance_between_signal_d[signal_id]))
    distance_between_signal.tail = "\n\t\t"
    edge_between_stop = ET.SubElement(line_obj, "edge_between_stop")
    for stop_id in line_obj_dic[line_id].edge_between_stop_d:
        edge_between_stop.set("s_" + stop_id, ' '.join(line_obj_dic[line_id].edge_between_stop_d[stop_id]))
    edge_between_stop.tail = "\n\t\t"
    edge_between_signal = ET.SubElement(line_obj, "edge_between_signal")
    for signal_id in line_obj_dic[line_id].edge_between_signal_d:
        edge_between_signal.set(signal_id, ' '.join(line_obj_dic[line_id].edge_between_signal_d[signal_id]))
    edge_between_signal.tail = "\n\t"
    line_obj.tail = "\n\t"
# 创建XML文件并写入数据
xml_tree = ET.ElementTree(root)
xml_file = "save_obj_line.add.xml"
xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
print(f"线路对象 XML 文件已生成。")
