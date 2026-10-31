from sim_obj import passenger
import xml.etree.ElementTree as ET


def get_stop_by_edge_id(stop_root_ex, edge_id_ex):
    stop_id = ""
    for child in stop_root_ex:
        if child.tag == "busStop" and child.get("lane")[0:-2] == edge_id_ex:
            stop_id = child.get("id")
            break
    if stop_id == "":
        print("警告，initialize_passenger的get_stop_by_edge_id没找到公交站！")
    return stop_id


def initialize_passenger_obj(stop_path_ex, passenger_rou_path_ex):
    stop_file = ET.parse(stop_path_ex)
    stop_root = stop_file.getroot()
    passenger_rou_file = ET.parse(passenger_rou_path_ex)
    passenger_rou_root = passenger_rou_file.getroot()
    """构建所有乘客对象"""
    passenger_obj_dic = {}
    for child in passenger_rou_root:
        if child.tag == "person":
            edge_id_list = []
            stop_id_list = []
            for sub_child in child:
                if sub_child.tag == "ride":
                    if sub_child.get("from") not in edge_id_list:
                        edge_id_list.append(sub_child.get("from"))
                        stop_id_list.append(get_stop_by_edge_id(stop_root, sub_child.get("from")))
                    if sub_child.get("to") not in edge_id_list:
                        edge_id_list.append(sub_child.get("to"))
                        stop_id_list.append(get_stop_by_edge_id(stop_root, sub_child.get("to")))
            if len(stop_id_list) == 2:
                passenger_obj = passenger.Passenger(child.get("id"), float(child.get("depart")), stop_id_list[0], "", stop_id_list[1], edge_id_list[0], "", edge_id_list[1])
                passenger_obj_dic[child.get("id")] = passenger_obj
            elif len(stop_id_list) == 3:
                passenger_obj = passenger.Passenger(child.get("id"), float(child.get("depart")), stop_id_list[0], stop_id_list[1], stop_id_list[2], edge_id_list[0], edge_id_list[1], edge_id_list[2])
                passenger_obj_dic[child.get("id")] = passenger_obj
            else:
                print("警告，initialize_passenger的initialize_passenger_obj乘客没有乘坐公交车或不止换乘了1次！")
    return passenger_obj_dic


passenger_obj_dic_1 = initialize_passenger_obj("../../b_network/3_bus_station.add.xml",
                                               "../../e_passenger_rou/3_modified_transfer_passenger_0805.rou.xml")
passenger_obj_dic_2 = initialize_passenger_obj("../../b_network/3_bus_station.add.xml",
                                               "../../e_passenger_rou/3_modified_period_workday_passenger_VC_0805.rou.xml")
passenger_obj_dic = {**passenger_obj_dic_1, **passenger_obj_dic_2}
# 创建XML根元素
root = ET.Element("passenger_obj_xml")
# 添加换行和空格
root.text = "\n\t"
for passenger_id in passenger_obj_dic.keys():
    passenger_obj = ET.SubElement(root, "passenger_obj")
    passenger_obj.set("passenger_id_s", passenger_obj_dic[passenger_id].passenger_id_s)
    passenger_obj.set("start_time_n", str(passenger_obj_dic[passenger_id].start_time_n))
    passenger_obj.set("start_stop_id_s", passenger_obj_dic[passenger_id].start_stop_id_s)
    passenger_obj.set("transfer_stop_id_s", passenger_obj_dic[passenger_id].transfer_stop_id_s)
    passenger_obj.set("end_stop_id_s", passenger_obj_dic[passenger_id].end_stop_id_s)
    passenger_obj.set("start_edge_id_s", passenger_obj_dic[passenger_id].start_edge_id_s)
    passenger_obj.set("transfer_edge_id_s", passenger_obj_dic[passenger_id].transfer_edge_id_s)
    passenger_obj.set("end_edge_id_s", passenger_obj_dic[passenger_id].end_edge_id_s)
    passenger_obj.tail = "\n\t"
# 创建XML文件并写入数据
xml_tree = ET.ElementTree(root)
xml_file = "save_obj_passenger.add.xml"
xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
print(f"乘客对象 XML 文件已生成。")
