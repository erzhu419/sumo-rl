from g_no_control.sim_obj import stop
import xml.etree.ElementTree as ET


def initialize_stop_obj(stop_path_ex):
    stop_file = ET.parse(stop_path_ex)
    stop_root = stop_file.getroot()
    """构建所有公交站对象"""
    stop_obj_dic = {}
    for child in stop_root:
        if child.tag == "busStop":
            stop_obj = stop.Stop(child.get("id"), child.get("lane")[0:-2], child.get("lane"), child.get("lines").split(" "))
            stop_obj_dic[child.get("id")] = stop_obj
    return stop_obj_dic


stop_obj_dic = initialize_stop_obj("../../b_network/3_bus_station.add.xml")
# 创建XML根元素
root = ET.Element("stop_obj_xml")
# 添加换行和空格
root.text = "\n\t"
for stop_id in stop_obj_dic.keys():
    stop_obj = ET.SubElement(root, "stop_obj")
    stop_obj.set("stop_id_s", stop_obj_dic[stop_id].stop_id_s)
    stop_obj.set("at_edge_s", stop_obj_dic[stop_id].at_edge_s)
    stop_obj.set("at_lane_s", stop_obj_dic[stop_id].at_lane_s)
    stop_obj.set("service_line_l", ' '.join(stop_obj_dic[stop_id].service_line_l))
    stop_obj.tail = "\n\t"
# 创建XML文件并写入数据
xml_tree = ET.ElementTree(root)
xml_file = "save_obj_stop.add.xml"
xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
print(f"车站对象 XML 文件已生成。")
