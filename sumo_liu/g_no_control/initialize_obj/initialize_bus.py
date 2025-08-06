from g_no_control.sim_obj import bus
import xml.etree.ElementTree as ET


def initialize_bus_obj(bus_rou_path_ex):
    bus_rou_file = ET.parse(bus_rou_path_ex)
    bus_rou_root = bus_rou_file.getroot()
    """构建所有公交车对象"""
    bus_obj_dic = {}
    for child in bus_rou_root:
        if child.tag == "vehicle":
            bus_obj = bus.Bus(child.get("id"), float(child.get("depart")), child[1].get("line"))
            bus_obj_dic[child.get("id")] = bus_obj
    return bus_obj_dic


bus_obj_dic = initialize_bus_obj("../../d_bus_rou/2_bus_timetable.rou.xml")
# 创建XML根元素
root = ET.Element("bus_obj_xml")
# 添加换行和空格
root.text = "\n\t"
for bus_id in bus_obj_dic.keys():
    bus_obj = ET.SubElement(root, "bus_obj")
    bus_obj.set("bus_id_s", bus_obj_dic[bus_id].bus_id_s)
    bus_obj.set("start_time_n", str(bus_obj_dic[bus_id].start_time_n))
    bus_obj.set("belong_line_id_s", bus_obj_dic[bus_id].belong_line_id_s)
    bus_obj.tail = "\n\t"
# 创建XML文件并写入数据
xml_tree = ET.ElementTree(root)
xml_file = "save_obj_bus.add.xml"
xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
print(f"公交车对象 XML 文件已生成。")
