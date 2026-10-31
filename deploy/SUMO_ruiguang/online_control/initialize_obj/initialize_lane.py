from sim_obj import lane
import xml.etree.ElementTree as ET


def initialize_lane_obj(net_path_ex):
    net_file = ET.parse(net_path_ex)
    net_root = net_file.getroot()
    """构建所有车道对象"""
    lane_obj_dic = {}
    for child in net_root:
        if child.tag == "edge" and child.get("function") != "internal":
            for sub_child in child:
                if sub_child.tag == "lane":
                    lane_obj = lane.Lane(sub_child.get("id"), float(sub_child.get("length")), child.get("from"), child.get("to"))
                    lane_obj_dic[sub_child.get("id")] = lane_obj
    return lane_obj_dic


lane_obj_dic = initialize_lane_obj("../../b_network/5g_changsha_bus_network_with_signal_d.net.xml")
# 创建XML根元素
root = ET.Element("lane_obj_xml")
# 添加换行和空格
root.text = "\n\t"
for lane_id in lane_obj_dic.keys():
    lane_obj = ET.SubElement(root, "lane_obj")
    lane_obj.set("lane_id_s", lane_obj_dic[lane_id].lane_id_s)
    lane_obj.set("length_n", str(lane_obj_dic[lane_id].length_n))
    lane_obj.set("from_junction_s", lane_obj_dic[lane_id].from_junction_s)
    lane_obj.set("to_junction_s", lane_obj_dic[lane_id].to_junction_s)
    lane_obj.tail = "\n\t"
# 创建XML文件并写入数据
xml_tree = ET.ElementTree(root)
xml_file = "save_obj_lane.add.xml"
xml_tree.write(xml_file, encoding="utf-8", xml_declaration=True)
print(f"车道对象 XML 文件已生成。")
