"""
SUMO适配器层 - 支持TraCI和LibSUMO的自动切换
提供统一的API接口，优先使用LibSUMO以获得更好的性能
"""

import sys
import os

USE_LIBSUMO = False
sumo_api = None

try:
    import libsumo
    sumo_api = libsumo
    USE_LIBSUMO = True
    print("使用LibSUMO - 高性能模式")
except ImportError:
    try:
        import traci
        sumo_api = traci  
        USE_LIBSUMO = False
        print("使用TraCI - 兼容模式")
    except ImportError:
        raise ImportError("无法导入LibSUMO或TraCI，请检查SUMO安装")

# 导出所有SUMO API模块
simulation = sumo_api.simulation
vehicle = sumo_api.vehicle
person = sumo_api.person
trafficlight = sumo_api.trafficlight
busstop = sumo_api.busstop
inductionloop = sumo_api.inductionloop

# 创建带缓存的lane和edge模块包装
class CachedLaneAPI:
    def __init__(self, original_api):
        self._api = original_api
        
    def getLength(self, lane_id):
        """获取车道长度（带缓存）"""
        try:
            from sumo_cache import cache
            return cache.get_lane_length(lane_id)
        except ImportError:
            # 如果缓存模块不可用，直接调用原API
            return self._api.getLength(lane_id)
    
    def __getattr__(self, name):
        """代理所有其他方法到原API"""
        return getattr(self._api, name)

class CachedEdgeAPI:
    def __init__(self, original_api):
        self._api = original_api
        
    def getLength(self, edge_id):
        """获取路段长度（带缓存）"""
        try:
            from sumo_cache import cache
            return cache.get_edge_length(edge_id)
        except ImportError:
            return self._api.getLength(edge_id)
    
    def __getattr__(self, name):
        return getattr(self._api, name)

# 使用缓存包装器
lane = CachedLaneAPI(sumo_api.lane)
edge = CachedEdgeAPI(sumo_api.edge)

def start(cmd_list):
    """启动SUMO仿真"""
    if USE_LIBSUMO:
        # LibSUMO不需要多线程参数，过滤掉--c和--threads参数
        filtered_cmd = []
        skip_next = False
        for arg in cmd_list:
            if skip_next:
                skip_next = False
                continue
            if arg in ["--c", "--threads"]:
                skip_next = True
                continue
            filtered_cmd.append(arg)
        
        sumo_api.start(filtered_cmd)
        print("LibSUMO启动成功")
        
        # 启动后立即预加载静态数据
        try:
            from sumo_cache import cache
            cache.preload_network_data()
        except Exception as e:
            print(f"预加载缓存数据失败: {e}")
    else:
        sumo_api.start(cmd_list)
        print("TraCI启动成功")

def close():
    """关闭SUMO仿真"""
    sumo_api.close()
    if USE_LIBSUMO:
        print("LibSUMO关闭")
    else:
        print("TraCI关闭")

def simulationStep():
    """执行一步仿真"""
    sumo_api.simulationStep()

def get_api_info():
    """获取当前使用的API信息"""
    return {
        'api_type': 'LibSUMO' if USE_LIBSUMO else 'TraCI',
        'use_libsumo': USE_LIBSUMO,
        'module': sumo_api
    }