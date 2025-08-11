"""
SUMO数据缓存模块 - 缓存静态数据以提升性能
"""

import sumo_adapter as sumo

class SUMOCache:
    def __init__(self):
        self.lane_lengths = {}
        self.edge_lengths = {}
        self.static_data_loaded = False
        
    def get_lane_length(self, lane_id):
        """获取车道长度（带缓存）"""
        if lane_id not in self.lane_lengths:
            # 直接调用原始API，避免循环递归
            import sumo_adapter
            self.lane_lengths[lane_id] = sumo_adapter.sumo_api.lane.getLength(lane_id)
        return self.lane_lengths[lane_id]
    
    def get_edge_length(self, edge_id):
        """获取路段长度（带缓存）"""
        if edge_id not in self.edge_lengths:
            import sumo_adapter
            self.edge_lengths[edge_id] = sumo_adapter.sumo_api.edge.getLength(edge_id)
        return self.edge_lengths[edge_id]
    
    def preload_network_data(self):
        """预加载网络静态数据"""
        if self.static_data_loaded:
            return
            
        print("预加载网络静态数据...")
        
        try:
            import sumo_adapter
            
            # 获取所有车道ID并预加载长度
            all_lanes = sumo_adapter.sumo_api.lane.getIDList()
            for lane_id in all_lanes:
                self.lane_lengths[lane_id] = sumo_adapter.sumo_api.lane.getLength(lane_id)
            
            print(f"✓ 预加载了 {len(all_lanes)} 个车道的长度数据")
            
            # 获取所有路段ID并预加载长度  
            all_edges = sumo_adapter.sumo_api.edge.getIDList()
            for edge_id in all_edges:
                try:
                    self.edge_lengths[edge_id] = sumo_adapter.sumo_api.edge.getLength(edge_id)
                except:
                    # 某些路段可能无法获取长度，跳过
                    pass
            
            print(f"✓ 预加载了 {len(self.edge_lengths)} 个路段的长度数据")
            
            self.static_data_loaded = True
            
        except Exception as e:
            print(f"预加载数据时出错: {e}")
    
    def clear_cache(self):
        """清空缓存"""
        self.lane_lengths.clear()
        self.edge_lengths.clear()
        self.static_data_loaded = False
        print("缓存已清空")
    
    def get_cache_stats(self):
        """获取缓存统计"""
        return {
            'lane_cache_size': len(self.lane_lengths),
            'edge_cache_size': len(self.edge_lengths),
            'static_data_loaded': self.static_data_loaded
        }

# 创建全局缓存实例
cache = SUMOCache()