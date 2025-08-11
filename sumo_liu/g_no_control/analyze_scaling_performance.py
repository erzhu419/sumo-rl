#!/usr/bin/env python3
"""
分析仿真步数增加时的性能缩放问题
"""

import time
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def analyze_step_scaling():
    print("="*80)
    print("仿真步数缩放性能分析")
    print("="*80)
    
    import sumo_adapter as sumo
    from sumolib import checkBinary
    
    sumoBinary = checkBinary("sumo")
    sumo_cfg_file = "no_control_sim_traci.sumocfg"
    
    # 测试不同步数的性能
    test_steps = [500, 1000, 2000, 4000, 8000]
    results = []
    
    for steps in test_steps:
        print(f"\n测试 {steps} 步仿真...")
        
        try:
            sumo.start([sumoBinary, "-c", sumo_cfg_file, "--threads", "16"])
            
            start_time = time.perf_counter()
            
            # 模拟简化的仿真循环
            vehicle_counts = []
            person_counts = []
            
            for step in range(steps):
                current_time = sumo.simulation.getTime()
                
                # 获取车辆和乘客数量（主要性能消耗点）
                vehicle_list = sumo.vehicle.getIDList()
                person_list = sumo.person.getIDList() 
                
                vehicle_counts.append(len(vehicle_list))
                person_counts.append(len(person_list))
                
                # 执行仿真步
                sumo.simulationStep()
                
                if step % 1000 == 0:
                    print(f"  步数 {step}: 车辆 {len(vehicle_list)}, 乘客 {len(person_list)}")
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # 计算统计信息
            avg_vehicles = sum(vehicle_counts) / len(vehicle_counts)
            max_vehicles = max(vehicle_counts) 
            avg_persons = sum(person_counts) / len(person_counts)
            max_persons = max(person_counts)
            
            results.append({
                'steps': steps,
                'duration': duration,
                'avg_vehicles': avg_vehicles,
                'max_vehicles': max_vehicles,
                'avg_persons': avg_persons,
                'max_persons': max_persons,
                'steps_per_sec': steps / duration,
                'time_per_step': duration / steps * 1000  # ms
            })
            
            print(f"  完成 {steps} 步，耗时 {duration:.2f}秒 ({steps/duration:.1f} 步/秒)")
            print(f"  平均车辆数: {avg_vehicles:.1f}, 最大: {max_vehicles}")
            print(f"  平均乘客数: {avg_persons:.1f}, 最大: {max_persons}")
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append({'steps': steps, 'duration': -1, 'error': str(e)})
        finally:
            sumo.close()
    
    return results

def analyze_performance_factors(results):
    print("\n" + "="*80)
    print("性能因子分析")
    print("="*80)
    
    print(f"{'步数':<8} {'耗时(s)':<10} {'步/秒':<10} {'ms/步':<10} {'平均车辆':<10} {'平均乘客':<10} {'线性度':<10}")
    print("-" * 80)
    
    baseline = None
    for result in results:
        if result.get('duration', -1) > 0:
            if baseline is None:
                baseline = result
                linearity = 1.0
            else:
                expected_time = baseline['duration'] * result['steps'] / baseline['steps']
                linearity = expected_time / result['duration']
            
            print(f"{result['steps']:<8} {result['duration']:<10.2f} {result['steps_per_sec']:<10.1f} "
                  f"{result['time_per_step']:<10.2f} {result['avg_vehicles']:<10.1f} "
                  f"{result['avg_persons']:<10.1f} {linearity:<10.2f}")

def identify_bottlenecks():
    print("\n" + "="*80)
    print("瓶颈识别分析")
    print("="*80)
    
    bottleneck_analysis = """
    可能的非线性性能瓶颈:

    1. 📈 车辆数量增长
       - 随着时间推移，路网中车辆数量增加
       - 更多车辆 = 更多API调用 (getLaneID, getSpeed, etc.)
       - 复杂度: O(车辆数量 × 步数)

    2. 👥 乘客数量增长  
       - 乘客随时间累积在站台
       - 更多乘客 = 更多状态检查
       - 复杂度: O(乘客数量 × 步数)

    3. 🔄 状态复杂度增长
       - 交通拥堵增加计算复杂度
       - 路径计算变得更复杂
       - 信号灯排队长度增加

    4. 💾 内存压力
       - 长时间仿真累积更多数据
       - GC压力增加
       - 缓存失效

    5. 🌐 网络效应
       - 车辆间相互影响增强
       - 系统状态更加复杂
    """
    
    print(bottleneck_analysis)

def suggest_optimizations():
    print("\n" + "="*80)
    print("优化建议")
    print("="*80)
    
    suggestions = """
    针对长时间仿真的优化策略:

    🎯 立即可实施:
    1. 减少不必要的API调用
       - 批量获取车辆/乘客列表
       - 缓存更多静态和半静态数据
       - 只在状态改变时更新

    2. 算法优化
       - 减少bus_running中的重复计算
       - 优化passenger_run的逻辑
       - 使用更高效的数据结构

    3. 内存管理
       - 定期清理不需要的历史数据
       - 限制历史记录的大小
       - 优化对象创建

    🔧 中期优化:
    4. 分块处理
       - 将大规模仿真分成小块
       - 使用检查点保存/恢复
       - 并行处理独立区域

    5. 智能调度
       - 跳过不活跃区域的更新
       - 自适应更新频率
       - 基于ROI的精度控制
    """
    
    print(suggestions)

def run_quick_profiling():
    print("\n" + "="*80)
    print("快速性能剖析 - 找出具体瓶颈")
    print("="*80)
    
    import sumo_adapter as sumo
    from sumolib import checkBinary
    
    # 创建简单的计时器
    timers = {}
    def time_section(name):
        class Timer:
            def __enter__(self):
                self.start = time.perf_counter()
                return self
            def __exit__(self, *args):
                duration = time.perf_counter() - self.start
                if name not in timers:
                    timers[name] = []
                timers[name].append(duration)
        return Timer()
    
    sumoBinary = checkBinary("sumo")
    sumo_cfg_file = "no_control_sim_traci.sumocfg"
    
    try:
        sumo.start([sumoBinary, "-c", sumo_cfg_file, "--threads", "16"])
        
        print("运行1000步详细计时分析...")
        
        for step in range(1000):
            with time_section("get_simulation_time"):
                current_time = sumo.simulation.getTime()
            
            with time_section("get_vehicle_list"):
                vehicle_list = sumo.vehicle.getIDList()
            
            with time_section("get_person_list"):  
                person_list = sumo.person.getIDList()
            
            with time_section("process_vehicles"):
                for vid in vehicle_list[:min(10, len(vehicle_list))]:  # 只处理前10个
                    if sumo.vehicle.getTypeID(vid) == "Bus":
                        pass  # 模拟bus处理
            
            with time_section("process_persons"):
                for pid in person_list[:min(10, len(person_list))]:  # 只处理前10个
                    pass  # 模拟passenger处理
            
            with time_section("simulation_step"):
                sumo.simulationStep()
        
        # 输出计时统计
        print("\n各部分平均耗时 (ms):")
        print("-" * 40)
        for name, times in timers.items():
            avg_time = sum(times) / len(times) * 1000
            total_time = sum(times) * 1000
            print(f"{name:<20} {avg_time:>8.3f}  (总计: {total_time:>6.1f}ms)")
        
    finally:
        sumo.close()

if __name__ == "__main__":
    print("仿真步数缩放性能分析工具")
    print("注意: 这将运行多个短期仿真来分析性能缩放")
    
    # 运行分析
    results = analyze_step_scaling()
    analyze_performance_factors(results) 
    identify_bottlenecks()
    suggest_optimizations()
    run_quick_profiling()
    
    print(f"\n🎯 关键结论:")
    print(f"   如果发现非线性增长，主要原因是车辆/乘客数量随时间增长")
    print(f"   解决方案: 优化循环逻辑，减少不必要的重复计算")