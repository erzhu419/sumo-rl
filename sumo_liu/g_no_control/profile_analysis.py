#!/usr/bin/env python3
"""
性能分析 - 使用line_profiler分析关键函数
"""

import time
import sys
import os

# 手动计时器装饰器
class SimpleTimer:
    def __init__(self):
        self.timers = {}
        self.call_counts = {}
    
    def time_function(self, func_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                if func_name not in self.timers:
                    self.timers[func_name] = 0
                    self.call_counts[func_name] = 0
                
                self.timers[func_name] += (end - start)
                self.call_counts[func_name] += 1
                return result
            return wrapper
        return decorator
    
    def print_stats(self):
        print("\n" + "="*80)
        print("函数耗时统计")
        print("="*80)
        print(f"{'函数名':<30} {'调用次数':<10} {'总时间(s)':<12} {'平均时间(ms)':<15}")
        print("-" * 80)
        
        # 按总时间排序
        sorted_timers = sorted(self.timers.items(), key=lambda x: x[1], reverse=True)
        
        for func_name, total_time in sorted_timers:
            count = self.call_counts[func_name]
            avg_time = (total_time / count * 1000) if count > 0 else 0
            print(f"{func_name:<30} {count:<10} {total_time:<12.4f} {avg_time:<15.2f}")
        
        total_measured_time = sum(self.timers.values())
        total_calls = sum(self.call_counts.values())
        print("-" * 80)
        print(f"{'总计':<30} {total_calls:<10} {total_measured_time:<12.4f}")

# 创建全局计时器
timer = SimpleTimer()

def patch_functions():
    """给关键函数打补丁以进行计时"""
    print("正在给关键函数添加计时器...")
    
    # 导入并修补bus.py中的关键方法
    try:
        sys.path.append('/home/erzhu419/mine_code/sumo-rl/sumo_liu/g_no_control/sim_obj')
        
        import bus
        import passenger
        
        # 修补Bus类的方法
        if hasattr(bus.Bus, 'bus_running'):
            original_bus_running = bus.Bus.bus_running
            bus.Bus.bus_running = timer.time_function('Bus.bus_running')(original_bus_running)
            print("✓ Bus.bus_running已添加计时")
        
        if hasattr(bus.Bus, 'bus_activate'):
            original_bus_activate = bus.Bus.bus_activate
            bus.Bus.bus_activate = timer.time_function('Bus.bus_activate')(original_bus_activate)
            print("✓ Bus.bus_activate已添加计时")
        
        # 修补Passenger类的方法
        if hasattr(passenger.Passenger, 'passenger_run'):
            original_passenger_run = passenger.Passenger.passenger_run
            passenger.Passenger.passenger_run = timer.time_function('Passenger.passenger_run')(original_passenger_run)
            print("✓ Passenger.passenger_run已添加计时")
        
        if hasattr(passenger.Passenger, 'passenger_activate'):
            original_passenger_activate = passenger.Passenger.passenger_activate
            passenger.Passenger.passenger_activate = timer.time_function('Passenger.passenger_activate')(original_passenger_activate)
            print("✓ Passenger.passenger_activate已添加计时")
        
        print("关键函数计时器添加完成\n")
        
    except Exception as e:
        print(f"添加计时器时出错: {e}")
        import traceback
        traceback.print_exc()

def patch_sumo_calls():
    """给SUMO API调用添加计时"""
    try:
        import sumo_adapter
        
        # 记录关键SUMO API的调用
        api_methods = [
            ('vehicle.getIDList', sumo_adapter.vehicle, 'getIDList'),
            ('vehicle.getLaneID', sumo_adapter.vehicle, 'getLaneID'), 
            ('vehicle.getSpeed', sumo_adapter.vehicle, 'getSpeed'),
            ('vehicle.getNextLinks', sumo_adapter.vehicle, 'getNextLinks'),
            ('vehicle.isAtBusStop', sumo_adapter.vehicle, 'isAtBusStop'),
            ('person.getIDList', sumo_adapter.person, 'getIDList'),
            ('person.getLaneID', sumo_adapter.person, 'getLaneID'),
            ('person.getVehicle', sumo_adapter.person, 'getVehicle'),
            ('lane.getLength', sumo_adapter.lane, 'getLength'),
            ('simulation.getTime', sumo_adapter.simulation, 'getTime'),
            ('simulationStep', sumo_adapter, 'simulationStep'),
        ]
        
        for method_name, module, attr_name in api_methods:
            if hasattr(module, attr_name):
                original_method = getattr(module, attr_name)
                timed_method = timer.time_function(f'SUMO.{method_name}')(original_method)
                setattr(module, attr_name, timed_method)
                print(f"✓ {method_name}已添加计时")
        
        print("SUMO API计时器添加完成\n")
        
    except Exception as e:
        print(f"添加SUMO计时器时出错: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("开始性能分析...")
    
    # 添加计时器
    patch_functions()
    patch_sumo_calls()
    
    # 记录开始时间
    start_time = time.perf_counter()
    
    try:
        # 运行主程序
        print("开始执行仿真...")
        exec(open('no_control_sim.py').read())
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time = time.perf_counter()
    
    print(f"\n仿真完成！总耗时: {end_time - start_time:.2f}秒")
    
    # 输出统计信息
    timer.print_stats()
    
    # 计算关键指标
    total_measured_time = sum(timer.timers.values())
    unmeasured_time = (end_time - start_time) - total_measured_time
    
    # 生成详细性能报告
    print(f"\n" + "="*100)
    print("📊 详细性能分析与瓶颈识别")
    print("="*100)
    
    # 按时间消耗排序分析
    sorted_timers = sorted(timer.timers.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'排名':<4} {'函数名':<35} {'调用次数':<10} {'总时间(s)':<12} {'平均时间(ms)':<12} {'占比':<8} {'瓶颈级别':<10}")
    print("-" * 100)
    
    for i, (func_name, total_time) in enumerate(sorted_timers, 1):
        count = timer.call_counts[func_name]
        avg_time = (total_time / count * 1000) if count > 0 else 0
        percentage = (total_time / total_measured_time * 100) if total_measured_time > 0 else 0
        
        # 瓶颈级别判断
        if percentage > 30:
            level = "🔥严重"
        elif percentage > 15:
            level = "⚠️主要"
        elif percentage > 5:
            level = "📊潜在"
        else:
            level = "✅正常"
        
        print(f"{i:<4} {func_name:<35} {count:<10} {total_time:<12.4f} {avg_time:<12.2f} {percentage:<8.1f}% {level:<10}")
    
    # 性能问题诊断
    print(f"\n" + "="*100)
    print("🔍 性能问题诊断")
    print("="*100)
    
    # 分析高频调用函数
    high_freq_funcs = [(name, count) for name, count in timer.call_counts.items() if count > 1000]
    if high_freq_funcs:
        print(f"🚨 高频调用函数 (调用次数 > 1000):")
        high_freq_funcs.sort(key=lambda x: x[1], reverse=True)
        for name, count in high_freq_funcs[:10]:
            total_time = timer.timers.get(name, 0)
            avg_time = (total_time / count * 1000) if count > 0 else 0
            print(f"   {name}: {count:,} 次调用, 平均 {avg_time:.3f} ms/次")
    
    # 分析耗时函数
    time_consuming = [(name, time) for name, time in timer.timers.items() if time > 0.1]
    if time_consuming:
        print(f"\n🐌 单次耗时较长的函数 (总耗时 > 0.1秒):")
        time_consuming.sort(key=lambda x: x[1], reverse=True)
        for name, total_time in time_consuming[:10]:
            count = timer.call_counts.get(name, 0)
            avg_time = (total_time / count * 1000) if count > 0 else 0
            print(f"   {name}: 总耗时 {total_time:.4f}s, 平均 {avg_time:.2f} ms/次")
    
    # 优化建议
    print(f"\n" + "="*100)
    print("💡 优化建议")
    print("="*100)
    
    if sorted_timers:
        top_func = sorted_timers[0]
        top_percentage = (top_func[1] / total_measured_time * 100) if total_measured_time > 0 else 0
        
        print(f"1. 最高优先级: 优化 '{top_func[0]}'")
        print(f"   - 占用 {top_percentage:.1f}% 的测量时间")
        print(f"   - 调用 {timer.call_counts.get(top_func[0], 0):,} 次")
        
        # 针对性建议
        func_name = top_func[0].lower()
        if 'bus_running' in func_name:
            print(f"   💡 建议: 优化公交车运行逻辑，减少SUMO API调用频率")
        elif 'passenger_run' in func_name:
            print(f"   💡 建议: 优化乘客状态更新，考虑批量处理")
        elif 'sumo' in func_name:
            print(f"   💡 建议: 考虑缓存SUMO API结果，减少重复查询")
        
        # 如果有多个高耗时函数
        if len(sorted_timers) > 1:
            print(f"\n2. 次要优化目标:")
            for i, (func_name, total_time) in enumerate(sorted_timers[1:4], 1):
                percentage = (total_time / total_measured_time * 100) if total_measured_time > 0 else 0
                print(f"   {i}. {func_name} ({percentage:.1f}%)")
    
    # 系统性能评估
    print(f"\n" + "="*100)
    print("📈 系统性能评估")
    print("="*100)
    
    effective_cpu_usage = (total_measured_time / (end_time - start_time) * 100) if (end_time - start_time) > 0 else 0
    
    print(f"实际总运行时间: {end_time - start_time:.2f}秒")
    print(f"被测量函数时间: {total_measured_time:.2f}秒 ({total_measured_time/(end_time - start_time)*100:.1f}%)")
    print(f"其他时间(初始化等): {unmeasured_time:.2f}秒 ({unmeasured_time/(end_time - start_time)*100:.1f}%)")
    print(f"有效CPU利用率: {effective_cpu_usage:.1f}%")
    
    # 性能等级评估
    if effective_cpu_usage > 80:
        performance_grade = "A (优秀)"
        comment = "CPU利用率很高，性能良好"
    elif effective_cpu_usage > 60:
        performance_grade = "B (良好)"
        comment = "CPU利用率适中，有一定优化空间"
    elif effective_cpu_usage > 40:
        performance_grade = "C (一般)"
        comment = "CPU利用率偏低，存在性能瓶颈"
    else:
        performance_grade = "D (需优化)"
        comment = "CPU利用率很低，可能存在严重瓶颈或等待"
    
    print(f"性能等级: {performance_grade}")
    print(f"评价: {comment}")
    
    # 保存详细报告到文件
    with open('detailed_performance_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("详细性能分析报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总运行时间: {end_time - start_time:.4f}秒\n")
        f.write(f"被测量函数时间: {total_measured_time:.4f}秒\n\n")
        
        f.write("函数耗时排行榜:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'函数名':<35} {'调用次数':<10} {'总时间(s)':<12} {'平均时间(ms)':<12} {'占比':<8}\n")
        f.write("-" * 80 + "\n")
        
        for func_name, total_time in sorted_timers:
            count = timer.call_counts[func_name]
            avg_time = (total_time / count * 1000) if count > 0 else 0
            percentage = (total_time / total_measured_time * 100) if total_measured_time > 0 else 0
            f.write(f"{func_name:<35} {count:<10} {total_time:<12.4f} {avg_time:<12.2f} {percentage:<8.1f}%\n")
        
        f.write(f"\n性能评估:\n")
        f.write(f"有效CPU利用率: {effective_cpu_usage:.1f}%\n")
        f.write(f"性能等级: {performance_grade}\n")
        f.write(f"评价: {comment}\n")
    
    print(f"\n📄 详细报告已保存到: detailed_performance_analysis.txt")

if __name__ == "__main__":
    main()