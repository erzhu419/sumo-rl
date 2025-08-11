#!/usr/bin/env python3
"""
性能分析脚本 - 使用cProfile分析no_control_sim.py的性能瓶颈
"""

import cProfile
import pstats
import io
import os
import time
import importlib.util

def run_performance_test():
    """运行性能测试"""
    print("开始性能分析...")
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    
    # 开始分析
    profiler.enable()
    
    try:
        # 导入并运行主程序
        # 需要重定向以避免重复导入问题
        import importlib.util
        spec = importlib.util.spec_from_file_location("no_control_sim", "no_control_sim.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
    except Exception as e:
        print(f"运行时出错: {e}")
        profiler.disable()
        return
    
    # 停止分析
    profiler.disable()
    
    # 创建统计对象
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    print("\n" + "="*80)
    print("性能分析报告")
    print("="*80)
    
    # 按累积时间排序，显示前20个函数
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    print("\n" + "-"*80)
    print("按自身时间排序的前20个函数:")
    print("-"*80)
    ps.sort_stats('tottime')
    ps.print_stats(20)
    
    # 保存详细报告到文件
    with open('performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("性能分析详细报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("按累积时间排序:\n")
        f.write("-"*40 + "\n")
        ps_file = pstats.Stats(profiler, stream=f)
        ps_file.sort_stats('cumulative')
        ps_file.print_stats(50)
        
        f.write("\n\n按自身时间排序:\n")
        f.write("-"*40 + "\n")
        ps_file = pstats.Stats(profiler, stream=f)
        ps_file.sort_stats('tottime')
        ps_file.print_stats(50)
        
        f.write("\n\nSUMO API调用统计:\n")
        f.write("-"*40 + "\n")
        ps_file = pstats.Stats(profiler, stream=f)
        ps_file.print_stats('sumo_adapter')
        ps_file = pstats.Stats(profiler, stream=f)
        ps_file.print_stats('sumo\\.')
    
    print(f"\n详细报告已保存到: performance_report.txt")
    
    # 生成函数消耗排行榜
    stats_data = []
    for func, stat in ps.stats.items():
        # stat 是一个包含 (cc, nc, tt, ct, callers) 的元组
        # cc: 原始调用次数, nc: 递归调用次数, tt: 总时间, ct: 累积时间
        cc, nc, tt, ct = stat[:4]
        stats_data.append({
            'function': f"{func[0]}:{func[1]}({func[2]})",
            'calls': cc,
            'tottime': tt,
            'cumtime': ct,
            'percall_tot': tt / cc if cc > 0 else 0,
            'percall_cum': ct / cc if cc > 0 else 0
        })
    
    # 按总时间排序找出最耗时的函数
    stats_data.sort(key=lambda x: x['tottime'], reverse=True)
    
    print(f"\n" + "="*100)
    print("🔥 最耗时的函数排行榜 (按自身时间)")
    print("="*100)
    print(f"{'排名':<4} {'函数名':<60} {'调用次数':<10} {'总时间(s)':<10} {'平均时间(ms)':<12} {'占比':<8}")
    print("-" * 100)
    
    total_time_measured = sum(x['tottime'] for x in stats_data)
    for i, func_data in enumerate(stats_data[:20], 1):
        func_name = func_data['function']
        if len(func_name) > 57:
            func_name = func_name[:54] + "..."
        avg_time_ms = func_data['percall_tot'] * 1000
        percentage = (func_data['tottime'] / total_time_measured * 100) if total_time_measured > 0 else 0
        
        print(f"{i:<4} {func_name:<60} {func_data['calls']:<10} {func_data['tottime']:<10.4f} {avg_time_ms:<12.2f} {percentage:<8.1f}%")
    
    # 按累积时间排序的排行榜
    stats_data.sort(key=lambda x: x['cumtime'], reverse=True)
    
    print(f"\n" + "="*100)
    print("📊 最耗时的函数排行榜 (按累积时间)")
    print("="*100)
    print(f"{'排名':<4} {'函数名':<60} {'调用次数':<10} {'累积时间(s)':<12} {'平均时间(ms)':<12} {'占比':<8}")
    print("-" * 100)
    
    total_cumtime = sum(x['cumtime'] for x in stats_data)
    for i, func_data in enumerate(stats_data[:15], 1):
        func_name = func_data['function']
        if len(func_name) > 57:
            func_name = func_name[:54] + "..."
        avg_time_ms = func_data['percall_cum'] * 1000
        percentage = (func_data['cumtime'] / total_cumtime * 100) if total_cumtime > 0 else 0
        
        print(f"{i:<4} {func_name:<60} {func_data['calls']:<10} {func_data['cumtime']:<12.4f} {avg_time_ms:<12.2f} {percentage:<8.1f}%")
    
    # 输出关键统计信息
    stats = ps.stats
    total_calls = sum(stat[0] for stat in stats.values())  # stat[0] 是调用次数
    total_time = sum(stat[2] for stat in stats.values())   # stat[2] 是总时间
    print(f"\n" + "="*80)
    print("📈 关键统计信息")
    print("="*80)
    print(f"总函数调用次数: {total_calls:,}")
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"被分析的函数数量: {len(stats):,}")
    print(f"平均每秒函数调用: {total_calls/(total_time or 1):,.0f}")
    
    # 专门分析业务逻辑函数
    print(f"\n" + "="*80)
    print("🚌 业务逻辑函数分析")
    print("="*80)
    
    business_functions = []
    for func_data in stats_data:
        if any(keyword in func_data['function'].lower() for keyword in 
               ['bus_running', 'passenger_run', 'bus_activate', 'passenger_activate', 
                'create_obj', 'save_data']):
            business_functions.append(func_data)
    
    if business_functions:
        print(f"{'函数名':<50} {'调用次数':<10} {'总时间(s)':<10} {'平均时间(ms)':<15}")
        print("-" * 90)
        for func_data in business_functions:
            func_name = func_data['function']
            if len(func_name) > 47:
                func_name = func_name[:44] + "..."
            avg_time_ms = func_data['percall_tot'] * 1000
            print(f"{func_name:<50} {func_data['calls']:<10} {func_data['tottime']:<10.4f} {avg_time_ms:<15.2f}")
    else:
        print("未检测到主要业务逻辑函数调用")

def analyze_sumo_calls():
    """专门分析SUMO API调用"""
    print("\n" + "="*80)
    print("SUMO API调用分析")
    print("="*80)
    
    # 简单的调用计数器
    import sumo_adapter
    
    # 记录原始方法
    original_methods = {}
    call_counts = {}
    call_times = {}
    
    def create_wrapper(method_name, original_method):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = original_method(*args, **kwargs)
            end_time = time.perf_counter()
            
            call_counts[method_name] = call_counts.get(method_name, 0) + 1
            call_times[method_name] = call_times.get(method_name, 0) + (end_time - start_time)
            return result
        return wrapper
    
    # 包装主要的API调用
    api_modules = ['vehicle', 'person', 'simulation', 'lane', 'trafficlight', 'busstop']
    
    for module_name in api_modules:
        if hasattr(sumo_adapter, module_name):
            module = getattr(sumo_adapter, module_name)
            for attr_name in dir(module):
                if not attr_name.startswith('_') and callable(getattr(module, attr_name)):
                    full_name = f"{module_name}.{attr_name}"
                    original_method = getattr(module, attr_name)
                    original_methods[full_name] = original_method
                    setattr(module, attr_name, create_wrapper(full_name, original_method))
    
    print("API包装完成，开始运行...")
    
    # 运行一个简化版本的仿真来分析API调用
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("no_control_sim", "no_control_sim.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"运行出错: {e}")
        return
    
    # 恢复原始方法
    for full_name, original_method in original_methods.items():
        module_name, attr_name = full_name.split('.', 1)
        module = getattr(sumo_adapter, module_name)
        setattr(module, attr_name, original_method)
    
    # 输出API调用统计
    print("\nSUMO API调用统计:")
    print("-" * 60)
    print(f"{'API方法':<30} {'调用次数':<10} {'总耗时(s)':<10} {'平均耗时(ms)':<15}")
    print("-" * 60)
    
    sorted_calls = sorted(call_counts.items(), key=lambda x: call_times.get(x[0], 0), reverse=True)
    
    for method_name, count in sorted_calls[:20]:
        total_time = call_times.get(method_name, 0)
        avg_time = (total_time / count * 1000) if count > 0 else 0
        print(f"{method_name:<30} {count:<10} {total_time:<10.4f} {avg_time:<15.2f}")
    
    print(f"\n总API调用次数: {sum(call_counts.values()):,}")
    print(f"总API调用时间: {sum(call_times.values()):.4f}秒")

if __name__ == "__main__":
    # 切换到正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("性能分析工具")
    print("1. 运行完整的性能分析")
    print("2. 运行SUMO API调用分析")
    
    try:
        choice = input("请选择 (1 or 2, 默认1): ").strip() or "1"
    except EOFError:
        choice = "1"  # 默认选择1
    
    if choice == "2":
        analyze_sumo_calls()
    else:
        run_performance_test()