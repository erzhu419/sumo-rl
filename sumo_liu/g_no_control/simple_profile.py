#!/usr/bin/env python3
"""
简化的性能分析脚本
"""

import cProfile
import pstats
import sys
import os
import time

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("开始性能分析...")
    
    # 使用cProfile运行
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.perf_counter()
    
    try:
        # 执行主程序
        exec(open('no_control_sim.py').read())
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        profiler.disable()
        end_time = time.perf_counter()
    
    print(f"\n实际运行时间: {end_time - start_time:.2f}秒")
    
    # 生成统计信息
    stats = pstats.Stats(profiler)
    
    print("\n" + "="*80)
    print("最耗时的20个函数 (按总时间)")
    print("="*80)
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    print("\n" + "="*80) 
    print("最耗时的20个函数 (按累积时间)")
    print("="*80)
    stats.sort_stats('cumtime')
    stats.print_stats(20)
    
    # 分析SUMO调用
    print("\n" + "="*80)
    print("SUMO相关函数调用")
    print("="*80)
    stats.print_stats('sumo_adapter')
    
    # 分析bus和passenger方法
    print("\n" + "="*80)
    print("Bus运行相关函数")
    print("="*80)
    stats.print_stats('bus_running')
    
    print("\n" + "="*80)
    print("Passenger运行相关函数")
    print("="*80)
    stats.print_stats('passenger_run')
    
    # 保存报告到文件
    with open('profile_stats.txt', 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        f.write(f"总运行时间: {end_time - start_time:.2f}秒\n\n")
        
        f.write("按总时间排序:\n")
        f.write("="*50 + "\n")
        stats.sort_stats('tottime')
        stats.print_stats(50)
        
        f.write("\n\n按累积时间排序:\n")
        f.write("="*50 + "\n")
        stats.sort_stats('cumtime')
        stats.print_stats(50)
        
        f.write("\n\nSUMO相关调用:\n")
        f.write("="*50 + "\n")
        stats.print_stats('sumo')
    
    print(f"\n详细报告保存到: profile_stats.txt")
    
    # 生成更详细的性能分析表格
    stats_dict = stats.stats
    func_stats = []
    for func, stat in stats_dict.items():
        # stat 是一个元组 (cc, nc, tt, ct, callers)
        cc, nc, tt, ct = stat[:4]
        func_stats.append({
            'function': f"{func[0]}:{func[1]}({func[2]})",
            'calls': cc,
            'tottime': tt,
            'cumtime': ct,
            'percall_tot': tt / cc if cc > 0 else 0,
            'percall_cum': ct / cc if cc > 0 else 0
        })
    
    # 按总时间排序
    func_stats.sort(key=lambda x: x['tottime'], reverse=True)
    
    print(f"\n" + "="*110)
    print("🎯 性能瓶颈识别 - 最消耗CPU的函数")
    print("="*110)
    print(f"{'排名':<4} {'函数名':<65} {'调用次数':<10} {'总时间(s)':<10} {'平均(ms)':<10} {'占比':<8}")
    print("-" * 110)
    
    total_time = sum(x['tottime'] for x in func_stats)
    for i, func_data in enumerate(func_stats[:25], 1):
        func_name = func_data['function']
        if len(func_name) > 62:
            func_name = func_name[:59] + "..."
        
        avg_ms = func_data['percall_tot'] * 1000
        percentage = (func_data['tottime'] / total_time * 100) if total_time > 0 else 0
        
        # 突出显示高消耗函数
        if percentage > 5.0:
            print(f"🔥{i:<3} {func_name:<65} {func_data['calls']:<10} {func_data['tottime']:<10.4f} {avg_ms:<10.2f} {percentage:<8.1f}%")
        elif percentage > 1.0:
            print(f"⚠️ {i:<3} {func_name:<65} {func_data['calls']:<10} {func_data['tottime']:<10.4f} {avg_ms:<10.2f} {percentage:<8.1f}%")
        else:
            print(f"  {i:<3} {func_name:<65} {func_data['calls']:<10} {func_data['tottime']:<10.4f} {avg_ms:<10.2f} {percentage:<8.1f}%")
    
    # 分析SUMO API调用性能
    print(f"\n" + "="*90)
    print("🌐 SUMO API调用性能分析")
    print("="*90)
    
    sumo_funcs = [f for f in func_stats if 'sumo' in f['function'].lower()]
    if sumo_funcs:
        sumo_total = sum(f['tottime'] for f in sumo_funcs)
        print(f"SUMO API总耗时: {sumo_total:.4f}秒 ({sumo_total/(end_time - start_time)*100:.1f}%)")
        print(f"{'API名称':<40} {'调用次数':<10} {'总时间(s)':<10} {'平均(ms)':<10} {'占比':<8}")
        print("-" * 85)
        
        for func_data in sumo_funcs[:15]:
            api_name = func_data['function'].split('/')[-1] if '/' in func_data['function'] else func_data['function']
            if len(api_name) > 37:
                api_name = api_name[:34] + "..."
            
            avg_ms = func_data['percall_tot'] * 1000
            percentage = (func_data['tottime'] / sumo_total * 100) if sumo_total > 0 else 0
            
            print(f"{api_name:<40} {func_data['calls']:<10} {func_data['tottime']:<10.4f} {avg_ms:<10.2f} {percentage:<8.1f}%")
    else:
        print("未检测到SUMO API调用")
    
    # 分析业务逻辑函数
    print(f"\n" + "="*90)
    print("🚌 业务逻辑函数性能分析")
    print("="*90)
    
    business_keywords = ['bus_running', 'passenger_run', 'bus_activate', 'passenger_activate', 
                        'create_obj', 'save_data', 'sim_obj']
    business_funcs = []
    for func_data in func_stats:
        if any(keyword in func_data['function'].lower() for keyword in business_keywords):
            business_funcs.append(func_data)
    
    if business_funcs:
        business_total = sum(f['tottime'] for f in business_funcs)
        print(f"业务逻辑总耗时: {business_total:.4f}秒 ({business_total/(end_time - start_time)*100:.1f}%)")
        print(f"{'函数名':<45} {'调用次数':<10} {'总时间(s)':<10} {'平均(ms)':<10} {'占比':<8}")
        print("-" * 90)
        
        for func_data in business_funcs:
            func_name = func_data['function'].split('/')[-1] if '/' in func_data['function'] else func_data['function']
            if len(func_name) > 42:
                func_name = func_name[:39] + "..."
            
            avg_ms = func_data['percall_tot'] * 1000
            percentage = (func_data['tottime'] / business_total * 100) if business_total > 0 else 0
            
            print(f"{func_name:<45} {func_data['calls']:<10} {func_data['tottime']:<10.4f} {avg_ms:<10.2f} {percentage:<8.1f}%")
    else:
        print("未检测到主要业务逻辑函数")
    
    # 总结性分析
    total_calls = sum(stat.callcount for stat in stats_dict.values())
    print(f"\n" + "="*80)
    print("📊 性能分析总结")
    print("="*80)
    print(f"总函数调用: {total_calls:,}")
    print(f"实际运行时间: {end_time - start_time:.2f}秒")
    print(f"分析的函数数量: {len(stats_dict):,}")
    print(f"平均每秒调用: {total_calls/(end_time - start_time):,.0f}")
    print(f"前3个最耗时函数占总时间的: {sum(f['tottime'] for f in func_stats[:3])/total_time*100:.1f}%")
    print(f"前10个最耗时函数占总时间的: {sum(f['tottime'] for f in func_stats[:10])/total_time*100:.1f}%")
    
    print(f"\n💡 优化建议:")
    if func_stats:
        top_func = func_stats[0]
        print(f"   1. 优先关注 '{top_func['function']}' - 占用 {top_func['tottime']/total_time*100:.1f}% 的时间")
        if len(func_stats) > 1:
            second_func = func_stats[1]
            print(f"   2. 其次关注 '{second_func['function']}' - 占用 {second_func['tottime']/total_time*100:.1f}% 的时间")
    
    if sumo_funcs and business_funcs:
        sumo_ratio = sum(f['tottime'] for f in sumo_funcs) / (end_time - start_time)
        business_ratio = sum(f['tottime'] for f in business_funcs) / (end_time - start_time) 
        if sumo_ratio > business_ratio:
            print(f"   3. SUMO API调用较多，考虑批量操作或缓存优化")
        else:
            print(f"   3. 业务逻辑耗时较多，考虑算法优化")

if __name__ == "__main__":
    main()