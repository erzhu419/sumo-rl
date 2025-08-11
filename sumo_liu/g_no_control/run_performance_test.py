#!/usr/bin/env python3
"""
性能分析脚本 - 分析no_control_sim.py的性能瓶颈
"""

import cProfile
import pstats
import io
import sys
import os
import time

def main():
    # 切换到正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("开始性能分析...")
    print("当前目录:", os.getcwd())
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    
    # 开始分析
    profiler.enable()
    start_time = time.perf_counter()
    
    try:
        # 直接运行no_control_sim.py的内容，但减少循环次数以便快速分析
        exec(open('no_control_sim.py').read())
        
    except Exception as e:
        print(f"运行时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止分析
        profiler.disable()
        end_time = time.perf_counter()
    
    print(f"\n仿真完成，总耗时: {end_time - start_time:.2f}秒")
    
    # 创建统计对象
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    
    print("\n" + "="*80)
    print("性能分析报告 - 按累积时间排序的前20个函数")
    print("="*80)
    
    # 按累积时间排序，显示前20个函数
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    
    print("\n" + "-"*80)
    print("按自身时间排序的前20个函数:")
    print("-"*80)
    ps.sort_stats('tottime')
    ps.print_stats(20)
    
    # 专门分析SUMO相关调用
    print("\n" + "-"*80)
    print("SUMO API相关调用:")
    print("-"*80)
    ps.print_stats('sumo')
    
    # 专门分析bus和passenger相关调用
    print("\n" + "-"*80)
    print("Bus和Passenger相关调用:")
    print("-"*80)
    ps.print_stats('bus_running|passenger_run')
    
    # 保存详细报告到文件
    with open('performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("性能分析详细报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"总运行时间: {end_time - start_time:.2f}秒\n\n")
        
        f.write("按累积时间排序:\n")
        f.write("-"*40 + "\n")
        ps_file = pstats.Stats(profiler)
        ps_file.sort_stats('cumulative')
        ps_file.print_stats(100, file=f)
        
        f.write("\n\n按自身时间排序:\n")
        f.write("-"*40 + "\n")
        ps_file.sort_stats('tottime')
        ps_file.print_stats(100, file=f)
        
        f.write("\n\nSUMO API调用统计:\n")
        f.write("-"*40 + "\n")
        ps_file.print_stats('sumo', file=f)
        
        f.write("\n\nBus和Passenger方法调用:\n")
        f.write("-"*40 + "\n")
        ps_file.print_stats('bus_running|passenger_run', file=f)
    
    print(f"\n详细报告已保存到: performance_report.txt")
    
    # 提取统计数据并找出瓶颈
    stats_dict = ps.stats
    stats_data = []
    for func, stat in stats_dict.items():
        # stat 是一个元组 (cc, nc, tt, ct, callers)
        cc, nc, tt, ct = stat[:4]
        stats_data.append({
            'function': f"{func[0]}:{func[1]}({func[2]})",
            'calls': cc,
            'tottime': tt,
            'cumtime': ct,
            'percall_tot': tt / cc if cc > 0 else 0,
            'percall_cum': ct / cc if cc > 0 else 0
        })
    
    # 按tottime排序
    stats_data.sort(key=lambda x: x['tottime'], reverse=True)
    
    print(f"\n" + "="*120)
    print("🔥 函数性能排行榜 - 找出最大的性能瓶颈")
    print("="*120)
    print(f"{'排名':<4} {'函数名':<70} {'调用次数':<10} {'总时间(s)':<10} {'平均(ms)':<10} {'占比':<8} {'类型':<10}")
    print("-" * 120)
    
    total_time = sum(x['tottime'] for x in stats_data)
    for i, stat in enumerate(stats_data[:30], 1):
        func_name = stat['function']
        if len(func_name) > 67:
            func_name = func_name[:64] + "..."
        
        avg_time_ms = stat['percall_tot'] * 1000
        percentage = (stat['tottime'] / total_time * 100) if total_time > 0 else 0
        
        # 判断函数类型
        func_type = "其他"
        if 'sumo' in func_name.lower():
            func_type = "SUMO API"
        elif any(keyword in func_name.lower() for keyword in ['bus_running', 'passenger_run', 'bus_activate']):
            func_type = "业务逻辑"
        elif any(keyword in func_name.lower() for keyword in ['create_obj', 'save_data']):
            func_type = "数据处理"
        elif any(keyword in func_name.lower() for keyword in ['socket', 'recv', 'send', 'connect']):
            func_type = "网络IO"
        
        # 使用不同标记突出重要函数
        if percentage > 10.0:
            marker = "🔥🔥"
        elif percentage > 5.0:
            marker = "🔥"
        elif percentage > 2.0:
            marker = "⚠️"
        elif percentage > 1.0:
            marker = "📊"
        else:
            marker = "  "
        
        print(f"{marker}{i:<3} {func_name:<70} {stat['calls']:<10} {stat['tottime']:<10.4f} {avg_time_ms:<10.2f} {percentage:<8.1f}% {func_type:<10}")
    
    # 分类统计分析
    print(f"\n" + "="*100)
    print("📊 分类性能统计")
    print("="*100)
    
    categories = {
        'SUMO API': [],
        '业务逻辑': [],
        '数据处理': [],
        '网络IO': [],
        'Python内建': [],
        '其他': []
    }
    
    for stat in stats_data:
        func_name = stat['function'].lower()
        if 'sumo' in func_name:
            categories['SUMO API'].append(stat)
        elif any(keyword in func_name for keyword in ['bus_running', 'passenger_run', 'bus_activate', 'passenger_activate']):
            categories['业务逻辑'].append(stat)
        elif any(keyword in func_name for keyword in ['create_obj', 'save_data', 'pickle', 'json']):
            categories['数据处理'].append(stat)
        elif any(keyword in func_name for keyword in ['socket', 'recv', 'send', 'connect', 'traci']):
            categories['网络IO'].append(stat)
        elif any(keyword in func_name for keyword in ['<built-in', '<method', 'isinstance', 'getattr', 'hasattr']):
            categories['Python内建'].append(stat)
        else:
            categories['其他'].append(stat)
    
    print(f"{'类别':<15} {'函数数量':<10} {'总时间(s)':<12} {'占比':<8} {'平均单函数耗时(ms)':<20}")
    print("-" * 80)
    
    for category, funcs in categories.items():
        if funcs:
            cat_total_time = sum(f['tottime'] for f in funcs)
            cat_percentage = (cat_total_time / total_time * 100) if total_time > 0 else 0
            avg_func_time = (cat_total_time / len(funcs) * 1000) if funcs else 0
            
            print(f"{category:<15} {len(funcs):<10} {cat_total_time:<12.4f} {cat_percentage:<8.1f}% {avg_func_time:<20.2f}")
    
    # 输出关键统计信息
    total_calls = sum(stat['calls'] for stat in stats_data)
    
    print(f"\n" + "="*80)
    print("📈 总体性能统计")
    print("="*80)
    print(f"总函数调用次数: {total_calls:,}")
    print(f"实际运行时间: {end_time - start_time:.2f}秒")
    print(f"被分析的函数数量: {len(stats_data):,}")
    print(f"平均每秒函数调用: {total_calls/(end_time - start_time):,.0f}")
    print(f"前5个函数占用时间: {sum(s['tottime'] for s in stats_data[:5])/total_time*100:.1f}%")
    print(f"前20个函数占用时间: {sum(s['tottime'] for s in stats_data[:20])/total_time*100:.1f}%")
    
    # 性能瓶颈识别和建议
    print(f"\n" + "="*80)
    print("🎯 性能瓶颈识别与优化建议")
    print("="*80)
    
    if stats_data:
        top_func = stats_data[0]
        top_percentage = (top_func['tottime'] / total_time * 100) if total_time > 0 else 0
        
        print(f"最大瓶颈: {top_func['function']}")
        print(f"          占用 {top_percentage:.1f}% 的总运行时间")
        print(f"          被调用 {top_func['calls']:,} 次")
        print(f"          平均每次调用耗时 {top_func['percall_tot']*1000:.2f} 毫秒")
        
        if top_percentage > 20:
            print("🔥 严重瓶颈！建议立即优化此函数")
        elif top_percentage > 10:
            print("⚠️  主要瓶颈，建议优先优化")
        elif top_percentage > 5:
            print("📊 潜在瓶颈，可考虑优化")
        
        # 针对性优化建议
        func_name = top_func['function'].lower()
        if 'sumo' in func_name:
            print("💡 优化建议: 考虑批量SUMO API调用，减少通信开销")
        elif 'bus_running' in func_name or 'passenger_run' in func_name:
            print("💡 优化建议: 优化业务逻辑算法，减少重复计算")
        elif top_func['calls'] > 10000:
            print("💡 优化建议: 函数调用次数很高，考虑缓存结果或批量处理")
        
        print(f"\n推荐优化顺序:")
        for i, stat in enumerate(stats_data[:5], 1):
            percentage = (stat['tottime'] / total_time * 100) if total_time > 0 else 0
            print(f"   {i}. {stat['function'][:60]} ({percentage:.1f}%)")
    
    print(f"\n📋 详细报告说明:")
    print(f"   - performance_report.txt: 完整的函数调用统计")
    print(f"   - 🔥🔥: 占用超过10%时间的严重瓶颈函数")
    print(f"   - 🔥: 占用5-10%时间的主要瓶颈函数")
    print(f"   - ⚠️: 占用2-5%时间的潜在瓶颈函数")

if __name__ == "__main__":
    main()