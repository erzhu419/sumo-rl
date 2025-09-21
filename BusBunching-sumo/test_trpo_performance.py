#!/usr/bin/env python3
import subprocess
import time
import psutil
import signal
import sys
import threading
from pathlib import Path

def run_with_timeout_and_monitor(timeout_seconds=300):  # 5分钟超时
    """
    运行trpo.py并监控性能，带超时限制
    """
    print(f"开始测试trpo.py性能，超时时间: {timeout_seconds}秒")
    
    # 性能监控数据
    performance_data = {
        'cpu_usage': [],
        'memory_usage': [],
        'timestamps': [],
        'process_count': []
    }
    
    # 启动进程
    process = subprocess.Popen(
        [sys.executable, 'trpo.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).parent
    )
    
    start_time = time.time()
    monitoring = True
    
    def monitor_performance():
        """监控性能的线程函数"""
        while monitoring:
            try:
                # 获取进程信息
                proc = psutil.Process(process.pid)
                
                # CPU使用率
                cpu_percent = proc.cpu_percent(interval=0.1)
                
                # 内存使用
                memory_info = proc.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 子进程数量
                children = proc.children(recursive=True)
                process_count = len(children) + 1
                
                # 记录数据
                current_time = time.time() - start_time
                performance_data['timestamps'].append(current_time)
                performance_data['cpu_usage'].append(cpu_percent)
                performance_data['memory_usage'].append(memory_mb)
                performance_data['process_count'].append(process_count)
                
                print(f"[{current_time:.1f}s] CPU: {cpu_percent:.1f}%, 内存: {memory_mb:.1f}MB, 进程数: {process_count}")
                
                time.sleep(2)  # 每2秒监控一次
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_performance)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # 等待进程完成或超时
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        
        monitoring = False  # 停止监控
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n=== 程序正常完成 ===")
            print(f"用时: {elapsed_time:.2f}秒")
            print(f"标准输出:\n{stdout}")
        else:
            print(f"\n=== 程序异常退出 ===")
            print(f"返回码: {process.returncode}")
            print(f"标准错误:\n{stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"\n=== 程序超时 ({timeout_seconds}秒) ===")
        
        # 强制终止进程
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            
            # 等待进程结束
            process.wait(timeout=5)
        except:
            # 如果无法正常终止，强制杀死
            try:
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except:
                pass
        
        monitoring = False
        
        # 获取部分输出
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                print(f"部分标准输出:\n{stdout}")
            if stderr:
                print(f"部分标准错误:\n{stderr}")
        except:
            pass
    
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        monitoring = False
    
    # 打印性能分析
    print(f"\n=== 性能分析 ===")
    if performance_data['timestamps']:
        max_cpu = max(performance_data['cpu_usage'])
        max_memory = max(performance_data['memory_usage'])
        avg_cpu = sum(performance_data['cpu_usage']) / len(performance_data['cpu_usage'])
        avg_memory = sum(performance_data['memory_usage']) / len(performance_data['memory_usage'])
        max_processes = max(performance_data['process_count'])
        
        print(f"运行时长: {performance_data['timestamps'][-1]:.2f}秒")
        print(f"最大CPU使用率: {max_cpu:.1f}%")
        print(f"平均CPU使用率: {avg_cpu:.1f}%")
        print(f"最大内存使用: {max_memory:.1f}MB")
        print(f"平均内存使用: {avg_memory:.1f}MB")
        print(f"最大进程数: {max_processes}")
        
        # 分析性能趋势
        if len(performance_data['cpu_usage']) > 10:
            first_half_cpu = sum(performance_data['cpu_usage'][:len(performance_data['cpu_usage'])//2])
            second_half_cpu = sum(performance_data['cpu_usage'][len(performance_data['cpu_usage'])//2:])
            
            if second_half_cpu > first_half_cpu * 1.5:
                print("⚠️  检测到CPU使用率随时间增加，可能存在性能衰减")
            elif second_half_cpu < first_half_cpu * 0.7:
                print("✅ CPU使用率趋于稳定或下降")
        
        # 检查是否有卡顿和等待
        stuck_periods = []
        waiting_periods = []
        low_cpu_periods = []
        
        for i in range(1, len(performance_data['cpu_usage'])):
            cpu = performance_data['cpu_usage'][i]
            timestamp = performance_data['timestamps'][i]
            
            # 检测卡顿 (CPU使用率极低且持续时间长)
            if cpu < 2 and i > 0:
                prev_cpu = performance_data['cpu_usage'][i-1]
                if prev_cpu < 2:  # 连续低CPU
                    if not waiting_periods or timestamp - waiting_periods[-1] > 10:
                        waiting_periods.append(timestamp)
            
            # 检测低CPU使用率时段
            if cpu < 10:
                low_cpu_periods.append(timestamp)
        
        if waiting_periods:
            print(f"⚠️  检测到 {len(waiting_periods)} 个可能的等待时段 (CPU < 2%)")
            
        if len(low_cpu_periods) > len(performance_data['cpu_usage']) * 0.7:
            print(f"⚠️  超过70%的时间CPU使用率低于10% - 可能存在I/O等待或进程间通信问题")
            
        # 分析CPU使用率模式
        high_cpu_periods = [cpu for cpu in performance_data['cpu_usage'] if cpu > 50]
        if high_cpu_periods:
            print(f"📊 高CPU使用率时段: {len(high_cpu_periods)} 次, 平均: {sum(high_cpu_periods)/len(high_cpu_periods):.1f}%")
        else:
            print("📊 未检测到高CPU使用率时段 - 程序可能在等待外部资源")
            
        # 分析内存使用模式
        memory_growth = performance_data['memory_usage'][-1] - performance_data['memory_usage'][0] if len(performance_data['memory_usage']) > 1 else 0
        if memory_growth > 50:  # 内存增长超过50MB
            print(f"⚠️  内存使用增长: {memory_growth:.1f}MB - 可能存在内存泄漏")
        
        # 分析进程数量变化
        if len(set(performance_data['process_count'])) > 1:
            print(f"📊 进程数量变化: {min(performance_data['process_count'])} - {max(performance_data['process_count'])}")
            
        # 计算程序运行效率
        total_time = performance_data['timestamps'][-1]
        active_time = sum(1 for cpu in performance_data['cpu_usage'] if cpu > 5) * 2  # 每2秒采样一次
        efficiency = (active_time / total_time) * 100 if total_time > 0 else 0
        print(f"📊 程序活跃度: {efficiency:.1f}% (CPU > 5%的时间占比)")
    
    return performance_data

def analyze_bottlenecks(performance_data):
    """分析性能瓶颈"""
    print(f"\n=== 瓶颈分析 ===")
    
    if not performance_data['timestamps']:
        print("没有性能数据可分析")
        return
        
    cpu_data = performance_data['cpu_usage']
    memory_data = performance_data['memory_usage']
    timestamps = performance_data['timestamps']
    
    # 分析初始化时间
    if len(timestamps) > 5:
        init_time = 10  # 假设前10秒是初始化
        init_samples = [i for i, t in enumerate(timestamps) if t < init_time]
        if init_samples:
            init_cpu_avg = sum(cpu_data[i] for i in init_samples) / len(init_samples)
            print(f"初始化阶段 (前{init_time}秒): 平均CPU {init_cpu_avg:.1f}%")
            
            if init_cpu_avg < 5:
                print("  -> ⚠️  初始化阶段CPU使用率很低，可能在等待文件加载或网络连接")
            elif init_cpu_avg > 80:
                print("  -> ⚠️  初始化阶段CPU使用率很高，可能在进行大量计算")
    
    # 分析运行阶段
    if len(timestamps) > 10:
        runtime_samples = [i for i, t in enumerate(timestamps) if t > 10]
        if runtime_samples:
            runtime_cpu_avg = sum(cpu_data[i] for i in runtime_samples) / len(runtime_samples)
            print(f"运行阶段: 平均CPU {runtime_cpu_avg:.1f}%")
            
            # 检测等待模式
            low_cpu_count = sum(1 for i in runtime_samples if cpu_data[i] < 5)
            if low_cpu_count > len(runtime_samples) * 0.8:
                print("  -> ⚠️  运行阶段80%以上时间CPU < 5%，程序可能在等待:")
                print("     - SUMO仿真步进")
                print("     - 网络I/O")
                print("     - 文件读写")
                print("     - 进程间通信")
    
    # 检测周期性模式
    if len(cpu_data) > 20:
        # 简单的周期性检测
        patterns = []
        window_size = 5
        for i in range(len(cpu_data) - window_size):
            window = cpu_data[i:i+window_size]
            avg_cpu = sum(window) / len(window)
            patterns.append(avg_cpu)
        
        # 检测是否有明显的高低交替
        high_periods = sum(1 for p in patterns if p > 30)
        low_periods = sum(1 for p in patterns if p < 10)
        
        if high_periods > 0 and low_periods > 0:
            print(f"检测到交替模式: {high_periods} 个高CPU时段, {low_periods} 个低CPU时段")
            if low_periods > high_periods * 2:
                print("  -> 程序大部分时间在等待，计算时间相对较短")
                
    # 建议
    print(f"\n=== 优化建议 ===")
    avg_cpu = sum(cpu_data) / len(cpu_data)
    
    if avg_cpu < 10:
        print("🎯 主要问题: CPU使用率过低")
        print("   建议检查:")
        print("   1. SUMO仿真步长设置是否过小")
        print("   2. 是否有不必要的sleep()或wait()调用") 
        print("   3. 网络延迟或I/O瓶颈")
        print("   4. Python GIL限制 (考虑使用多进程)")
        
    elif avg_cpu > 80:
        print("🎯 主要问题: CPU使用率过高")
        print("   建议检查:")
        print("   1. 算法复杂度是否可以优化")
        print("   2. 是否有不必要的重复计算")
        print("   3. 数据结构是否高效")
        
    else:
        print("✅ CPU使用率正常，需要进一步分析其他指标")

if __name__ == "__main__":
    import os
    # 检查环境
    print("检查SUMO环境...")
    if 'SUMO_HOME' not in os.environ:
        print("❌ 未设置SUMO_HOME环境变量")
        sys.exit(1)
    
    # 检查必要文件
    required_files = ['trpo.py', 'env.py']
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        sys.exit(1)
    
    print("✅ 环境检查通过")
    
    # 运行测试
    timeout = 300  # 5分钟超时
    if len(sys.argv) > 1:
        try:
            timeout = int(sys.argv[1])
        except ValueError:
            print("超时参数必须是整数(秒)")
            sys.exit(1)
    
    performance_data = run_with_timeout_and_monitor(timeout)
    
    # 运行详细的瓶颈分析
    analyze_bottlenecks(performance_data)