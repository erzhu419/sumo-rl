# SAC v2 Bus 依赖关系层级图

## 🎯 主程序启动流程

```
sac_v2_bus.py
    │
    ├── 📦 直接导入模块
    │   ├── torch, numpy, gym (标准库)
    │   ├── normalization.py (本地模块)
    │   └── env.sim.env_bus (Legacy环境，fallback用)
    │
    ├── 🔧 参数解析 (argparse)
    │   ├── --sumo_bridge: 'SUMO_ruiguang.online_control.rl_bridge:build_bridge'
    │   ├── --sumo_root: 'SUMO_ruiguang/online_control'
    │   └── --render: 启用SUMO GUI
    │
    ├── 🌉 Bridge工厂模式 (第644行)
    │   │
    │   └── importlib.import_module('SUMO_ruiguang.online_control.rl_bridge')
    │       │
    │       └── build_bridge() 函数
    │           │
    │           ├── 返回: decision_provider (获取决策事件)
    │           ├── 返回: action_executor (执行动作)
    │           ├── 返回: reset_callback (重置环境)
    │           └── 返回: close_callback (关闭环境)
    │
    ├── 🏗️ 环境创建 (第662-670行)
    │   │
    │   └── SumoBusHoldingEnv()
    │       ├── decision_provider (来自bridge)
    │       ├── action_executor (来自bridge)
    │       ├── schedule_file: 'initialize_obj/save_obj_bus.add.xml'
    │       └── root_dir: 'SUMO_ruiguang/online_control'
    │
    └── 🤖 训练循环 (第720+行)
        ├── state_dict[line_id][bus_id] 结构
        ├── action_dict[line_id][bus_id] 结构
        └── replay_buffer.push(s,a,r,s',done)
```

## 🔗 SUMO集成模块详细层级

```
SUMO_ruiguang.online_control.rl_bridge
    │
    ├── 📋 SumoRLBridge 类
    │   │
    │   ├── _start_traci() → 启动SUMO仿真
    │   │   └── traci.start([sumo-gui/sumo, "-c", control_sim_traci_period.sumocfg])
    │   │
    │   ├── _load_objects() → 加载仿真对象
    │   │   │
    │   │   ├── f_8_create_obj.create_obj_fun()
    │   │   │   │
    │   │   │   ├── 📄 解析XML配置文件:
    │   │   │   │   ├── save_obj_bus.add.xml (公交车时刻表)
    │   │   │   │   ├── save_obj_stop.add.xml (车站信息)
    │   │   │   │   ├── save_obj_line.add.xml (线路信息)
    │   │   │   │   ├── save_obj_signal.add.xml (信号灯)
    │   │   │   │   ├── save_obj_lane.add.xml (车道)
    │   │   │   │   └── save_obj_passenger.add.xml (乘客)
    │   │   │   │
    │   │   │   └── 🏭 创建仿真对象:
    │   │   │       ├── bus_obj_dic: Dict[str, Bus]
    │   │   │       ├── stop_obj_dic: Dict[str, Stop]
    │   │   │       ├── line_obj_dic: Dict[str, Line]
    │   │   │       ├── signal_obj_dic: Dict[str, Signal]
    │   │   │       ├── lane_obj_dic: Dict[str, Lane]
    │   │   │       └── passenger_obj_dic: Dict[str, Passenger]
    │   │   │
    │   │   ├── e_8_gurobi_test_considerbusnum_V3.get_static_info()
    │   │   │   └── 返回: BusCap, AveAlightingTime, AveBoardingTime
    │   │   │
    │   │   └── d_8_compute_running_time.get_sorted_busline_edge()
    │   │       ├── 解析: intersection_delay/a_sorted_busline_edge.xml
    │   │       └── 返回: sorted_busline_edge_d, involved_tl_ID_l
    │   │
    │   ├── fetch_events() → RL决策提供者
    │   │   │
    │   │   ├── _advance_one_step() → 推进仿真一步
    │   │   │   ├── 更新车站状态: stop.update_stop_state()
    │   │   │   ├── 更新公交车状态: bus.bus_running()
    │   │   │   ├── 更新乘客状态: passenger.passenger_run()
    │   │   │   └── traci.simulationStep()
    │   │   │
    │   │   └── _collect_new_events() → 收集决策事件
    │   │       └── 返回: List[DecisionEvent]
    │   │           ├── line_id: str (线路ID)
    │   │           ├── bus_id: str (车辆ID)
    │   │           ├── stop_id: str (车站ID)
    │   │           ├── forward_headway: float (前向车头时距)
    │   │           ├── backward_headway: float (后向车头时距)
    │   │           ├── waiting_passengers: int (等车乘客数)
    │   │           └── base_stop_duration: float (基础停站时间)
    │   │
    │   └── apply_action() → RL动作执行者
    │       └── traci.vehicle.setStopParameter(bus_id, "duration", holding_time)
    │
    └── 🏭 build_bridge() 工厂函数
        └── 返回字典:
            ├── 'decision_provider': 获取决策事件的回调
            ├── 'action_executor': 执行holding动作的回调
            ├── 'reset_callback': 重置仿真的回调
            └── 'close_callback': 关闭仿真的回调
```

## 🏛️ 环境包装层

```
SumoBusHoldingEnv (rl_env.py)
    │
    ├── 📖 _load_schedule()
    │   ├── 解析: save_obj_bus.add.xml
    │   ├── 计算每条线路的median headway
    │   │   ├── 7X: 360秒, 7S: 360秒
    │   │   ├── 122S: 480秒, 122X: 540秒
    │   │   ├── 406S: 420秒, 406X: 480秒
    │   │   └── 311S: 600秒, 311X: 660秒...
    │   └── self._line_headway[line_id] = headway
    │
    ├── 🔄 step() 方法流程
    │   ├── _apply_actions() → 调用action_executor
    │   ├── _advance_until_state() → 调用decision_provider
    │   ├── _register_event() → 创建状态观测
    │   │   ├── 状态向量: [line_idx, bus_idx, station_idx, time_period, direction,
    │   │   │              forward_headway, backward_headway, waiting_passengers,
    │   │   │              target_headway, base_stop_duration] (10维)
    │   │   └── 奖励计算: -(|fh-th| + |bh-th|)/2 - |fh-bh|*0.5
    │   └── 返回: state_dict[line_id][bus_id] = [obs_history]
    │
    └── 📊 特征规格
        ├── 分类特征 (5个): line_id, bus_id, station_id, time_period, direction
        └── 连续特征 (5个): forward_headway, backward_headway, waiting_passengers,
                            target_headway, base_stop_duration
```

## 🔧 仿真对象层 (sim_obj/)

```
f_8_create_obj.create_obj_fun()
    │
    ├── 🚌 sim_obj.bus.Bus
    │   ├── bus_activate() → 激活公交车
    │   ├── bus_running() → 运行状态更新
    │   ├── just_server_stop_data_d → 刚服务的站点数据
    │   └── depart_stop_time_d → 离站时间记录
    │
    ├── 🚏 sim_obj.stop.Stop
    │   ├── update_stop_state() → 更新站点状态
    │   ├── get_passenger_arriver_rate() → 乘客到达率
    │   └── get_initial_just_leave_data() → 初始离站数据
    │
    ├── 🚇 sim_obj.line.Line
    │   ├── stop_id_l → 站点ID列表
    │   └── 线路相关属性
    │
    ├── 🚦 sim_obj.signal.Signal
    │   ├── get_attribute_by_traci() → 从TraCI获取属性
    │   └── get_pass_line() → 获取通过线路
    │
    ├── 🛣️ sim_obj.lane.Lane
    │   └── 车道几何信息
    │
    └── 🧍 sim_obj.passenger.Passenger
        ├── passenger_activate() → 激活乘客
        └── passenger_run() → 乘客行为更新
```

## 🗂️ 配置文件层

```
SUMO_ruiguang/online_control/initialize_obj/
    │
    ├── 📋 save_obj_bus.add.xml → 📊 公交车发车时刻表
    │   └── 用途: 计算线路特定headway, 创建bus对象
    │
    ├── 🚏 save_obj_stop.add.xml → 📍 车站位置和属性
    │   └── 用途: 创建stop对象, 乘客集散点
    │
    ├── 🚇 save_obj_line.add.xml → 🗺️ 线路站点顺序
    │   └── 用途: 创建line对象, 定义运营路径
    │
    ├── 🚦 save_obj_signal.add.xml → 🔴 信号灯控制
    │   └── 用途: 创建signal对象, 交通信号协调
    │
    ├── 🛣️ save_obj_lane.add.xml → 🛤️ 道路网络拓扑
    │   └── 用途: 创建lane对象, 路网连接关系
    │
    └── 🧍 save_obj_passenger.add.xml → 👥 乘客OD分布
        └── 用途: 创建passenger对象, 需求模式

intersection_delay/a_sorted_busline_edge.xml → 🚌 公交线路路段顺序
    └── 用途: 计算行驶时间, 路径规划

control_sim_traci_period.sumocfg → ⚙️ SUMO仿真配置
    └── 用途: TraCI连接配置, 仿真参数设置
```

## 💻 运行时数据流

```
启动命令: PYTHONPATH=. python LSTM-RL/sac_v2_bus.py --render --max_episodes 1
    │
    ├── 🔍 模块解析: PYTHONPATH=. 让Python找到SUMO_ruiguang/
    ├── 🖥️ GUI启用: --render 触发sumo-gui而非sumo
    └── ⏱️ 快速测试: --max_episodes 1 只运行1个episode
    │
    ▼
环境初始化流程:
    sac_v2_bus.py → rl_bridge → f_8_create_obj → XML解析 → SUMO对象创建
    │
    ▼
训练循环数据流:
    DecisionEvent → state_dict[line][bus] → SAC网络 → action → TraCI控制 → 环境反馈
    │                     │                    │           │
    └─────────────── (s,a,r,s') ──────────────┴── Replay Buffer ──┘
```

## 🏗️ 架构特点总结

- **🔄 桥接模式**: rl_bridge.py作为SUMO与RL的适配器
- **🏭 工厂模式**: build_bridge()返回标准化回调接口
- **📦 模块复用**: 复用g_8_SUMO.py的核心组件但不直接调用
- **🎯 专门化**: 针对bus holding control优化的环境接口
- **⚡ 高效**: 避免重复实现,直接复用成熟的SUMO仿真逻辑