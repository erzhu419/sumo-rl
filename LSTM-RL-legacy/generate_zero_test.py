import re
import os

def create_zero_action_script(source, target):
    with open(source, 'r') as f:
        content = f.read()
    
    # 替换所有的 get_action 采样成全 0 动作
    content = re.sub(
        r"action = sac_trainer\.policy_net\.get_action\([^)]+\.float\(\), deterministic=DETERMINISTIC\)",
        "action = np.zeros(action_dim, dtype=np.float32)",
        content
    )
    
    # 将模型保存、log 等不必要的东西关掉，并设置跑1回合
    content = content.replace("args.max_episodes", "1")
    content = content.replace("if eps % args.plot_freq == 0:", "if False:")
    
    with open(target, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("生成全 0 动作测试脚本中...")
    create_zero_action_script('sac_v2_bus_SUMO_linear_penalty.py', 'sac_zero_vanilla.py')
    create_zero_action_script('ensemble_version/sac_ensemble_SUMO_linear_penalty.py', 'sac_zero_ensemble.py')
    print("生成完毕。请运行以下命令进行对比:")
    print("python sac_zero_vanilla.py --train --use_sumo_env")
    print("python sac_zero_ensemble.py --train --use_sumo_env")
