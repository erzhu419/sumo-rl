import re

def create_zero_action_script(source, target):
    with open(source, 'r') as f:
        content = f.read()
    
    # Corrected regex to match ANY assignment of get_action
    content = re.sub(
        r"sac_trainer\.policy_net\.get_action\([^)]+\)",
        "np.zeros(action_dim, dtype=np.float32)",
        content
    )
    
    # Change episodes to 1 for quick test and disable plots
    content = content.replace("args.max_episodes", "1")
    content = content.replace("if eps % args.plot_freq == 0:", "if False:")
    
    with open(target, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("生成全 0 动作测试脚本 V2 中...")
    create_zero_action_script('sac_v2_bus_SUMO_linear_penalty.py', 'sac_zero_vanilla.py')
    create_zero_action_script('ensemble_version/sac_ensemble_SUMO_linear_penalty.py', 'sac_zero_ensemble.py')
    
    # Fix paths for ensemble zero script
    with open('sac_zero_ensemble.py', 'r') as f:
        content = f.read()
    content = content.replace("os.path.join(os.path.dirname(__file__), '../..')", "os.path.join(os.path.dirname(__file__), '..')")
    with open('sac_zero_ensemble.py', 'w') as f:
        f.write(content)

    print("生成完毕。请运行以下命令进行对比:")
    print("python sac_zero_vanilla.py --train --use_sumo_env")
    print("python sac_zero_ensemble.py --train --use_sumo_env")
