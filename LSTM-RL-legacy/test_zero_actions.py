import re

def create_zero_action_script(source, target):
    with open(source, 'r') as f:
        content = f.read()
    
    # Precise replacement for both scripts
    content = re.sub(
        r"action = sac_trainer\.policy_net\.get_action\([^)]+\)[\s]*,[\s]*deterministic=DETERMINISTIC\)",
        "action = np.zeros(action_dim, dtype=np.float32)",
        content
    )
    content = re.sub(
        r"action = sac_trainer\.policy_net\.get_action\([^)]+\.float\(\), deterministic=DETERMINISTIC\)",
        "action = np.zeros(action_dim, dtype=np.float32)",
        content
    )
    
    with open(target, 'w') as f:
        f.write(content)

create_zero_action_script('sac_v2_bus_SUMO_linear_penalty.py', 'sac_zero_vanilla.py')
create_zero_action_script('ensemble_version/sac_ensemble_SUMO_linear_penalty.py', 'sac_zero_ensemble.py')
