import os

def fix_action(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "sac_trainer.policy_net.get_action" in line:
            indent = line[:len(line) - len(line.lstrip())]
            if "action_dict[line_id][bus_id] =" in line:
                lines[i] = indent + "action_dict[line_id][bus_id] = np.zeros(action_dim, dtype=np.float32)\n"
            elif "action =" in line:
                lines[i] = indent + "action = np.zeros(action_dim, dtype=np.float32)\n"
            elif "a =" in line:
                lines[i] = indent + "a = np.zeros(action_dim, dtype=np.float32)\n"
                
    with open(filepath, 'w') as f:
        f.writelines(lines)

fix_action('sac_zero_vanilla.py')
fix_action('sac_zero_ensemble.py')
