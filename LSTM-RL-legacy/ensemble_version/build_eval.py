import os
import sys

def main():
    target_ext = 'sac_ensemble_SUMO_linear_penalty.py'
    with open(target_ext, 'r') as f:
        lines = f.readlines()

    new_lines = []
    
    for i, line in enumerate(lines):
        # Run only 1 episode
        if "for eps in range(start_episode, args.max_episodes):" in line:
            line = line.replace("for eps in range(start_episode, args.max_episodes):", "for eps in range(1):")
        
        # Inject load_model before the episode starts
        if "episode_start_time = time.time()" in line:
            indent = line[:len(line) - len(line.lstrip())]
            ckpt_path = "os.path.join(SAVE_ROOT, 'model', 'sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long', 'checkpoint_episode_40')"
            inject = f"{indent}sac_trainer.load_model({ckpt_path})\n"
            new_lines.append(inject)
            
        # Enforce deterministic evaluations
        if "DETERMINISTIC = False" in line:
            line = line.replace("DETERMINISTIC = False", "DETERMINISTIC = True")
            
        # Disable gradient updates
        if "if len(replay_buffer) > args.batch_size and" in line:
            line = line.replace("if len(replay_buffer) >", "if False and len(replay_buffer) >")
            
        # Prevent checkpoint overwriting
        if "sac_trainer.save_model" in line:
            line = line.replace("sac_trainer.save_model", "pass # sac_trainer.save_model")
            
        new_lines.append(line)

    with open('eval_v7long_ep40_det.py', 'w') as f:
        f.writelines(new_lines)
    print("Created eval_v7long_ep40_det.py")

if __name__ == '__main__':
    main()
