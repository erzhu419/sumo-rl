import os
import sys

def main():
    target_ext = 'sac_ensemble_SUMO_linear_penalty.py'
    with open(target_ext, 'r') as f:
        lines = f.readlines()

    new_lines = []
    
    # We want to replace the outer episode loop with a custom loop that iterates over checkpoints
    # Find the start of the episode loop
    loop_start_idx = -1
    for i, line in enumerate(lines):
        if "for eps in range(start_episode, args.max_episodes):" in line or "for eps in range(args.max_episodes):" in line:
            loop_start_idx = i
            break
            
    if loop_start_idx == -1:
        print("Could not find loop start")
        sys.exit(1)
        
    # Append everything before the loop
    # But inject a custom imports and variables at the top
    top_code = """
import numpy as np
import os
"""
    new_lines.extend(lines[:loop_start_idx])
    
    # Replace the loop with our custom checkpoint loading loop
    indent = lines[loop_start_idx][:len(lines[loop_start_idx]) - len(lines[loop_start_idx].lstrip())]
    
    custom_loop = f"""{indent}args.train = False
{indent}model_dir = "model/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long"
{indent}recovered_rewards = []
{indent}for ep_idx in range(49):
{indent}    ckpt = os.path.join(model_dir, f"checkpoint_episode_{{ep_idx}}")
{indent}    if not os.path.exists(ckpt + "_policy"):
{indent}        print(f"Skipping {{ep_idx}}")
{indent}        recovered_rewards.append(0.0)
{indent}        continue
{indent}    print(f"Loading {{ckpt}}")
{indent}    sac_trainer.load_model(ckpt)
{indent}    # Start simulation episode
"""
    new_lines.append(custom_loop)
    
    # Now append the inside of the loop, skipping the parts that save logs/models
    # We'll just run the simulation and grab `episode_reward`
    inside_loop = False
    for i in range(loop_start_idx + 1, len(lines)):
        line = lines[i]
        curr_indent = len(line) - len(line.lstrip())
        
        # Stop inclusion when we hit the sac_trainer.save_model("final") 
        if "sac_trainer.save_model(os.path.join(MODEL_DIR, \"final\"))" in line:
            break
            
        # Disable gradient updates
        if "if len(replay_buffer) > args.batch_size and" in line:
            line = line.replace("if len(replay_buffer) >", "if False and len(replay_buffer) >")
            
        # Disable logging to file
        if "np.save" in line or "plot(" in line or "sac_trainer.save_model" in line:
            line = f"{' ' * curr_indent}pass\n"
            
        # At the end of the episode loop (where it prints Episode: X), we want to append the reward
        if "print(" in line and "Episode Reward:" in line and "Duration:" in line:
            new_lines.append(line)
            # Inject saving reward
            save_inject = f"{' ' * curr_indent}recovered_rewards.append(episode_reward)\n"
            new_lines.append(save_inject)
            continue
            
        new_lines.append(line)
        
    end_code = f"""
{indent}np.save(os.path.join(model_dir, "recovered_first_48_eval.npy"), np.array(recovered_rewards))
{indent}print("Done recovery! Max:", np.max(recovered_rewards), "at index", np.argmax(recovered_rewards))
"""
    new_lines.append(end_code)
    
    with open('eval_recovery_script.py', 'w') as f:
        f.writelines(new_lines)
        
if __name__ == '__main__':
    main()
