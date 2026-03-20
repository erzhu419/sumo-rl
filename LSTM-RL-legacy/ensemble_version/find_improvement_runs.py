import os
import numpy as np
import glob

def identify_improvement_runs():
    logs_dir = "/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/logs"
    reward_files = glob.glob(os.path.join(logs_dir, "*/rewards.npy"))
    
    threshold_low = -1500000
    threshold_high = -800000
    
    results = []
    
    for r_file in sorted(reward_files):
        try:
            dir_name = os.path.basename(os.path.dirname(r_file))
            data = np.load(r_file)
            if data.ndim == 0 or len(data) < 10:
                continue
                
            # Take average of first 5 and last 5 to be robust to noise
            start_reward = np.mean(data[:5])
            end_reward = np.mean(data[-5:])
            max_reward = np.max(data)
            
            improvement = end_reward - start_reward
            
            status = "N/A"
            if start_reward <= threshold_low + 200000 and end_reward >= threshold_high - 100000:
                status = "MATCH"
            
            results.append({
                'run': dir_name,
                'start': start_reward,
                'end': end_reward,
                'max': max_reward,
                'len': len(data),
                'match': status
            })
        except Exception as e:
            pass

    print(f"{'Run Name':<60} | {'Start':<10} | {'End':<10} | {'Max':<10} | {'Len':<5} | {'Status'}")
    print("-" * 110)
    
    for r in results:
        match_str = f"*** {r['match']} ***" if r['match'] == "MATCH" else ""
        print(f"{r['run']:<60} | {r['start']/1e6:>9.2f}M | {r['end']/1e6:>9.2f}M | {r['max']/1e6:>9.2f}M | {r['len']:<5} | {match_str}")

if __name__ == "__main__":
    identify_improvement_runs()
