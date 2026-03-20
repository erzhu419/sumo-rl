import numpy as np
import os

log_dir = 'logs/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long'
npy_path = os.path.join(log_dir, 'rewards.npy')
backup_path = os.path.join(log_dir, 'rewards_latest_backup.npy')

if os.path.exists(npy_path):
    import shutil
    shutil.copy(npy_path, backup_path)
    
    r = np.load(npy_path)
    # The actual length recovered was 101.
    # If the user says episode 11 (index 11) is actually episode 60 (index 60)...
    # 60 - 11 = 49. Therefore, 49 episodes are missing from the front.
    
    missing_episodes = 49
    # We pad with NaNs to show they are missing, but matplotlib will handle them safely.
    padding = np.full(missing_episodes, np.nan)
    
    new_r = np.concatenate((padding, r))
    
    np.save(npy_path, new_r)
    print(f"Padded {missing_episodes} missing episodes at the start.")
    print(f"New length: {len(new_r)}")
    print(f"Max is now at index: {np.nanargmax(new_r)}")
else:
    print("Cannot find rewards.npy to pad.")
