import os
import numpy as np

run_name = "Production_Augmented_BangBang_V7_Long"
model_dir = f"model/sac_ensemble_SUMO_linear_penalty_{run_name}"
log_dir = f"logs/sac_ensemble_SUMO_linear_penalty_{run_name}"

all_results = []
for w in range(14):
    res_path = f"eval_workers/results_{w}.npy"
    if os.path.exists(res_path):
        res = np.load(res_path)
        for r in res:
            all_results.append((int(r[0]), float(r[1])))

all_results.sort(key=lambda x: x[0])
recovered_rewards = [r[1] for r in all_results]

np.save(os.path.join(model_dir, "recovered_first_48_eval.npy"), np.array(recovered_rewards))

main_npy_path = os.path.join(log_dir, 'rewards.npy')
if os.path.exists(main_npy_path):
    main_r = np.load(main_npy_path)
    
    if len(main_r) >= 49:
        if np.all(np.isnan(main_r[:49])):
            main_r = main_r[49:]
            
    new_combined = np.concatenate((np.array(recovered_rewards), main_r))
    np.save(main_npy_path, new_combined)
    
    print(f"Successfully merged! New length: {len(new_combined)}")
else:
    print("Main file not found.")
