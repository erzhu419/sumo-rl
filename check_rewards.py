import numpy as np
path = '/home/erzhu419/mine_code/sumo-rl/LSTM-RL-legacy/ensemble_version/logs/sac_ensemble_SUMO_linear_penalty_Production_Augmented_BangBang_V7_Long/reward.npy'
data = np.load(path)
print("Shape:", data.shape)
print("Latest rewards (35-45):")
for i in range(35, min(46, len(data))):
    print(f"Episode {i}: {data[i]}")
