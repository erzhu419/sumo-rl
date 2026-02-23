import numpy as np
import matplotlib.pyplot as plt
import os
import sys

log_dir_2d = 'logs/sac_ensemble_SUMO_linear_penalty_speed_control_run'
log_dir_1d = 'logs/sac_ensemble_SUMO_linear_penalty_holding_only_test'

output_path = '/home/erzhu419/.gemini/antigravity/brain/f432b4f8-d48f-4aa3-a3bf-c12d52fcb1d7/metrics_compare.png'

metrics = ['rewards.npy', 'q_values_episode.npy', 'log_probs_episode.npy', 'alpha_values_episode.npy']

fig, axs = plt.subplots(4, 1, figsize=(10, 15))

for i, metric in enumerate(metrics):
    try:
        data_2d = np.load(os.path.join(log_dir_2d, metric))
        axs[i].plot(data_2d, label='2D (Speed+Hold)', color='blue')
        print(f"Loaded {metric} for 2D, length: {len(data_2d)}")
    except Exception as e:
        print(f"Error loading {metric} for 2D: {e}")
        
    try:
        data_1d = np.load(os.path.join(log_dir_1d, metric))
        axs[i].plot(data_1d, label='1D (Hold Only)', color='red')
        print(f"Loaded {metric} for 1D, length: {len(data_1d)}")
    except Exception as e:
        print(f"Error loading {metric} for 1D: {e}")

    axs[i].set_title(metric.replace('.npy', ''))
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_xlabel('Episodes / Plot Freq')
    if metric == 'log_probs_episode.npy':
        axs[i].set_yscale('symlog')

plt.tight_layout()
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
