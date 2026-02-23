import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

log_path = sys.argv[1]

# Load Trajectory Log
df = pd.read_csv(log_path)

# Extract Hold and Speed from Action Column
# Format is either "Hold", "Hold_Speed", or numeric
def parse_action(x):
    if pd.isna(x):
        return [0.0, 1.0]
    
    parts = str(x).split('_')
    if len(parts) >= 2:
        return [float(parts[0]), float(parts[1])]
    else:
        val = float(parts[0])
        if val <= 5.0 and val > 0.0:
            return [0.0, val]
        else:
            return [val, 1.0]

parsed = df['Action'].apply(parse_action)
df['Hold'] = parsed.apply(lambda x: x[0])
df['Speed'] = parsed.apply(lambda x: x[1])

# Group by Step in chunks of 1000 for smoothing
df['Step_Bin'] = (df['Step'] // 5000) * 5000
agg_df = df.groupby('Step_Bin').agg({'Hold': 'mean', 'Speed': 'mean'}).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot Hold
axes[0].plot(agg_df['Step_Bin'], agg_df['Hold'], color='blue', label='Avg Hold Time')
axes[0].set_title('Average Holding Time Output over Steps')
axes[0].set_ylabel('Hold Time (s)')
axes[0].legend()
axes[0].grid(True)

# Plot Speed
axes[1].plot(agg_df['Step_Bin'], agg_df['Speed'], color='red', label='Avg Speed Ratio')
axes[1].set_title('Average Speed Ratio Output over Steps')
axes[1].set_xlabel('Environment Steps')
axes[1].set_ylabel('Speed Ratio')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('action_convergence.png')
print("Plot saved to action_convergence.png")
