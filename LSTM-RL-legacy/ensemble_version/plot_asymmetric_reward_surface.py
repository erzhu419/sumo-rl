import numpy as np
import plotly.graph_objs as go

# Target headways
t_f = 720
t_b = 120
R = t_f / max(t_b, 1e-6)

# Headway ranges
fh_range = np.linspace(100, 1000, 100)
bh_range = np.linspace(50, 400, 100)
X, Y = np.meshgrid(fh_range, bh_range)

# Smooth Tanh scaled target deviations
def get_smooth_asymmetric_reward(fh, bh):
    dev_f = abs(fh - t_f)
    dev_b = abs(bh - t_b)
    
    # 距离惩罚
    fwd_reward = -dev_f
    bwd_reward = -dev_b
    
    weight = dev_f / (dev_f + dev_b + 1e-6)
    
    # 比例同频惩罚 (Proportional Similarity Bonus)
    sim_bonus = -abs(fh - R * bh) * 0.5 / ((1 + R)/2)
    
    reward = fwd_reward * weight + bwd_reward * (1-weight) + sim_bonus
    
    # Tanh 平滑衰减惩罚 (超出50%容差限度后急剧下跌)
    f_pen = 20.0 * np.tanh((abs(fh - t_f) - 0.5 * t_f) / 30.0)
    b_pen = 20.0 * np.tanh((abs(bh - t_b) - 0.5 * t_b) / 30.0)
    
    penalty_factor = f_pen + b_pen
    reward -= max(0.0, penalty_factor)
    
    return reward

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = get_smooth_asymmetric_reward(X[i, j], Y[i, j])

# 计算梯度，使边缘下落呈现颜色变化
dz_dx, dz_dy = np.gradient(Z)
grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)

trace = go.Scatter3d(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    mode='markers',
    marker=dict(
        size=3,
        color=grad_mag.flatten(), 
        colorscale='Viridis',
        opacity=0.9,
    )
)

layout = go.Layout(
    title=f'Smoothed Asymmetric Reward Surface (Target {t_f}:{t_b})',
    scene=dict(
        xaxis_title='Forward Headway (s)',
        yaxis_title='Backward Headway (s)',
        zaxis_title='Reward',
        camera=dict(eye=dict(x=1.7, y=1.7, z=1.1))
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.write_html("asymmetric_reward_surface.html")
print("Render complete! Saved to asymmetric_reward_surface.html")
