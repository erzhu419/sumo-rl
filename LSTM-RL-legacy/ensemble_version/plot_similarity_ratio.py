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

# Old metric: Ridge scaling
def get_ridge_reward(fh, bh):
    # Absolute deviation for fwd/bwd
    dev_f = abs(fh - t_f)
    dev_b = abs(bh - t_b)
    
    # Simple linear penalty for fwd/bwd distance
    fwd_reward = -dev_f
    bwd_reward = -dev_b
    
    weight = dev_f / (dev_f + dev_b + 1e-6)
    
    # Current implemented metric
    sim_bonus = -abs(fh - R * bh) * 0.5 / ((1 + R)/2)
    
    reward = fwd_reward * weight + bwd_reward * (1-weight) + sim_bonus
    
    # Current threshold logic
    if dev_f > 0.5 * t_f or dev_b > 0.5 * t_b:
        reward -= 20
        
    return reward

Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = get_ridge_reward(X[i, j], Y[i, j])

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
    title=f'Linear Reward Surface (Target {t_f}:{t_b})',
    scene=dict(
        xaxis_title='Forward Headway (s)',
        yaxis_title='Backward Headway (s)',
        zaxis_title='Reward'
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.write_html("reward_surface_asymmetric.html")
print("Image saved to reward_surface_asymmetric.jpg")
