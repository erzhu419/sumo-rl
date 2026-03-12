import numpy as np

for a_hold, a_speed in [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-0.5, -0.6], [0.5, 0.4]]:
    hold = np.clip((a_hold + 1.0) * 60.0, 0.0, 120.0)
    
    logits = np.array([
        -abs(a_speed - (-0.8)),
        -abs(a_speed - (-0.4)),
        -abs(a_speed - 0.0),
        -abs(a_speed - 0.4),
        -abs(a_speed - 0.8)
    ]) * 5.0
    
    probs = np.exp(logits) / np.sum(np.exp(logits))
    print(f"Original [-1, 1]: ({a_hold}, {a_speed})")
    print(f"  Mapped Hold (0-120): {hold}")
    print(f"  Speed Softmax Probs (0.8, 0.9, 1.0, 1.1, 1.2):")
    for val, p in zip([0.8, 0.9, 1.0, 1.1, 1.2], probs):
        print(f"    {val}: {p*100:.1f}%")
