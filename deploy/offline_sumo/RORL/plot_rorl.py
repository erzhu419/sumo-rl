import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_rorl(log_dir):
    csv_path = os.path.join(log_dir, 'progress.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Map headers to more readable names
    # replay_buffer/size,policy_trainer/Policy Smooth Loss,policy_trainer/Q OOD Loss,
    # policy_trainer/q_ood_uncertainty_reg,policy_trainer/QFs Loss,policy_trainer/Policy Loss,
    # evaluation/Average Returns,Epoch
    
    metrics = {
        'evaluation/Average Returns': 'Average Evaluation Return',
        'policy_trainer/QFs Loss': 'Q-Function Loss',
        'policy_trainer/Policy Loss': 'Policy Loss',
        'policy_trainer/Q OOD Loss': 'Q OOD Loss (Uncertainty)',
        'policy_trainer/Qs Predictions Mean': 'Mean Q Prediction'
    }

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axs = [axs]

    for i, (col, title) in enumerate(metrics.items()):
        if col in df.columns:
            axs[i].plot(df['Epoch'], df[col], label=title)
            axs[i].set_title(title)
            axs[i].set_xlabel('Epoch')
            axs[i].grid(True)
        else:
            axs[i].text(0.5, 0.5, f'Metric {col} not found', ha='center')

    plt.tight_layout()
    save_path = os.path.join(log_dir, 'training_results.png')
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="Path to RORL log directory")
    args = parser.parse_args()
    plot_rorl(args.log_dir)
