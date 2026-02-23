import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def compare_logs(new_log_dir, old_log_dir):
    new_csv = os.path.join(new_log_dir, 'progress.csv')
    old_csv = os.path.join(old_log_dir, 'progress.csv')
    
    if not os.path.exists(new_csv):
        print(f"Error: New log {new_csv} not found")
        return
    if not os.path.exists(old_csv):
        print(f"Error: Old log {old_csv} not found")
        return
        
    df_new = pd.read_csv(new_csv)
    df_old = pd.read_csv(old_csv)
    
    # Calculate Last 10 Average
    avg_new = df_new['return'].tail(10).mean()
    avg_old = df_old['return'].tail(10).mean()
    
    print(f"Average Return (Last 10 Epochs):")
    print(f"New Data (Aligned): {avg_new:.2f}")
    print(f"Old Data (Baseline): {avg_old:.2f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_new['epoch'], df_new['return'], label=f'New Data (Avg: {avg_new:.1f})', color='green')
    plt.plot(df_old['epoch'], df_old['return'], label=f'Old Data (Avg: {avg_old:.1f})', color='red', alpha=0.6)
    
    plt.title("CQL Performance Comparison: New vs Old Data")
    plt.xlabel("Epoch")
    plt.ylabel("Evaluation Return")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.getcwd(), 'cql_comparison.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Hardcoded for now based on exploration
    base_dir = "offline_sumo/logs"
    new_dir = os.path.join(base_dir, "cql_20260113-210158")
    old_dir = os.path.join(base_dir, "cql_20260113-101657")
    
    compare_logs(new_dir, old_dir)
