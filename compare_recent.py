import pandas as pd
import glob
import os

dirs = sorted(glob.glob('offline_sumo/logs/cql_*'))
for d in dirs[-5:]:
    df_path = os.path.join(d, 'progress.csv')
    print(f'--- {os.path.basename(d)} ---')
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        print(df[['epoch', 'loss', 'q_value', 'return']].tail(3).to_string(index=False))
