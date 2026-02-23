import pandas as pd
import sys

try:
    df_v = pd.read_csv('action_log_vanilla.csv')
    df_e = pd.read_csv('action_log_ensemble.csv')
    
    print(f"Vanilla rows: {len(df_v)}")
    print(f"Ensemble rows: {len(df_e)}")
    
    min_len = min(len(df_v), len(df_e))
    diffs = []
    
    for i in range(min_len):
        row_v = df_v.iloc[i]
        row_e = df_e.iloc[i]
        
        # compare step, bus, etc.
        if (row_v['LineID'] != row_e['LineID']) or (row_v['BusID'] != row_e['BusID']) or (row_v['Step'] != row_e['Step']) or (round(row_v['Action'],4) != round(row_e['Action'],4)):
            diffs.append((i, row_v.to_dict(), row_e.to_dict()))
            if len(diffs) > 5:
                break
                
    if len(diffs) == 0:
        print(f"First {min_len} rows are perfectly identical in action_logs.")
    else:
        print(f"Found {len(diffs)} differences:")
        for idx, rv, re in diffs:
            print(f"Row {idx}:")
            print(f"  V: {rv}")
            print(f"  E: {re}")
            
except Exception as e:
    print(f"Error: {e}")
