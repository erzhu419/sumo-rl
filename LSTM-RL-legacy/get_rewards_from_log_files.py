import re

def extract(path):
    try:
        with open(path, "r") as f:
            for line in f:
               if "Episode: 0" in line:
                   print(f"{path}: {line.strip()}")
    except:
        pass

extract("trajectory_log.csv")
extract("action_log.csv")
extract("log_v.txt")
extract("log_e.txt")

