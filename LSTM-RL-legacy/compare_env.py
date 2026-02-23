with open("sac_zero_vanilla.py", "r") as f: v_lines = f.readlines()
with open("sac_zero_ensemble.py", "r") as f: e_lines = f.readlines()

def extract_env_init(lines):
    start = next(i for i, l in enumerate(lines) if "env_bus" in l)
    end = next(i for i, l in enumerate(lines) if "trajectory_log" in l)
    return lines[start:end]

v_init = extract_env_init(v_lines)
e_init = extract_env_init(e_lines)

import difflib
diff = list(difflib.unified_diff(v_init, e_init, n=0))
if len(diff) > 0:
    print("Found diff in Env Init:")
    print("".join(diff))
else:
    print("Env Init is IDENTICAL.")
