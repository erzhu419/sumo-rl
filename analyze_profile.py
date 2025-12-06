import pstats
import sys

file_path = "sumo-rl3.pstat"
output_path = "profile_report.txt"
try:
    with open(output_path, 'w') as f:
        p = pstats.Stats(file_path, stream=f)
        f.write(f"Analysis of {file_path}:\n")
        f.write("-" * 40 + "\n")
        f.write("Top 50 by Cumulative Time (tottime + subcalls):\n")
        p.sort_stats('cumulative').print_stats(50)
        f.write("-" * 40 + "\n")
        f.write("Top 50 by Total Time (internal time only):\n")
        p.sort_stats('tottime').print_stats(50)
    print(f"Report saved to {output_path}")
except Exception as e:
    print(f"Error reading profile: {e}")
