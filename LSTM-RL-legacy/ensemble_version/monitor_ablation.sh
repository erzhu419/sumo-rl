#!/bin/bash
# Monitor 3 ablation trainings side by side
echo "========================================="
echo "  ABLATION STUDY MONITOR $(date '+%H:%M:%S')"
echo "========================================="
echo ""

for label_log in "f543609(old,hw=360):train_abl_f543609.log" "c9f82e8(new,hw=real):train_abl_c9f82e8.log" "HEAD(current):train_abl_HEAD.log"; do
  label="${label_log%%:*}"
  logfile="${label_log##*:}"
  
  echo "--- $label ---"
  if [ -f "$logfile" ]; then
    last_ep=$(grep "Episode:" "$logfile" | tail -1)
    last_alpha=$(grep "Alpha Loss:" "$logfile" | tail -1)
    if [ -n "$last_ep" ]; then
      echo "  $last_ep" | sed 's/^  */  /'
      echo "  $last_alpha" | sed 's/^  */  /'
    else
      echo "  (still initializing...)"
    fi
  else
    echo "  (log not found)"
  fi
  echo ""
done

echo "--- All Episode Rewards ---"
printf "%-6s | %-22s | %-22s | %-22s\n" "Ep" "f543609(hw=360)" "c9f82e8(hw=real)" "HEAD(current)" 
printf "%-6s-+-%-22s-+-%-22s-+-%-22s\n" "------" "----------------------" "----------------------" "----------------------"

paste <(grep "Episode:" train_abl_f543609.log 2>/dev/null | awk -F'[|]' '{gsub(/[^0-9]/,"",$1); gsub(/[^0-9.-]/,"",$2); printf "%s|%s\n",$1,$2}') \
      <(grep "Episode:" train_abl_c9f82e8.log  2>/dev/null | awk -F'[|]' '{gsub(/[^0-9.-]/,"",$2); printf "%s\n",$2}') \
      <(grep "Episode:" train_abl_HEAD.log      2>/dev/null | awk -F'[|]' '{gsub(/[^0-9.-]/,"",$2); printf "%s\n",$2}') | \
while IFS=$'\t' read -r col1 col2 col3; do
  ep="${col1%%|*}"
  r1="${col1##*|}"
  printf "%-6s | %-22s | %-22s | %-22s\n" "$ep" "${r1:-(-)}" "${col2:-(-)}" "${col3:-(-)}"
done 2>/dev/null

echo ""
echo "--- Processes ---"
ps aux | grep "sac_ensemble" | grep python | grep -v grep | awk '{printf "  PID:%s CPU:%s%% MEM:%.0fMB Time:%s\n",$2,$3,$6/1024,$10}'
