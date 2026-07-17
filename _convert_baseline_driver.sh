#!/usr/bin/env bash
# Convert the remaining baseline xArm rounds (round4..round6) to per-round LeRobot
# datasets, sequentially one at a time. round1, round1good, round2, round3 are
# already converted; round4 was interrupted (its partial output is wiped+redone).
# Outputs: local/xarm_baseline_round{4,5,6} under HF_LEROBOT_HOME.
set -u

ROOT=/media/huzheyuan/data0/huzheyuan_folder_backup/dual_xarms/dual_xarms_sim/data/ADVERSARIAL
export HF_LEROBOT_HOME=/media/huzheyuan/data0/lerobot
LOGDIR=/tmp/xarm_baseline_convert
mkdir -p "$LOGDIR"
cd /home/huzheyuan/openpi-test || exit 1

echo "DRIVER RESUME (sequential; round1,round1good,round2,round3 already done) $(date) pid=$$ host=$(hostname)" >> "$LOGDIR/_driver.log"

rounds=(
  xarm_baseline_round4
  xarm_baseline_round5
  xarm_baseline_round6
)
MAXJOBS=1

for r in "${rounds[@]}"; do
  # Throttle to MAXJOBS concurrent converter processes (=1 -> strictly sequential).
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  (
    echo "START $r $(date)" >> "$LOGDIR/_driver.log"
    uv run examples/xarm/convert_xarm_data_to_lerobot.py \
      --raw-root "$ROOT" --repo-id "local/$r" --rounds "$r" \
      > "$LOGDIR/$r.log" 2>&1
    rc=$?
    echo "DONE $r rc=$rc $(date)" >> "$LOGDIR/_driver.log"
  ) &
done

wait
echo "DRIVER ALL DONE (sequential) $(date)" >> "$LOGDIR/_driver.log"
