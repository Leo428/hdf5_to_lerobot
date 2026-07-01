#!/usr/bin/env bash
#SBATCH --job-name=rawverify
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/rawverify_%j.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/rawverify_%j.err
set -uo pipefail
cd ~/hdf5_to_lerobot
export HASHJOBS=32
PY=~/miniforge3/bin/python3
rc=0
for fam in sim_double_insert real_hang; do
  echo "############ VERIFY $fam ############"
  $PY -u verify_raw_upload.py "$fam" || rc=1
done
echo "OVERALL_VERIFY_RC=$rc"
exit $rc
