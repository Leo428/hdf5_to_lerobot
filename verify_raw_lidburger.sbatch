#!/usr/bin/env bash
# Byte-verify lid+burger RAW uploads: re-hash every local HDF5 (SHA-256) and compare path+size+
# lfs.sha256 against BOTH huzheyuan/ and CMU-AIRe/ public repos. Prints per-repo VERDICT + overall.
#   sbatch ~/hdf5_to_lerobot/verify_raw_lidburger.sbatch
#SBATCH --job-name=lbverify
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lbverify_%j.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lbverify_%j.err
set -uo pipefail
cd ~/hdf5_to_lerobot
export HF_TOKEN=$(tr -d '[:space:]' < ~/hf_key_2026.txt)   # never printed
unset HF_HUB_OFFLINE 2>/dev/null || true
export HASHJOBS=32
PY=~/miniforge3/bin/python3
rc=0
for fam in real_lid real_burger; do
  echo "############ VERIFY $fam ############"
  $PY -u verify_raw_upload.py "$fam" || rc=1
done
echo "OVERALL_VERIFY_RC=$rc"
exit $rc
