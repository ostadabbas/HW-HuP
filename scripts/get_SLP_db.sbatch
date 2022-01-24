#!/bin/bash
#SBATCH --job-name=SLP_db
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=short
#SBATCH --mem=64Gb
#SBATCH --time=4:00:00
#SBATCH --output=bch_outs/SLP_db.%j.out
#SBATCH --error=bch_outs/SLP_db.%j.err
# process the SLP db
source ./env_dis.sh

echo process get_3d_dp.py and get_db_SLP_SPIN.py with hn_ratio ${1:-0.7}
python datasets/preprocess/get_3d_dp.py  --ratio_head_neck ${1:-0.7}
python datasets/preprocess/get_db_SLP_SPIN.py  --ratio_head_neck ${1:-0.7}