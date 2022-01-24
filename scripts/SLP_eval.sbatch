#!/bin/bash
#SBATCH --job-name=IDP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=64Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/IDP.%j.out
#SBATCH --error=bch_outs/IDP.%j.err
# run evaluation code , input name
source ./env_dis.sh
name=${1:-SLP_3D_vis_d_e30}
echo run SLP_eval
python eval.py --name $name --svImg_freq 3 --testIter -1