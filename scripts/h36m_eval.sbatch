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
# give different exp name to estimate evaluate h36m performance
source ./env_dis.sh

name=${1:-h36m_2d_ft_e4}
echo run SLP_eval
python eval.py --name $name --eval_ds h36m-p1 --svImg_freq 1 --testIter -1