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
# evaluate with pretrained on all cover conditions for individual modNm,  1. modNm 2, true mod in
source ./env_dis.sh

modNm=${1:-depth}
mod=${2:-depth}
cmd=python
echo run SLP_eval
$cmd eval.py --name SLP_2D_${modNm}_preT --svImg_freq 3 --testIter -1 --checkpoint data/model_checkpoint.pt \
 --li_mod_SLP $mod --cov_li uncover cover1 cover2