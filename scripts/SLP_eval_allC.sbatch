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
# run evaluation code , input the exp name ,  the modality  in double quote if multiple, mod is actually used. have to save at every single as cov next to each other
source ./env_dis.sh

name=${1:-SLP_3D_stk_e30}
mod=${2:-depth}
cmd=python
echo run SLP_eval all C
$cmd eval.py --name $name --svImg_freq 1 --testIter -1 \
 --li_mod_SLP $mod --cov_li uncover cover1 cover2