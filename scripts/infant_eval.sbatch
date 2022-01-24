#!/bin/bash
#SBATCH --job-name=IDP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=32Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/IDP.%j.out
#SBATCH --error=bch_outs/IDP.%j.err
# use only the infant camL200 2d400 e500 version, input the MIMM or SyRIP for eval
source ./env_dis.sh

eval_ds=${1:-MIMM}
echo run infant_eval
python eval.py --name infant_camL200_ft_e640 --eval_ds $eval_ds --svImg_freq 1 --testIter -1 --if_cam_rct True
#python eval.py --name infant_ft_e180 --eval_ds $eval_ds --svImg_freq 1 --testIter -1