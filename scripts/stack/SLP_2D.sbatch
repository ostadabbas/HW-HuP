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
# 2D version, stacked modalities
source ./env_dis.sh

epoch=20
echo run SLP_2D stk $epoch
python train.py --name SLP_2D_stk_e$epoch --epoch_dp -1 --if_depth False --if_vis False --pretrained_checkpoint=data/model_checkpoint.pt --run_smplify --num_epochs $epoch --trainIter -1 --summary_steps 100 --batch_size 64 --ignore_3d \
 --li_mod_SLP depth IR PM --cov_li uncover cover1 cover2