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
# no smplify , direct 2d and depth like a naive fine tuning, IR version check initial guess summary back to 100 if full run
source ./env_dis.sh

epoch=20
echo run SLP_3D_vis_d_noOpt stk  $epoch
python train.py --name SLP_3D_vis_d_noOpt_stk_e$epoch --epoch_dp 0 --if_depth True --if_vis True --pretrained_checkpoint=data/model_checkpoint.pt --num_epochs $epoch \
 --trainIter -1 --summary_steps 20 --batch_size 64 --ignore_3d \
 --li_mod_SLP depth IR PM  --cov_li uncover cover1 cover2