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
# furhter add depth supervision.
# use depth added
source ./env_dis.sh

echo run SLP_3D_vis_d_cp0.1
python train.py --name SLP_3D_vis_d_cp0.1 --epoch_dp 2 --if_depth True --if_vis True --pretrained_checkpoint=data/model_checkpoint.pt --run_smplify --num_epochs 3 \
 --trainIter -1 --summary_steps 30 --batch_size 64 --pose_loss_weight 0.1 --opt_wt_pose 4.78