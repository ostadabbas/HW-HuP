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
# no prior in optimization process, evolve freely.
source ./env_dis.sh

epoch=${1:-30} # 30 epoch
echo run SLP_3D_vis_d_op0 $epoch
python train.py --name SLP_3D_vis_d_op0_e$epoch --epoch_dp 10 --if_depth True --if_vis True --pretrained_checkpoint=data/model_checkpoint.pt --run_smplify --num_epochs $epoch \
 --trainIter -1 --summary_steps 100 --batch_size 64 --opt_wt_pose 0.