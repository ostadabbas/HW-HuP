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
# SLP 3D HMR , no smplify in the loop --run_smplify  --ignore_3d --ratio_head_neck defaut 0.7. 2d for combined version
# combine h36m and 2d
source ./env_dis.sh
epoch=${1:-4} # 30 epoch
echo run h36m_ft
python train.py --name h36m_2d_ft_e$epoch --epoch_dp 2 --if_depth True --if_vis True --run_smplify --pretrained_checkpoint=data/model_checkpoint.pt --num_epochs $epoch \
 --trainIter 2 --summary_steps 50 --batch_size 64 --train_ds h36m_2d --checkpoint_steps 250 --openpose_train_weight 0