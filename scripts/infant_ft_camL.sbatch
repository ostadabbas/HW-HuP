#!/bin/bash
#SBATCH --job-name=IDP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=32Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/HWS.%j.out
#SBATCH --error=bch_outs/HWS.%j.err   # heuristic weakly supervised human pose
# run infant fine tuning on MIMM SyRIP, cam rct to similar size, normal smplify iters  dp 100/130, add cam loss directly
#* current state, no save iter.  no dp to test 2D +3D to find a converged points., 320 longer
source ./env_dis.sh

epoch=${1:-640} # 30 epoch
echo run infant_camL
python train.py --name infant_camL200_ft_e$epoch --epoch_dp 420 --if_depth True --if_vis True --run_smplify --pretrained_checkpoint=data/model_checkpoint.pt --num_epochs $epoch \
 --trainIter -1 --summary_steps 50 --batch_size 64 --train_ds infant --openpose_train_weight 1 \
 --epoch_step $epoch --if_cam_rct True --cam_weight 1. --epoch_cam 200 --resume
