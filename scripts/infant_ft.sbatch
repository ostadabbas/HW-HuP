#!/bin/bash
#SBATCH --job-name=IDP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=32Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/HWS.%j.out
#SBATCH --error=bch_outs/HWS.%j.err   # heuristic weakly supervised human pose
# run infant fine tuning on MIMM SyRIP, longer smplify iters 200
source ./env_dis.sh

epoch=${1:-180} # 30 epoch
echo run infant_ft
python train.py --name infant_ft_e$epoch --epoch_dp 160 --if_depth True --if_vis True --run_smplify --pretrained_checkpoint=data/model_checkpoint.pt --num_epochs $epoch \
 --trainIter -1 --summary_steps 50 --batch_size 64 --train_ds infant --checkpoint_steps 250 --openpose_train_weight 1 \
 --epoch_step $epoch --num_smplify_iters 200 --if_cam_rct False --resume