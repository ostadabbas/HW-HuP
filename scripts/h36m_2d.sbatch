#!/bin/bash
#SBATCH --job-name=HWS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=64Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/HWS.%j.out
#SBATCH --error=bch_outs/HWS.%j.err
# depth at epoch_dp epoch, with h36m2d
# tmp:  sumary steps 100
source ./env_dis.sh

epoch_dp=${1:-46} # 30 epoch    to test 35(40)
epoch=${2:-50} # 30 epoch    2, 5 for test
echo run h36m_2d scratch
python train.py --name h36m_2d_d${epoch_dp}_e$epoch --epoch_dp $epoch_dp --if_depth True --if_vis True --run_smplify --num_epochs $epoch \
 --trainIter -1 --summary_steps 100 --batch_size 64 --train_ds h36m_2d --checkpoint_steps 500 --openpose_train_weight 0 \
 --resume --epoch_step 12