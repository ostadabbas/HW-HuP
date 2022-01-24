#!/bin/bash
#SBATCH --job-name=IDP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=120Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/IDP.%j.out
#SBATCH --error=bch_outs/IDP.%j.err
# the original training process, maybe the dsNm needs to be no SLP for orignal version


python train.py --name train_demo --pretrained_checkpoint=data/model_checkpoint.pt --run_smplify --summary_step 2 --batch_size 3
