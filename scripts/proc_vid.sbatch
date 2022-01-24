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
# processing the video , generate the combined video out
source ./env_dis.sh

echo run video processing script

# for SLP RGB
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/SLP/vidRGB.avi \
#--exp_fd logs/SLP_3D_vis_d_e30
# HMR version
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/SLP/vidRGB.avi \
#--exp_fd logs/SLP_2D_HMR_e30
# SPIN version
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/SLP/vidRGB.avi \
#--exp_fd logs/SLP_2D_e30
# for SLP IR
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/SLP/vidIR.avi \
#--exp_fd logs/SLP_3D_vis_d_IR_e30

# for infant baby1 , can change to baby 2
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/infants/baby4.mp4 \
#--exp_fd logs/infant_camL200_ft_e640
# for infant baby2
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/infants/baby_s3.mp4 \
#--exp_fd logs/infant_camL200_ft_e640 \
#--bb 290 150 760 420
# for infant alex_stair
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/infants/alex_stair.mp4 \
#--exp_fd logs/infant_camL200_ft_e640 \
#--bb 330 60 630 420
# for infant alex_stair
#python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/infants/alex_kitchen.mp4 \
#--exp_fd logs/infant_camL200_ft_e640 \
#--bb 330 120 630 470
# for infant alex_stair
python proc_vid.py --vid_file /scratch/liu.shu/datasets/vid/infants/alex_kitchen1.mp4 \
--exp_fd logs/infant_camL200_ft_e640 \
--bb 330 90 630 500
