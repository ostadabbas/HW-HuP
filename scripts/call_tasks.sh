#!/bin/bash
# just call the tasks I want to run
#source ./env_dis.sh
#

# echo or sbatch or sh
cmd=sbatch
## for training loop
epoch=${1:-30} # 30 epoch
#sbatch --time $( t2sd ) scripts/SLP_2D.sbatch $epoch
#sbatch --time $( t2sd ) scripts/SLP_2D_HMR.sbatch $epoch
#sbatch --time $( t2sd ) scripts/SLP_3D.sbatch $epoch
#sbatch --time $( t2sd ) scripts/SLP_3D_vis.sbatch $epoch
#sbatch --time $( t2sd ) scripts/SLP_3D_vis_d.sbatch $epoch
#sbatch --time $( t2sd ) scripts/SLP_3D_vis_d_noOpt.sbatch $epoch
#sbatch --time $( t2sd ) scripts/SLP_3D_vis_d_op0.sbatch $epoch

## testing loop, get needed result here,  only 2d hmr and vis_d ran
# naive,  j3d_dp, op0 no prior from source,  noOpt direct .
#sbatch --time $( t2sd ) scripts/SLP_eval_preT.sbatch SLP_2D_preT
#sbatch --time $( t2sd ) scripts/SLP_eval.sbatch SLP_2D_e30
#sbatch --time $( t2sd ) scripts/SLP_eval.sbatch SLP_2D_HMR_e30
#sbatch --time $( t2sd ) scripts/SLP_eval.sbatch SLP_3D_e30
#sbatch --time $( t2sd ) scripts/SLP_eval.sbatch SLP_3D_vis_e30
#sbatch --time $( t2sd ) scripts/SLP_eval.sbatch SLP_3D_vis_d_e30
#sbatch --time $( t2sd ) scripts/SLP_eval.sbatch SLP_3D_vis_d_noOpt_e30
#sbatch --time $( t2sd ) scripts/SLP_eval.sbatch SLP_3D_vis_d_op0_e30

## for multi-modal train, default e30, depth , IR , PM,   mod can be "depth IR PM" in quote, string can not pass in
# single will not interpret
#modNm=stk
#mod="depth IR PM"
#$cmd scripts/allC/SLP_2D.sbatch "$modNm" "$mod"
#$cmd scripts/allC/SLP_2D_HMR.sbatch "$modNm" "$mod"
#$cmd scripts/allC/SLP_3D_vis_d.sbatch "$modNm" "$mod"
#$cmd scripts/allC/SLP_3D_vis_d_noOpt.sbatch "$modNm" "$mod"

#
# testing loop for the  multi-modal allC , gives the name and true mod, stk with "d ir pm" others same name
#modNm=stk
#mod="depth IR PM"
modNm=PM
mod=PM
#$cmd scripts/SLP_eval_preT_allC.sbatch "${modNm}" "${mod}"
#$cmd scripts/SLP_eval_allC.sbatch "SLP_2D_${modNm}_e30" "$mod"
#$cmd scripts/SLP_eval_allC.sbatch "SLP_2D_HMR_${modNm}_e30" "$mod"
$cmd scripts/SLP_eval_allC.sbatch "SLP_3D_vis_d_${modNm}_e30" "$mod"
#$cmd scripts/SLP_eval_allC.sbatch "SLP_3D_vis_d_noOpt_${modNm}_e30" "$mod"

