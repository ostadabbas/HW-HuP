"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Add SLP operation,  read model from exp folder or from load. Result to exp folder.
read either from specific model weights, or from the latest of current one.
exp specific evaluation.
this only works for the one with depth evaluaitons
Example usage:
```
python3 eval_ori.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import os.path as osp
from glob import glob

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset, SLP_RD, SPIN_ADP
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
import utils.utils as ut_t

from utils.renderer import Renderer
from models import hmr, SMPL
from utils import TrainOptions

action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
                'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']


def run_evaluation(model, dataset_name, dataset, out_fd='logs/tmp',
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50, svImg_freq=3, iter=-1, if_ldImg=False, if_cam_rct=False):
    """Run evaluation on the datasets and metrics we report in the paper.
    if_ldImg: use the lead image to save the  no image , 5 bits
    """
    # context setting
    result_file = osp.join(out_fd, 'eval_rst.npz')
    metric_file = osp.join(out_fd, 'eval_metric.json')
    metric = {}     # dictionary
    if svImg_freq>0:
        vid_fd = osp.join(out_fd, 'vis')
        if not osp.exists(vid_fd):
            os.makedirs(vid_fd)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    if_ptc = True
    if dataset_name in ['MIMM', 'SyRIP']:
        smpl_pth = osp.join(config.SMPL_MODEL_DIR, 'SMIL.pkl')
        if dataset_name == 'SyRIP':     # only SyRIP on depth
            if_ptc = False
        t_smil = torch.tensor([0.0, 0, -0.46], requires_grad=False).cuda()  # no grad
        s_smil = 2.75
        # if_cam_rct = True
    else:
        smpl_pth = config.SMPL_MODEL_DIR
        # if_cam_rct = False

    smpl_neutral = SMPL(smpl_pth,
                        create_transl=False).to(device)     # if infan tuse the SMIL model
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    renderer = PartRenderer()
    render_vis = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)
    # add the img render to gen images.  every several parts.
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # MPJPE and Reconstruction error for each action
    mpjpe_dict = {}
    recon_err_dict = {}
    for action in action_names:
        mpjpe_dict[action] = []
        recon_err_dict[action] = []

    # images name list
    imgnames_list = []

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # aligned depth error
    ptc_err = np.zeros(len(dataset))    # reconstrcuted
    depth_err = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    eval_depth = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp' or dataset_name == 'SLP' or dataset_name == 'MIMM':
        eval_pose = True    # eval pose these 3
        eval_depth = True
    elif dataset_name == 'lsp':     # lsp for masks and parts
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14 # from h36 17 ->  lsp +  pelvis, spine, neck(upper)
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14 # from smpl 24 ->  lsp neck , head(h36) pelvis, spine, 17 jaw
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        if iter>0 and step == iter:
            break
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        # print('get gt_pose', gt_pose)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices # with batch x jt, root centered no trans
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]
        imgs_RGB = batch['img_RGB'].to(device)
        depths_dn = batch['depth_dn'].to(device)
        gt_keypoints_3d = batch['pose_3d'].cuda()
        masks2 = batch['mask2']
        imgnames = batch['imgname']
        gt_2d = batch['keypoints']

        imgnames_list += imgnames
        # read in depth ,  pred depth, mask out, estimate the bias, update depth, ptc calculation , crop(bb) or (mask)
        
        with torch.no_grad():   # get pred
            pred_rotmat, pred_betas, pred_camera = model(images)  # pose, shape, camera(ortho projection), z, x, y ?
            if if_cam_rct:
                pred_camera += t_smil
                pred_camera[:, 0] *= s_smil  # 64 match 3 non singular dimension

            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)  # why 24 ? 24 joint each a 3x3
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72) # flatten
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()

        camera_translation_bch = torch.stack([pred_camera[:, 1], pred_camera[:, 2],2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],dim=-1)  # metric ,  pred_cam [-1, 1] res range ,  pred_camera [ z,  x, y]? last dim? , render flip the x, direction, neural render not change.
        idx_bs = step * batch_size  # idx of the smpl

        for i, pred_v in enumerate(pred_vertices):  # loop bch
            # if if_svImg:    # only the first in batch will be saved, otherwise loop it
            idx = idx_bs + i    # current bs
            img_RGB_t = imgs_RGB[i].cpu().numpy() # to image format
            img_RGB_t = np.transpose(img_RGB_t, (1,2,0))    # 0 ~1
            if if_ldImg:
                raise ValueError('not implemneted')
            else:
                img_rd = img_RGB_t
            mask2 = masks2[i]     # mask 2
            depth_dn_t = depths_dn[i].cpu().numpy().squeeze()     # remove leading channel
            # Calculate camera parameters for rendering
            camera_translation_t = camera_translation_bch[i].cpu().numpy()    # bch 1st
            pred_vertices0 = pred_v.cpu().numpy() # single sample
            img_shape, valid_mask, rend_depth = render_vis(pred_vertices0, camera_translation_t, img_rd) # 0 ~1
            valid_mask = valid_mask.squeeze()   # get rid of the  end 1 dim
            # Render side views
            aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]  # x, y ,z   right, up ,  outward
            center = pred_vertices0.mean(axis=0)
            rot_vertices = np.dot((pred_vertices0 - center), aroundy) + center  # rotate body
            mask_u = np.logical_and(mask2.cpu().numpy(), valid_mask)

            # if mask_u all false, then set 0 to error -1 filter out later,  d_dn_t not change
            if (~mask_u).all(): # if empty valid
                err_d_t = -1.       # specific label
            else:
                d_rend_valid = rend_depth[mask_u]   # rend depth value range
                d_dn_valid = depth_dn_t[mask_u]     # vec only
                # d_dn_mask = np.logical_and(depth_dn_t<2.11, depth_dn_t>0)  # 2.1m, no boundary only bed, sligh 0.01 margin to cover all bed, depth denoised
                depth_dn_t = depth_dn_t + (d_rend_valid.mean() - d_dn_valid.mean())  # the z direction, err: infant empty slice
                d_dn_valid = d_dn_valid + (d_rend_valid.mean() - d_dn_valid.mean())  # the z direction
                # print('idx', idx)
                # print('d_rend_valid shape', d_dn_valid.shape)
                err_d_t = np.absolute(d_rend_valid - d_dn_valid).mean() * 1000.
                # rend_depth[rend_depth==0] = rend_depth[mask_u].max() + 0.5  # 0.5 far away,
                # err infant empty, for max  assume no mask is there.

            # print("current d error", err_d_t)
            # print('rend_depth range', rend_depth.min(), rend_depth.max())
            # print("rend_depth upper corner", rend_depth[:10,:10])   # check corner area
            # print("depth_dn_t rg", depth_dn_t.min(), depth_dn_t.max())
            # print('mask type:', valid_mask.dtype)       # mask type bool



            depth_err[idx]= err_d_t # average error, to mm, broad cast 2224,224 ->  7053

            if svImg_freq>0 and idx % svImg_freq == 0:        # only greater than 0 save the image

                trans = camera_translation_t.copy()   # ptc, x,y z to world, x, -y , -z
                trans[1] *= -1  # y opposite direction.

                if if_ptc and err_d_t>0:
                    ptc = ut_t.get_ptc_mask(depth_dn_t, [constants.FOCAL_LENGTH, ] * 2,
                                            mask=mask_u)  # empty ptc give None  to not render
                    ptc[:, 1] *= -1  # y flipped? point up?
                    ptc = ptc - trans  # camera coordinate, to world
                    rot_ptc = np.dot((ptc - center), aroundy) + center
                else:
                    ptc = None
                    rot_ptc = None  # not render

                # get the ptc version front view
                img_shape, _, _ = render_vis(pred_vertices0, camera_translation_t, img_rd, ptc=None)  # 0 ~1 only if want to have ptc on front view
                img_shape_white, _, _ = render_vis(pred_vertices0, camera_translation_t, np.ones_like(img_rd), ptc=None)  # smpl without bg f

                img_shape_side_ptc, _, _ = render_vis(rot_vertices, camera_translation_t, np.ones_like(img_rd), ptc=rot_ptc)  # white bg
                img_shape_side, _, _ = render_vis(rot_vertices, camera_translation_t, np.ones_like(img_rd), ptc=None)  # white bg
                pth_img_front = osp.join(vid_fd, '{:05d}_f.jpg'.format(idx))
                cv2.imwrite(pth_img_front, 255 * img_shape_white[:, :, ::-1])
                pth_img_front_RGB = osp.join(vid_fd, '{:05d}_f_RGB.jpg'.format(idx))        # image with RGB
                cv2.imwrite(pth_img_front_RGB, 255 * img_shape[:, :, ::-1])
                pth_img_side = osp.join(vid_fd, '{:05d}_s.jpg'.format(idx))
                cv2.imwrite(pth_img_side, 255 * img_shape_side[:, :, ::-1]) # tuple indices must be slices? not tuple?
                pth_img_side_ptc = osp.join(vid_fd, '{:05d}_s_ptc.jpg'.format(idx))
                cv2.imwrite(pth_img_side_ptc, 255 * img_shape_side_ptc[:, :, ::-1]) # tuple indices must be slices? not tuple?
                # save th depth_dn, depth_rd , mask
                pth_img = osp.join(vid_fd, '{:05d}_d_rd.jpg'.format(idx))
                cv2.imwrite(pth_img, ut_t.normImg(rend_depth, if_toClr=False))
                pth_img = osp.join(vid_fd, '{:05d}_d_dn.jpg'.format(idx))
                cv2.imwrite(pth_img, ut_t.normImg(depth_dn_t, if_toClr=False))
                pth_img = osp.join(vid_fd, '{:05d}_mask.jpg'.format(idx))
                cv2.imwrite(pth_img, ut_t.normImg(valid_mask.astype(np.uint8), if_toClr=False))

            # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)  # batches, reg to h36m
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name or 'SLP' in dataset_name or 'MIMM' in dataset_name:    # use the pseudo depth
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]      # j24  to 17 , mpi-inf-3dp  to 14,  first 13 same as  lsp + head(h36) pelvis,spine, jaw
            # For 3DPW get the 14 common joints from the rendered shape, root centered
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)  # regressor to h36m  17?  use the h36m version, all the N x 6950 6950*17 -> N  get the mat regressor
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]  # h36m to lsp + format
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)  # all to h36m joints
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]      # h36m-17 - > lsp+17
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis     # centered, after alignment,

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() # sum x,y,z, then mean over jts
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)  # Procrustes error
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # MPJPE to actions, only for h36m
            if 'h36m' in dataset_name: # only for h36m, can switch here to different ds
                for img_idx in range(len(imgnames_list)):
                    curr_name = imgnames_list[img_idx]
                    tmp_name = curr_name.split('/')[-1]
                    curr_action, _, _ = tmp_name.split('.')
                    curr_action = curr_action.split('_')[1]

                    mpjpe_dict[curr_action].append(mpjpe[img_idx])
                    recon_err_dict[curr_action].append(recon_err[img_idx])

        # If mask or part evaluation, render the mask and part images
        # if eval_masks or eval_parts:
        #     mask, parts = renderer(pred_vertices, pred_camera)
        mask, parts, _ = renderer(pred_vertices, pred_camera)      # try if render part works, too many to unpack?
        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))
        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print()
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()
            if eval_depth:
                depth_err_eval = depth_err[:step * batch_size]
                print('Aligned depth error: ', depth_err_eval[depth_err_eval != -1].mean())     # not equal to -1
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print()
        metric['MPJPE'] = 1000 * mpjpe.mean()
        metric['MPJPE_PA'] = 1000 * recon_err.mean()
        for act in action_names:
            if mpjpe_dict[act] == []:
                print(act + '_MPJPE: ' + str(0.0))
                print(act + '_Reconstruction Error: ' + str(0.0))
                print()
            else:
                print(act + '_MPJPE: ' + str(1000 * sum(mpjpe_dict[act]) / len(mpjpe_dict[act])))
                print(act + '_Reconstruction Error: ' + str(1000 * sum(recon_err_dict[act]) / len(recon_err_dict[act])))
                print()
        metric['MPJPE_act'] = mpjpe_dict    # action dictionary
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()
    if eval_depth:
        print('Aligned depth error: ', depth_err.mean())
        print()
        metric['err_depth'] = depth_err.mean()
        # for each individual cases, check the errors
        metric['err_depth_cat'] = [depth_err[::3].mean(), depth_err[1::3].mean(), depth_err[2::3].mean()]   # save error for each cover. only work for full cover cases.

    with open(metric_file,'w') as f:        # save the result.
        json.dump(metric,f)
        print('metric result saved to {}'.format(metric_file))
        f.close()


if __name__ == '__main__':
    # args = parser.parse_args()
    args = TrainOptions().parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        print('load ckpt from {}'.format(checkpoint))
    else:   # read from the exp folder
        ckpt_fd = osp.join('logs', args.name, 'checkpoints')
        li_pt = glob(osp.join(ckpt_fd, "*.pt"))
        ckpt_pth = sorted(li_pt)[-1]   # the latest ckpt
        checkpoint = torch.load(ckpt_pth)
        print('load ckpt from {}'.format(ckpt_pth))

    out_fd = osp.join('logs', args.name)
    # result_file = osp.join('logs', args.name, 'test_rst.npz')

    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    if args.eval_ds == 'SLP':
        args.SLP_fd = os.path.join(args.ds_fd, 'SLP', args.SLP_set)  # SLP folder [danaLab|simLab]
        args.phase='test'
        ds_SLP = SLP_RD(args, phase='test', rt_hn=args.ratio_head_neck)
        # print(ds_SLP.pthDesc_li[:10])       # why from sub 1
        dataset = SPIN_ADP(args, 'SLP', ds_SLP, is_train=False)
    else:
        dataset = BaseDataset(args, args.eval_ds, is_train=False) # new h36m-p1

    # Run evaluation
    if True:        # save every 15
        # args.svImg_freq = -1        # not save
        # args.testIter = 1
        print('if_cam_rct', args.if_cam_rct)
        run_evaluation(model, args.eval_ds, dataset, out_fd = out_fd,
                       batch_size=args.batch_size,
                       shuffle=args.shuffle_test,
                       log_freq=args.log_freq, iter=args.testIter, svImg_freq=args.svImg_freq, if_cam_rct= args.if_cam_rct)  # default not shuffle
