import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from datasets import MixedDataset
from models import hmr, SMPL
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from utils.renderer import Renderer
from utils import BaseTrainer
from datasets import SLP_RD, SPIN_ADP
from utils.part_utils import PartRenderer
from utils import CheckpointDataLoader, CheckpointSaver
import os.path as osp
from tqdm import tqdm
import utils.utils as ut_t

import config
import constants
from .fits_dict import FitsDict


class Trainer(BaseTrainer):
    
    def init_fn(self, phase='train'):
        # phase only used to save the dictionaries, gives different phase test or train
        opt = self.options
        if self.options.train_ds =='SLP':   # if to h36m
            self.options.SLP_fd = osp.join(opt.ds_fd, 'SLP', opt.SLP_set)
            self.SLP_rd = SLP_RD(self.options, phase=phase, rt_hn=opt.ratio_head_neck)        # choose the correct ratio

            self.train_ds = SPIN_ADP(self.options, 'SLP', self.SLP_rd, ignore_3d=self.options.ignore_3d, is_train=(phase =='train'))
        else:  # h36m
             self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)        # to change to target


        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        if opt.train_ds == 'infant':
            smpl_pth = osp.join(config.SMPL_MODEL_DIR, 'SMIL.pkl')
        else:
            smpl_pth = config.SMPL_MODEL_DIR
        self.smpl = SMPL(smpl_pth,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_cam = nn.L1Loss().to(self.device)    # quick to device
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        # Initialize SMPLify fitting module
        self.smplify = SMPLify(step_size=1e-2, batch_size=self.options.batch_size, num_iters=self.options.num_smplify_iters, focal_length=self.focal_length, wt_pose=opt.opt_wt_pose, wt_shp=opt.opt_wt_shp, smpl_pth=smpl_pth)
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)  # if there is fit in folder , it will reads, keep all fit dict in an instance, save only timeout

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

        # SLP parameters
        self.if_dp = False  # dp and 3d flip with each other


        # cam rct
        # self.t_smil = torch.tensor([0.05, 0, -0.46], requires_grad=False).cuda()    # no grad
        self.t_smil = torch.tensor([0.0, 0, -0.46], requires_grad=False).cuda()    # no grad
        self.s_smil = 2.75
        self.if_cam = True  # dp and 3d flip with each other cam is working at the begining

    def finalize(self):
        self.fits_dict.save()   # will save to ds_name.npy

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight, if_op_hip=True):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        # set the hip openpose hip conf as original
        # conf_ori = conf.copy()
        conf[:, :25] *= openpose_weight     #  if op good gives weight for 9, 12
        # update the op weight , r hip l hip 9, 12 for op updating
        # conf[:, 9] = conf_ori[:, 9]
        # conf[:, 12] = conf_ori[:, 12]
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, if_op_hip=True):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        # map pred op_hip to hip
        if if_op_hip:   # use openpose hip,  op rlh,  9, 12  r,l  , map op hip to pred
            pred_keypoints_3d[:, 27, :] = pred_keypoints_3d[:, 9, :]        # op to ori
            pred_keypoints_3d[:, 28, :] = pred_keypoints_3d[:, 12, :]
            pred_keypoints_3d[:, 39, :] = pred_keypoints_3d[:, 8, :]       # the hip
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]        #  first 25 is from op. can update
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()      # last conf, n_bchx 24 x 1
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1] # N x n_jt x 4 ? select the 3d part
        conf = conf[has_pose_3d == 1]   # if not if_vis conf all 1
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1] # select the has 3d one
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def depth_loss(self, dp_pred, dp_gt, mask, has_depth):
        '''
        from mask out the area of the pred and
        :param dp_pred:
        :param dp_gt:
        :param maks: should be bool for indexing
        :param has_depth:
        :return:
        '''
        mask = mask[has_depth==1]   # index should be correct

        # debug
        # dp_pred_t = ut_t.normImg(dp_pred[0].detach().cpu().numpy(), if_toClr=False)
        # dp_pred_t = dp_pred[0].detach().cpu().numpy()
        # dp_gt_t = dp_gt[0].detach().cpu().numpy()
        # mask_t = mask[0].detach().cpu().numpy().astype(np.uint8)
        # # cv2.normalize(img, normalizedImg, 0, 255, cv.NORM_MINMAX)
        #
        # print('test 1st in the loop')
        # # print('mask sum', mask[0].sum())  # how many ones there
        # print('dp_pred_t min max', dp_pred_t.min(), dp_pred_t.max())
        # print('dp_gt_t min max', dp_gt_t.min(), dp_gt_t.max())
        # print('after masking')
        # print('dp_pred_t min max', dp_pred_t[mask_t==1].min(), dp_pred_t[mask_t==1].max())
        # print('dp_gt_t min max', dp_gt_t[mask_t==1].min(), dp_gt_t[mask_t==1].max())
        #
        # cv2.imshow('dp pred', dp_pred_t)
        # cv2.imshow('dp gt', dp_gt_t)
        # cv2.imshow('mask ', mask_t.astype(np.uint8)*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        dp_pred = dp_pred[has_depth == 1][mask==1]  # two mased out vec  values  n_bch x l, if nothing masked out over image and patch
        dp_gt = dp_gt[has_depth == 1][mask==1]

        # print('dp_pred shape after', dp_pred.shape) # should be indexed in to a long list like , 615680 totally flattened.
        # diff_vec = (dp_pred - dp_gt)[mask]
        criterion = nn.MSELoss().to(self.device)
        # criterion = nn.L1Loss().to(self.device)         #  change later for different d
        if len(dp_pred) > 0:        # if there is masked area
            return criterion(dp_pred, dp_gt)       # no target?
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:  # if there is smpl loss ,else return 0
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        # gt_joints = input_batch['pose_3d'] # 3D pose
        gt_joints = input_batch['pose_3d_dp'] # 3D pose
        has_smpl = input_batch['has_smpl'].byte() # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte() # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset, index in the ds
        mask2 = input_batch['mask2']  # mask of depth data
        batch_size = indices.shape[0]


        # SLP additional
        has_depth = input_batch['has_depth']
        try:
            depth_dn = input_batch['depth_dn']
        except AttributeError:
            img_shp = images.shape  # the batch shape
            img_shp[1] = 1      # single ch
            depth_dn = torch.zeros(img_shp)     # all 0
        depth_dn = depth_dn.squeeze()   # get rid of the channel dim
        renderer_pt = PartRenderer()


        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])    # may be 0
        gt_model_joints = gt_out.joints     # model estimated, could be neutral one if not exists
        gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]  # only 4590, an instance keep the fitting parameters
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints


        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)        # 2d is [-1, 1 ]?

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss, may be singular value
        # gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)
        #

        try:
            gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length,
                                            img_size=self.options.img_res)
        except:
            name = input_batch['depthname']
            np.savez('err_data.npz', depthname=name, model_joints=gt_model_joints.cpu().numpy(),
                     keypoints_2d_orig=gt_keypoints_2d_orig.cpu().numpy())
            print('all invisible 2d joints, skip iter')
            return None, None

        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length,
                                         img_size=self.options.img_res)

        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,
                                                       0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                                       gt_keypoints_2d_orig).mean(dim=-1)

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)   # pred_cam [ z', x, y]
        if self.options.if_cam_rct:     # rectify updating
            pred_camera += self.t_smil
            pred_camera[:,0] *= self.s_smil     # 64 match 3 non singular dimension

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints        # 49 joints?

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)], dim=-1)
        # x,y trans fix,  z is 2f/res * s so s is res dependant , so normalize by half res, higher res lower z will always project to z

        camera_center = torch.zeros(batch_size, 2, device=self.device)  # center 0
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.) # should be -1 to 1?

        if self.options.run_smplify:

            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1) # 3 x 4 to homo
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            # Run SMPLify optimization starting from the network prediction, update pose beta joint
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                                        pred_pose.detach(), pred_betas.detach(),
                                        pred_cam_t.detach(),
                                        0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                        gt_keypoints_2d_orig)
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)    # simplify only 2d

            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)  # if it is better
            # print('new opt_joint_loss', opt_joint_loss) # original suppose to be large
            # print('new loss', new_opt_joint_loss)

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]     # update


            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())  # update only better , update all in fits dict
        else:
            update = torch.zeros(batch_size, device=self.device).byte()

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters, if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]      # only update the
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]      # update with gt , opt_cam_t best
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl

        opt_keypoints_2d = perspective_projection(opt_joints,
                                                  rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                  translation=opt_cam_t,
                                                  focal_length=self.focal_length,
                                                  camera_center=camera_center)


        opt_keypoints_2d = opt_keypoints_2d / (self.options.img_res / 2.)   # [-1, 1]


        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # Compute 3D keypoint loss, add vis loss ----
        if self.if_dp:      # update the depth
            # print('set the 3d loss to 0')
            loss_keypoints_3d = torch.FloatTensor(1).fill_(0.).to(self.device)  # 0 3d loss
        else:
            loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)      # joints 3d directly supervision. has confidence n_bchx4, h36m all conf to 1

        # depth sup,  with  updated render  beta and pose to render, 2d+depth (misaligned due to the flipping, 2d+3d_dp,  2d+3d_dp + depth (full), 2d+3d_dp + (vis filtering)
        # align the dp_dn
        if self.if_dp:
            # print('before part render')
            mask, parts, dp_rd = renderer_pt(pred_vertices, pred_camera.detach())        # original detached, meter
            # print('part render done')
            mask_u = mask * mask2.float()         # torch and  gramma
            dp_rd = dp_rd     # to meter
            # d0 = dp_rd.mean(dim=[1,2], keepdim=True) - depth_dn.mean(dim=[1, 2], keepdim=True)
            d0 = ((dp_rd - depth_dn)* mask_u).sum(dim=[1,2], keepdim=True)/ mask_u.sum(dim=[1,2], keepdim=True)
            depth_dn_r = depth_dn + d0.detach()    # update to render similar level @ dim2  a224 b64
            # print('depth dn updated')
            # check the range of the depth_dn

            loss_depth = self.depth_loss(dp_rd, depth_dn_r, mask_u.detach(), has_depth)  # n_bch xh x w , detach to not affec teh mask  bool() not workiing for 1.1
            # print('loss_depth done')
        else:
            loss_depth = torch.FloatTensor(1).fill_(0.).to(self.device)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)    # 0 not has_smpl
        if self.if_cam:
            loss_cam = self.criterion_cam(pred_cam_t, opt_cam_t)
        else:
            loss_cam = torch.FloatTensor(1).fill_(0.).to(self.device)  # 0 3d loss
        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape +\
               self.options.keypoint_loss_weight * loss_keypoints +\
               self.options.keypoint_loss_weight_3d * loss_keypoints_3d +\
               self.options.pose_loss_weight * loss_regr_pose +\
               self.options.beta_loss_weight * loss_regr_betas +\
               self.options.depth_weight * loss_depth +\
               self.options.cam_weight * loss_cam +\
               ((torch.exp(-pred_camera[:,0]*10)) **2).mean()     # e^(-10*d_cam)^2  so never negative distance.
        loss *= 60


        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'opt_vertices': opt_vertices,     # no detach? no in graph?
                  'pred_cam_t': pred_cam_t.detach(),    # model output transform to 1,2,0 order (original seems to be z , x, y)
                  'opt_cam_t': opt_cam_t}
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item(),
                  'loss_depth': loss_depth.detach().item(),
                  'loss_cam': loss_cam.detach().item(),
                  }

        return output, losses

    def get_static_dict(self):
        '''
        get the static fitting dict with provided model , one epoch only
        :return:
        '''
        train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                 batch_size=self.options.batch_size,
                                                 num_workers=self.options.num_workers,
                                                 pin_memory=self.options.pin_memory,
                                                 shuffle=False)
        # Iterate over all batches in an epoch
        trainIter = self.options.trainIter
        for step, batch in enumerate(tqdm(train_data_loader, desc='loop the ds, save out the fits_dict...',
                                          total=len(self.train_ds) // self.options.batch_size,
                                          initial=train_data_loader.checkpoint_batch_idx),
                                     train_data_loader.checkpoint_batch_idx):  # enumerate(value, start)
            if trainIter > 0 and step >= trainIter:  # stop the train iteration
                break
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = self.train_step(batch)  # output loss
            self.step_count += 1
            # get pose and beta , update the list
            # Tensorboard logging every summary_steps steps

        self.finalize()     # save to exp_fd/dsNm_fits.npy



    def train_summaries(self, input_batch, output, losses, if_rend_ptc=True):
        images = input_batch['img']
        if if_rend_ptc:
            depth_dn = input_batch['depth_dn']
        else:
            depth_dn = None
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)    # unnormalization
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        mask2 = input_batch['mask2']
        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images, depth_ts=depth_dn, mask=mask2)
        # images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images, depth_ts=None)
        images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images, depth_ts=depth_dn, mask=mask2)
        # images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images, depth_ts=None)
        # print('imges_pred range', images_pred.min(), images_pred.max(), images_pred.shape)  #
        # print('imges_opt range', images_opt.min(), images_opt.max(), images_opt.shape)
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            # print('write loss {} : {} at step {}'.format(loss_name, val, self.step_count))
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

    def update_state(self, flg_dp=0, flg_cam=1):
        ''' update the traning state
        only set 1 or stop, no recover operation.
        '''
        if flg_dp == 1:       # stop 3d add dp
            # print('switch to the dp mode, ban 3d.')
            self.if_dp = True   # begin to train dp  end 3d
            # print('if_dp set to True')
        # else:
        #     print("nothing happend")
        if flg_cam == 0:
            self.if_cam = False
        # else:
        #     self.if_cam = True
