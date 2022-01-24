import os
import json
import argparse
import numpy as np
from collections import namedtuple
import os.path as osp

class TrainOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', default='SLP_demos', help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained checkpoint at the beginning training')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=30, help='Total number of training epochs')
        train.add_argument('--epoch_step', type=int, default=12, help='The epoch number of each train.')
        train.add_argument('--trainIter', type=int, default=-1, help='control the train epoch iteration')

        train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=64, help='Batch size') # ori batch 64
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--test_steps', type=int, default=1000, help='Testing frequency during training')
        train.add_argument('--checkpoint_steps', type=int, default=-1, help='Checkpoint saving frequency, -1 not save steps.')   # ori 10000
        train.add_argument('--checkpoint_epoch_steps', type=int, default=10, help='Checkpoint saving frequency')
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network')
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]') 
        train.add_argument('--noise_factor', type=float, default=0.4, help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]') 
        train.add_argument('--scale_factor', type=float, default=0.25, help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]') 
        train.add_argument('--ignore_3d', default=False, action='store_true', help='Ignore GT 3D data (for unpaired experiments') 
        train.add_argument('--shape_loss_weight', default=0, type=float, help='Weight of per-vertex loss') 
        train.add_argument('--keypoint_loss_weight', default=5., type=float, help='Weight of 2D and 3D keypoint loss') 
        train.add_argument('--keypoint_loss_weight_3d', default=1., type=float, help='Weight of 3D keypoint loss')
        train.add_argument('--pose_loss_weight', default=0.1, type=float, help='Weight of SMPL pose loss')  # original 1.
        train.add_argument('--beta_loss_weight', default=0.001, type=float, help='Weight of SMPL betas loss') 
        train.add_argument('--openpose_train_weight', type=float, default=1., help='Weight for OpenPose keypoints during training')     # open openpose wt
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training') 
        train.add_argument('--depth_weight', type=float, default=1., help='Weight for GT keypoints during training')
        train.add_argument('--cam_weight', type=float, default=0., help='Weight of the camera loss. if not use, gives 0')
        train.add_argument('--run_smplify', default=False, action='store_true', help='Run SMPLify during training')     # in loop or not
        train.add_argument('--smplify_threshold', type=float, default=100., help='Threshold for ignoring SMPLify fits during training') 
        train.add_argument('--num_smplify_iters', default=100, type=int, help='Number of SMPLify iterations') 

        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)

        # SLP part
        train.add_argument('--train_ds', default='SLP', help='the training dataset, [SLP|h36m|h36m_2d|infant]')   # SLP or h36m | h36m_2d h36m only h36m h36m_2d add mpii, infant,h36m_2d combine 2d and 3d, not ptc render in train
        train.add_argument('--li_mod_SLP', nargs='+', default=['RGB'], help='the mods for training of SLP, [RGB, depth, IR, PM]' )
        train.add_argument('--cov_li', nargs='+', default=['uncover'], help='the cover conditions')
        train.add_argument('--sz_pch', default=224, help='patch size for SLP')
        train.add_argument('--SLP_set', default='danaLab')
        train.add_argument('--ds_fd', default='/scratch/liu.shu/datasets')
        train.add_argument('--fc_depth', default=50., help='the depth normalization factor to pixel')
        train.add_argument('--if_depth', type=bool, default=False, help='if supervise depth')
        train.add_argument('--if_vis', type=bool, default=False, help='if use vis infor toherwise all visible') # from fundamental 2d only
        train.add_argument('--if_cam_rct', type=bool, default=False, help='if rectify the camera output for infant ft.') # from fundamental 2d only
        train.add_argument('--epoch_dp', type=int, default=-1, help='when the depth come into effect, 3d stop. ')
        train.add_argument('--epoch_cam', type=int, default=-1, help='when the cam loss stop effect, -1 then never stop')
        train.add_argument('--ratio_head_neck', type=float, default=0.8, help='ratio of the head to neck.') # first round 0.7 version try 0.8
        train.add_argument('--opt_wt_pose', type=float, default=4.78, help='ratio of the head to neck. for optimization loop, already good') # ori 4.78
        train.add_argument('--opt_wt_pose_dRt', type=float, default=0.85, help='decrease ratio of the opt_wt, -1 for not decrease, otherwise opt_wt times it each time.') # ori 4.78
        train.add_argument('--opt_wt_shp', type=float, default=5., help='ratio of the head to neck.')

        # test part
        train.add_argument('--eval_ds', default='SLP',
                            choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp', 'SLP', 'MIMM','SyRIP'], help='Choose evaluation dataset')
        train.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
        train.add_argument('--shuffle_test', default=False, action='store_true', help='Shuffle data')
        train.add_argument('--testIter', type=int, default=-1, help='Shuffle data')
        train.add_argument('--svImg_freq', type=int, default=3, help='save image frequency in test session. smaller than 0 not save.')

        return 

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        # add SLP fd
        self.args.SLP_fd = osp.join(self.args.ds_fd, 'SLP', self.args.SLP_set)
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)      #  a named turple object called  json_args
                return json_args    # only take the complete ones, cost time actually
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')   # logs/exp/checkpoints/ date.ckpt
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
