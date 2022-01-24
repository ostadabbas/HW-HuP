from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from utils.utils import plot_image_frame

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        # dataset: ds name
        super(BaseDataset, self).__init__() # bound obj
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]      # img dir?, give SLP_danaLab, image dir + image_name
        self.depth_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])    # npz dict array, ds folder, ds file
        print('loading from', config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']
        
        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)      # smpl pose
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']       # if there is S key there is 3D infor
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        # Get proxy 3D pose, if available
        try:
            self.pose_3d_dp = self.data['S_dp']  # if there is S_dp key there is 3D infor
            self.has_pose_3d_dp = 1
        except KeyError:
            self.has_pose_3d_dp = 0
        if ignore_3d:
            self.has_pose_3d_dp = 0

        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))

        # if dataset == 'h36m' and not if_h36m_2d:  # follow original use 2d
        #     keypoints_gt[:,:,2] = 0
        #     print('ban h36m 2d')

        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        # get the has_depth
        self.has_depth = 1
        if not options.if_depth and self.is_train:
            self.has_depth = 0  # ignore the depth part

        try:
            self.depthname = self.data['depthname']
        except KeyError:
            self.has_depth = 0

        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor],  normalize human size
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn=None):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        # rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        # rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        # rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        if pn is not None:      # will only affect RGB as depth no pn
            for i in range(rgb_img.shape[2]):  # int not subscrible
                rgb_img[:, :, i] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, i] * pn[i]))
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1. # -1 to 1
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])  # simplified mat * vec to indicate the product order
        # flip the x coordinates,  rotation only, so should be centered 3d
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()  # per channel noise
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print('can not read image {}'.format(imgname))
            # print(imgname)
        orig_shape = np.array(img.shape)[:2]    # iamge ref before assignment

        # Load depth
        if self.has_depth:
            try:
                depthname = join(self.depth_dir, self.depthname[index])
                depth_dn = np.load(depthname).copy().astype(np.float32)
            except TypeError:
                print('can not read depth {}'.format(depthname))
                self.has_depth = 0
                # print(depthname)

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn) # img(255) -> cxhxw 0~1
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['img_RGB'] = img
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Process depth
        if self.has_depth:
            # get jt_dp_rng , get lower max , resize will softer
            # set 0 point to lower max
            depth_dn[depth_dn == 0] = depth_dn.max()
            depth_dn = depth_dn[..., None] / 1000.0  # add last channel, to m
            # get pelvis pose , get root depth
            keypoints = self.keypoints[index].copy()
            # kp_pelvis = (keypoints[27]+ keypoints[28])/2.
            kp_pelvis = keypoints[39]       # pelvis point
            depth_pelvis = depth_dn[int(kp_pelvis[1]+0.5), int(kp_pelvis[0]+0.5), 0]    # in m

            # create mask for depth data
            # original  0.5 , 0.2  too wide only 5 cm
            margin_max = 0.05  # the depth range into
            margin_min = 0.05  # depth out, the head part
            S = self.pose_3d[index].copy()  # only 24

            ## get the range
            # from the j3d
            # depth_jt_valid = S[S[:, 3] > 0][:, 2]
            # rng = np.array([depth_jt_valid.min() - margin_min, depth_jt_valid.max() + margin_max]) + depth_pelvis
            # from depth_dn + 2d, get all valid 2d,  loop get all depth , then min max
            kps_valid = keypoints[25:37]
            kps_valid = kps_valid[kps_valid[:, 2]>0]       # only 1st 2 and valid
            # keypoints_valid = keypoints[S[:, 3]>0]       # all
            li_dp_jt = []
            for kp in kps_valid[:,:2]:
                kp = (kp+0.5).astype(int)
                li_dp_jt.append(depth_dn[kp[1], kp[0], 0])
            dp_jts = np.array(li_dp_jt)
            # print('dp_jt is', dp_jts)
            rng = [dp_jts.min()-margin_min, dp_jts.max()+margin_max]
            rng = np.array(rng)
            depth_dn = self.rgb_processing(depth_dn * 255.0, center, sc * scale, rot, flip,
                                           pn=None)  # keep depth in meter for later, channel first

            # Store depth to use it in visualization
            item['depth_dn'] = depth_dn.astype(np.float32)
            item['depthname'] = depthname
            clone_dp = np.copy(depth_dn)
            clone_dp = np.squeeze(clone_dp)  # to squeeze for show, channel 2 
            # mask = np.zeros_like(depth_dn)
            # mask[depth_dn>rg[0] and depth_dn<rg[1]] = 1 # set to the
            mask = np.logical_and(clone_dp > rng[0], clone_dp<rng[1])

            if 0:       # for debug
                print('depth range', clone_dp.min(), clone_dp.max())
                # print('jt min max', depth_jt_valid.min(), depth_jt_valid.max())
                # print("pelvis depth", depth_pelvis)
                print("seg range", rng)

                print('the dn is')
                plot_image_frame([np.squeeze(clone_dp)])
                print('the mask is')
                plot_image_frame([mask])

            # original only set the max value to 0
            # clone_dp = np.copy(depth_dn)
            # clone_dp = np.squeeze(clone_dp)
            # mask = np.ones(clone_dp.shape)
            # mask[clone_dp == np.amax(clone_dp)] = 0     # largest value point is 0
            # bb mask1
            # get jt_d_rng, then depth_dn gen mask2
            # intersection
            item['mask2'] = mask.astype(np.float32)     # no channel
            # print(depthname)

        else:
            item['depth_dn'] = np.zeros((1, img.shape[1], img.shape[2])).astype(np.float32)
            item['depthname'] = ''
            item['mask2'] = np.zeros((img.shape[1], img.shape[2])).astype(np.float32)


        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            # item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
            S_dp = self.pose_3d_dp[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
            item['pose_3d_dp'] = torch.from_numpy(
                self.j3d_processing(S_dp, rot, flip)).float()  # for depth proxy , SLP gt and dp the same.
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)
            item['pose_3d_dp'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()  # 49 openpose + gt

        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['has_depth'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        # depth_dn
        # pose_3d_dp         depth3d
        # check depth pure depth raw .
        # if d3d not correct,

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)
