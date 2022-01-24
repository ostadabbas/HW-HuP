'''
Adapted other ds for base_dataset_dp depth proxy method
all to compatible  data format, interface function:
get_img_cb,,  RGB aligned, stacked, normalized.
get_depthRaw, RGB aligned, depth raw
mainly to cope with the SLP which needs alignment. Otherwise, can simply save all aligned one to share common interface
'''

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Normalize
# from .base_dataset import BaseDataset
# inherent init will call the parent method and data is missing for it
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import constants
from os.path import join
import cv2
from tqdm import tqdm

class SPIN_ADP(Dataset):
	'''
	for the SLP adaptation work, other type not implemented yet
	'''
	def __init__(self, options, dataset, ds, ignore_3d=False, use_augmentation=True, is_train=True):
		'''
		:param dataset: the dataset nam
		:param ds:  the ds_reader
		'''
		super(SPIN_ADP, self).__init__()    # paranet Dataset
		self.dsNm = dataset + '_' + options.SLP_set # SLP_danaLab
		self.ds = ds
		self.options = options
		self.is_train = is_train
		self.img_dir = ds.dsFd      # ds folder
		assert dataset in ['SLP'], '{} not implemented yet'.format(dataset)
		if 'SLP' == dataset:
			self.mods_avail = ['RGB', 'depth', 'IR', 'PM']
		else:
			self.mods_avail = ['RGB']        # most ds
		self.n_cov = len(options.cov_li)        # how many cover conditions
		self.idxs_jt = [8, 9, 12, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43] # the jt used include std24 + openpose  hips
		self.li_mod_SLP = options.li_mod_SLP  # the SLP mod list
		means = []
		stds = []
		for mod in options.li_mod_SLP:      # add together one by one
			means = means+ ds.means[mod]
			stds = stds + ds.stds[mod]
		self.normalize_img = Normalize(mean=means, std=stds)    # only input, output depth_dn should be original
		# self.data = self.get_data() #  update the self.data and imgname (list
		self.dataset_dict = {'SLP_'+ options.SLP_set: 0}       # for fitting purpose  'SLP_danaLab_fits.npz'  72 pose + 10 beta
		self.data = self.ds.db_SPIN #  update the self.data and imgname (list
		# make the self. data structure similar to facilitate the next step
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
			self.pose = self.data['pose'].astype(np.float)  # smpl pose
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
			self.pose_3d = self.data['S']  # if there is S key there is 3D infor
			self.has_pose_3d = 1
		except KeyError:
			self.has_pose_3d = 0
		if ignore_3d:
			self.has_pose_3d = 0

		# Get 2D keypoints
		try:
			keypoints_gt = self.data['part']
		except KeyError:
			keypoints_gt = np.zeros((len(self.imgname), 24, 3))
		try:
			keypoints_openpose = self.data['openpose']
		except KeyError:
			keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
		# update the hip to the op hips, open pose needs to
		keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)
		# update op hip, clean original visibility, shift to op
		keypoints[:, 9, :] = keypoints[:, 27, :]
		keypoints[:, 12, :] = keypoints[:, 28, :]
		keypoints[:, 8, :] = keypoints[:, 39, :]        # mid hip
		keypoints[:, 27, 2] = 0 # clean up
		keypoints[:, 28, 2] = 0
		keypoints[:, 39, 2] = 0     # clean hip 14+25 = 39 , no hip
		# print('kp 0 is', keypoints[0])

		self.keypoints = keypoints      # whole ds


		# Get gender data, if available
		try:
			gender = self.data['gender']    # 0 male, 1 female,  -1 neutral
			self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
		except KeyError:
			self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

		# get the has_depth
		self.has_depth = 1
		if not options.if_depth:
			self.has_depth = 0   # ignore the depth part

		# self.length = self.scale.shape[0]
		# self.length = len(self.scale)
		self.length = ds.n_smpl

	def get_data(self):
		'''
		from the given ds get the SPIN compatible data. when get image, use our own reading strategy, no need to read more, pay attention to the RGB channel order.
		feed in needed, also check if ptc is needed ? no align depth ,then simply recover with virtual cam
		:param dsNm:
		:param ds:
		:return:
		'''
		imgnames_, scales_, centers_, parts_, Ss_ = [], [], [], [], []      # parts, 2d, Ss 3d
		vis_ = []       # the visibility
		ds = self.ds
		ratio_head_neck = 0.9
		if self.dsNm == 'SLP':
			# get imgnames,  parts, centers, scales,  SS (3d) ,  scale = 1.2 bb/200, how large comparte to 200
			for i in tqdm(range(self.ds.n_smpl), desc='processing SLP ADP db'):
				arr, jt, bb = ds.get_array_joints(i, mod='RGB') # sq true
				j3d_dp = ds.get_j3d_dp(i)
				center = [(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2]  # should be pixel
				scale = 0.9 * max(bb[2] - bb[0], bb[3] - bb[1]) / 200.
				part = np.zeros([24, 3])        # for 2d part
				# first 14 lsp, but change 13(head t) to 18 for h36m, supervise 0:12, 18, then dp also update these , 14 to pelvis
				# add pelvis in middle
				part14 = jt[:, :2]  # the i
				part[:14] = np.hstack([part14, np.ones([14, 1])])
				part[13,2] = 0      # remove this no ht
				part[14, :2] = (part14[2] +part14[3])/2.    # pelvis
				part[14, 2] = 1
				part[18, :2] = part14[12] + ratio_head_neck * (part14[13] - part14[12])
				part[18, 2] = 1
				vis = 1 - jt[:,2]   # vis or not

				S24 = np.zeros([24, 4])
				jt_dp_3d = ds.get_j3d_dp(i)/1000. # from mm to m
				jt_dp_3d[:,:3] -= jt_dp_3d[14, :3]  # pelvis centered
				S24[:13, :3] = jt_dp_3d[:13, :3]
				S24[:13, 3] = 1     # first 13 good
				# add 14 pelvis,  18  head
				S24[14] = jt_dp_3d[14]   # pelvis
				S24[14, 3] = 1
				S24[18] = jt_dp_3d[13]      # head
				S24[18] = 1

				# update the rst list
				imgnames_.append(self.ds.get_img_pth(i))
				centers_.append(center)
				scales_.append(scale)
				parts_.append(part)         # 2d
				Ss_.append(S24)  # rooot cen
				vis_.append(vis)

			data = {
				'imgname': imgnames_,       # RGB only
				'center': np.array(centers_),
				'scale': scales_,
				'part': parts_,
				'S': Ss_,
				'vis': vis_
			}

		else:
			print('{} interface not implemented'.format(self.dsNm))
			exit(-1)

		return data

	def augm_params(self):
		"""Get augmentation parameters."""
		flip = 0  # flipping
		pn = np.ones(3)  # per channel pixel-noise
		rot = 0  # rotation
		sc = 1  # scaling
		if self.is_train:
			# We flip with probability 1/2
			if np.random.uniform() <= 0.5:
				flip = 1

			# Each channel is multiplied with a number
			# in the area [1-opt.noiseFactor,1+opt.noiseFactor]
			pn = np.random.uniform(1 - self.options.noise_factor, 1 + self.options.noise_factor, 3)

			# The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
			rot = min(2 * self.options.rot_factor,
			          max(-2 * self.options.rot_factor, np.random.randn() * self.options.rot_factor))

			# The scale is multiplied with a number
			# in the area [1-scaleFactor,1+scaleFactor],  normalize human size
			sc = min(1 + self.options.scale_factor,
			         max(1 - self.options.scale_factor, np.random.randn() * self.options.scale_factor + 1))
			# but it is zero with probability 3/5
			if np.random.uniform() <= 0.6:
				rot = 0

		return flip, pn, rot, sc

	def rgb_processing(self, rgb_img, center, scale, rot, flip, pn=None):
		"""Process rgb image and do augmentation. make it compatible with random channel dp can be large enough, make channel first later"""
		# if no pn then not change channel
		rgb_img = crop(rgb_img, center, scale,
		               [constants.IMG_RES, constants.IMG_RES], rot=rot)
		# flip the image
		if flip:
			rgb_img = flip_img(rgb_img)
		# in the rgb image we add pixel noise in a channel-wise manner
		if pn is not None:
			for i in range(rgb_img.shape[2]):        # int not subscrible
				rgb_img[:, :, i] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, i] * pn[i]))
		# (3,224,224),float,[0,1]
		# rgb_img = rgb_img.astype('float32')/255.0
		rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
		# if rgb_img.ndim>1:  # not touching single channel version
		# 	rgb_img = np.transpose(rgb_img, (2,0,1))

		return rgb_img      # channel first

	def j2d_processing(self, kp, center, scale, r, f):
		"""Process gt 2D keypoints and apply all augmentation transforms."""
		nparts = kp.shape[0]
		for i in range(nparts):
			kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
			                       [constants.IMG_RES, constants.IMG_RES], rot=r)
		# convert to normalized coordinates
		kp[:, :-1] = 2. * kp[:, :-1] / constants.IMG_RES - 1.  # -1 to 1
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
			sn, cs = np.sin(rot_rad), np.cos(rot_rad)
			rot_mat[0, :2] = [cs, -sn]
			rot_mat[1, :2] = [sn, cs]
		S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1])  # simplified mat * vec to indicate the product order
		# flip the x coordinates,  rotation only, so should be centered 3d
		if f:
			S = flip_kp(S)
		S = S.astype('float32') # vis is kept as original
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


	def get_img_cb(self, mods=['RGB']):
		'''
		from the mods get combined images, with mean, std channel wise from slp_rd, for later processing, or save in config files, but depth will be different,
		:return:
		'''
		if self.dsNm == 'SLP':
			pass
		else:       # add elif  if needed
			print('{} interface not implemented'.format(self.dsNm))
			exit(-1)

	def __getitem__(self, index):
		item = {}
		scale = self.scale[index].copy()
		center = self.center[index].copy()

		# Get augmentation parameters
		flip, pn, rot, sc = self.augm_params()  # per channel noise

		# Load image
		imgname = join(self.img_dir, self.imgname[index])

		li_img = []
		for mod in self.li_mod_SLP:
			img = self.ds.get_array_A2B(idx=index, modA=mod, modB='RGB')    # all to RGB, read RGB order, 0 ~ 1 order
			if 'RGB' != mod:        # add on dim
				img = img[..., None]  # add dim
			li_img.append(img)

		img_cb = np.concatenate(li_img, axis=-1)  # last dim, joint mods, input
		# output image
		depth_dn = self.ds.get_array_A2B(idx=index, modA='depth_dn', modB='RGB', if_uncover=True)   # the covered version
		depth_dn = depth_dn[...,None]/1000.0   # add last channel, to m
		# get ref RGB
		# img_RGB = cv2.imread(imgname)[:,:,::-1].copy()/255.0 # RGB 01       # db is only 2 covers version
		img_RGB = self.ds.get_array_A2B(idx=index, modA='RGB', modB='RGB').copy()/255.0  # the RGB version will be covered cases
		orig_shape = np.array(img.shape)[:2]  # image ref before assignment

		# Get SMPL parameters, if available
		if self.has_smpl[index]:
			pose = self.pose[index].copy()
			betas = self.betas[index].copy()
		else:
			pose = np.zeros(72)
			betas = np.zeros(10)

		# Process image
		img = self.rgb_processing(img_cb, center, sc * scale, rot, flip, pn)
		# adpt image channel to 3
		img = torch.from_numpy(img).float()
		# process the depth_dn
		depth_dn = self.rgb_processing(depth_dn*255.0, center, sc * scale, rot, flip, pn=None)   # keep depth in meter for later, channel first
		img_RGB = self.rgb_processing(img_RGB*255.0, center, sc * scale, rot, flip, pn) # channel first

		# transfer img channelt to 3
		ch = img.shape[0]
		img_normed = self.normalize_img(img)
		if ch == 1: #single channel
			img_normed = torch.cat([img_normed,]*3, dim=0)  # channel wise
		elif ch ==2:
			img_normed = torch.cat([img_normed, img_normed.mean(dim=0, keepdim=True)], dim=0)
		# Store image before normalization to use it in visualization
		# item['img'] = self.normalize_img(img)
		item['img'] = img_normed
		item['depth_dn'] = depth_dn     # depth supervision not normalize,  auto tensor
		item['img_RGB'] = img_RGB     # ref image, not normalized, cropped
		item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
		item['betas'] = torch.from_numpy(betas).float()
		item['imgname'] = imgname
		# make all valid mask2
		mask2 = np.ones((img_normed.shape[1], img_normed.shape[2])).astype(np.float32)
		item['mask2'] = torch.from_numpy(mask2).float()

		# Get 3D pose, if available
		if self.has_pose_3d:    # has get real , otherwise use zeros
			S = self.pose_3d[index].copy()        # db already in meter
			if not self.options.if_vis:  # all -> pelvis visible all -> ht, pelvis 1,  hd_h3d ,1  ht 0
				S[:15, 3] = 1       # all vis, 15 jts
				S[18, 3] = 1        # hip vis
				S[13, 3] = 0        # no ht
			item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
			item['pose_3d_dp'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()        # for depth proxy , SLP gt and dp the same.
		else:
			item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)
			item['pose_3d_dp'] = torch.zeros(24, 4, dtype=torch.float32)

		# Get 2D keypoints and apply augmentation transforms
		keypoints = self.keypoints[index].copy()
		# kp should always be visible

		keypoints[:, 2] = 0
		keypoints[self.idxs_jt, 2] = 1      # all 1
		# keypoints[[8,9,12], 2] = 1  # op_pose visible
		# keypoints[[27,28,39], 2] = 0  # op_pose visible

		item['keypoints'] = torch.from_numpy(
			self.j2d_processing(keypoints, center, sc * scale, rot, flip)).float()  # 49 openpose + gt

		item['has_smpl'] = self.has_smpl[index]
		item['has_pose_3d'] = self.has_pose_3d
		item['has_depth'] = self.has_depth
		item['scale'] = float(sc * scale)
		item['center'] = center.astype(np.float32)      # list now
		item['orig_shape'] = orig_shape
		item['is_flipped'] = flip
		item['rot_angle'] = np.float32(rot)
		item['gender'] = self.gender[index]
		item['sample_index'] = int(index/self.n_cov)        # every 3 belongs to one subj_frames.
		item['dataset_name'] = self.dsNm

		# if self.options.if_vis:
		# 	item['vis'] = self.data['vis'][index]       # add vis, depth_dn chk
		# else:
		# 	item['vis'] = 1 # always vis

		try:        # no mask or part name
			item['maskname'] = self.maskname[index]
		except AttributeError:
			item['maskname'] = ''
		try:
			item['partname'] = self.partname[index]
		except AttributeError:
			item['partname'] = ''

		return item

	def __len__(self):
		return self.length