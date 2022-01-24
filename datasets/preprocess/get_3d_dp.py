'''
to get the pseudo 3d depth proxy from the SLP data, save result out.
use h36m head with 0.75 of neck to head top position. interpolation joint with and vis of the two ends.
read from denoised verison (dn), jt_3d in depth coordinate (mm).

'''
import os
from os.path import dirname, join, abspath
import sys
print(os.getcwd())
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))
from datasets import SLP_RD
import os.path as osp
import numpy as np
from tqdm import tqdm
import argparse

# from utils import vis
from utils import utils
import warnings
warnings.filterwarnings('ignore')       # see if surpress the cv
import json

class opt_t:
	SLP_fd = ''
	sz_pch = 224
	fc_depth = 50. # down scale depth
	cov_li = ['uncover']        # process only uncovered

if __name__ == '__main__':
	'''
	add the j2d ->  13 + head(h36m) + pelvis 
	'''
	# add parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--ratio_head_neck', type=float, default=0.7, help='ratio of the head to neck.')
	labNms = [
		'danaLab',
		'simLab'
	]
	args = parser.parse_args()
	ratio_head_neck = args.ratio_head_neck

	# ratio_head_neck = 0.9       # to get the h36m head  0.9

	for labNm in labNms:
		opt_t.SLP_fd = osp.join('/home/liu.shu/datasets', 'SLP', labNm)
		SLP_rd = SLP_RD(opt_t, phase='all', if_3d_dp=False, if_SPIN_db=False)
		# print(SLP_rd.pthDesc_li[4458:])     # no problem
		# exit(0)

		li_3d_dp = []
		li_3d_dp_subj = []  #  to keep each subj to be compatible to SLP jt format
		i_chk = 15   # prob: [28, 34, 131, 165, 174, 176, 178, 197, 240, 257, 269, 297, 301, 302, 307, 332, 375, 424] ， sim: 93 101
		n_test = 1 # how many to test
		# n_test = SLP_rd.n_smpl  # how many to test
		thr_l = 1500
		thr_h = 2400        # possible threshold
		li_prob_missing = []
		li_prob_bg = []     # for jt fall to bg area
		idxs_zero = []
		rst_d = []
		for i in tqdm(range(SLP_rd.n_smpl)):
		# for i in tqdm(range(i_chk, i_chk+n_test)):   # debug check only 1
			arr, jt, bb = SLP_rd.get_array_joints(i, mod='depth_dn')    # directly use the smoothed, !! depth coordinate
			jt[-1,:2] = jt[-2,:2] + ratio_head_neck *(jt[-1, :2] - jt[-2,:2])   # h36m like head, head share the head top visibility
			jt_pelvis = (jt[2] + jt[3])/2
			if jt[2,2] == 0 and jt[3,2] == 0 :  # all not occluded
				jt_pelvis[2] = 0
			else:
				jt_pelvis[2] = 1
			jt = np.vstack([jt, jt_pelvis])     # should be 15 x3
			shp = jt.shape
			assert shp[1] == 3, 'jt shp has to be 3 with confidence yet get {}'.format(shp[1])
			jt_25d_dp = np.zeros([shp[0], 3])     # 15 joints
			jt_3d_dp = np.zeros([shp[0], 4])     # 15 joints

			jt_25d_dp[:, :2] = jt[:, :2]
			jt_3d_dp[:, 3] = jt[:, 2]       # the occlusion 1 visible 0

			jt_near = (jt[:,:2]+0.5).astype(int)
			jt_25d_dp[:, 2] = arr[jt_near[:, 1], jt_near[:, 0]]        # use original or filtered,  1202 out of 1024
			j3d_t = utils.pixel2cam(jt_25d_dp, SLP_rd.f_d, SLP_rd.c_d)
			jt_3d_dp[:,:3] = j3d_t  # first 3 columns

			if not jt_25d_dp[:, 2].all():       # check for missing.  if there missing 0 point
				li_prob_missing.append(i)
				print('id {} get depth vec'.format(i))
				print(jt_25d_dp[:, 2])

			if not (jt_3d_dp[:,2]<thr_h).all():
				print('idx {} out of range on ground'.format(i))
				jt_3d_dp[jt_3d_dp[:,2]>thr_h, 3] = 1    # set to 1 for occluded
				print('missing label', jt_3d_dp)

			rst_d.append(jt_25d_dp[:, 2])       # for hist vis
			# update the final rst
			li_3d_dp_subj.append(jt_3d_dp.tolist())
			if (i + 1) % 45 == 0:  # if the last one
				li_3d_dp.append(li_3d_dp_subj)
				li_3d_dp_subj = []

		# quality check
		print('problem indices', li_prob_missing)

		sv_pth = osp.join('data', 'SLP_{}_3d_dp_h36_hn{}.json'.format(labNm, ratio_head_neck))      # joint mimic h36m, the head area, head neck position
		if True:
			with open(sv_pth, 'w') as f:
				print('saving the 3d_dp to {}'.format(sv_pth))
				json.dump(li_3d_dp, f)
			# dump the json file li_3d_dp

			with open(sv_pth, 'r') as f:
				rst = json.load(f)
				print('{} result reading test subj0 frm 0'.format(sv_pth))
				print(rst[0][0])
				print('shape', np.array(rst[0][0]).shape)
				print("subj num", len(rst))


