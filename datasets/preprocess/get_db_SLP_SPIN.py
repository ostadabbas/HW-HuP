'''
get the SLP db for SPIN. All saved in a similar structure, only for uncover,  extend when needed.
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

from scipy import interpolate
import cv2
import matplotlib.pyplot as plt
# from utils import vis
from utils import utils
import warnings
warnings.filterwarnings('ignore')       # see if surpress the cv

class opt_t:
	SLP_fd = ''
	sz_pch = 224
	fc_depth = 50. # down scale depth
	cov_li = ['uncover']        # process only uncovered

if __name__ == '__main__':
	'''
	add the j2d ->  13 + head(h36m) + pelvis 
	'''
	labNms = [
		'danaLab',
		'simLab'
	]

	parser = argparse.ArgumentParser()
	parser.add_argument('--ratio_head_neck', type=float, default=0.8, help='ratio of the head to neck.')
	labNms = [
		'danaLab',
		'simLab'
	]
	args = parser.parse_args()
	ratio_head_neck = args.ratio_head_neck


	for labNm in labNms:
		opt_t.SLP_fd = osp.join('/home/liu.shu/datasets', 'SLP', labNm)
		SLP_rd = SLP_RD(opt_t, phase='all', if_3d_dp=True, if_SPIN_db=False, rt_hn=ratio_head_neck)    # no spin gen spin
		ds = SLP_rd
		imgnames_, scales_, centers_, parts_, Ss_ = [], [], [], [], []  # parts, 2d, Ss 3d
		vis_ = []  # the visibility

		for i in tqdm(range(ds.n_smpl), desc='processing SLP ADP db'):
			arr, jt, bb = ds.get_array_joints(i, mod='RGB')  # sq true
			j3d_dp = ds.get_j3d_dp(i)       # in rgb
			center = [(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2]  # should be pixel
			scale = 0.9 * max(bb[2] - bb[0], bb[3] - bb[1]) / 200.
			part = np.zeros([24, 3])  # for 2d part
			# first 14 lsp, but change 13(head t) to 18 for h36m, supervise 0:12, 18, then dp also update these , 14 to pelvis
			part[:14] = jt[:]
			part[:14, 2] = 1 - jt[:,2]
			part[14] = (jt[2] + jt[3]) / 2.  # pelvis if both visible then visible
			# part[14, 2] = (1-jt[2, 2]) * (1-jt[3, 2]) # both jt not occluded then 1
			part[14, 2] = 1 # both jt not occluded then 1
			part[18] = jt[12] + ratio_head_neck * (jt[13] - jt[12])        # h36 head wrong?
			part[18, 2] = 1      # head visibility
			part[13, 2] = 0  # remove this no ht, head not visible


			S24 = np.zeros([24, 4])
			jt_dp_3d = ds.get_j3d_dp(i) / 1000.  # from mm to m
			jt_dp_3d[:, :3] -= jt_dp_3d[14, :3]  # pelvis centered
			S24[:15, :3] = jt_dp_3d[:15, :3]        # pevis missing?
			S24[:15, 3] = 1 - jt_dp_3d[:15, 3]  # first 13 good
			S24[18] = jt_dp_3d[13]  # head center
			S24[18, 3] = 1 - jt_dp_3d[13, 3]       # 1 for visible
			S24[13,3] = 0   # clear head top

			# update the rst list
			imgnames_.append(ds.get_img_pth(i))
			centers_.append(center)
			scales_.append(scale)
			parts_.append(part)  # 2d
			Ss_.append(S24)  # rooot cen
			# vis_.append(vis)

		data = {
			'imgname': imgnames_,  # RGB only
			'center': np.array(centers_),
			'scale': scales_,
			'part': parts_,
			'S': Ss_,
			# 'vis': vis_
		}

		sv_pth = osp.join('data', 'SLP_{}_SPIN_db_hn{}.npz'.format(labNm, ratio_head_neck))      # joint mimic h36m, the head area
		if True:
			np.savez(sv_pth,
			         imgname=imgnames_,
			         center=centers_,
			         scale=scales_,
			         part=parts_,
			         S= Ss_,
			         vis=vis_)
			print('db saved to {}'.format(sv_pth))

			data_in = np.load(sv_pth)
			print('part shape', data_in['part'].shape)  # 315 24 3
			print('part 0',  data_in['part'][0])
			print('S 0', data_in['S'][0])

