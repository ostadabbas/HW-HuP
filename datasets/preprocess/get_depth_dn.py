'''
get denoised SLP depth data.
use multiprocessing to deal with.
S24 jt_3d centered on pelvis
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
from kinect_smoothing.utils import plot_image_frame, plot_trajectories, plot_trajectory_3d
from kinect_smoothing import HoleFilling_Filter, Denoising_Filter
from multiprocessing import Pool, cpu_count
import time
import cv2

# class opt_t:        # pseudo opts for SLP_rd
# 	SLP_fd = ''
# 	sz_pch = 224
# 	fc_depth = 50.  # down scale depth
# 	cov_li = ['uncover']  # process only uncovered

def get_depth_dn(pth_ori, pth_tar, if_downSmpl=False):
	'''
	Get denoised depth frame and save the result from input and output path.
	read in the pth ori, then save to pth tar. The folder should be build before calling
	:param pth_ori:
	:param pth_tar:
	:return: N/A
	'''
	tm_st = time.time()
	arr = np.load(pth_ori)
	# 1920 x 1080 ds4 = 480 x 270
	if if_downSmpl:
		arr = cv2.resize(arr.astype(float), (480, 270))

	hole_filter = HoleFilling_Filter(flag='min')
	arr_hf = hole_filter.smooth_image(arr)
	noise_filter = Denoising_Filter(flag='anisotropic', theta=60)
	arr_dn = noise_filter.smooth_image(arr_hf)  # denoised version
	if if_downSmpl:
		arr_dn = cv2.resize(arr_dn, (1920, 1080))
	np.save(pth_tar, arr_dn)
	# print('{} processed with time cost {}'.format(pth_ori, time.time()-tm_st))

def run_get_depth_dn_SLP(if_chk=False):
	ds_fd = '/home/liu.shu/datasets'
	nm_ori = 'SLP'
	nm_aug = nm_ori    # augmented SLP datasets
	# nm_aug = 'SLP_AUG'      # augmented SLP datasets
	p = Pool(28)
	labNms = [
		'danaLab',
		# 'simLab'
	]
	dct_nSubj = {
		'danaLab':102,
		'simLab': 7
	}
	cov_li = ['uncover', 'cover1', 'cover2']
	print("current cpu numbers :{}".format(cpu_count()))
	print('main pid: {}'.format(os.getpid()))

	tm_st = time.time()

	for labNm in labNms:
		nSubj = dct_nSubj[labNm]
		# nSubj = 2       # try two person first
		for i in tqdm(range(nSubj), 'processing lab {}'.format(labNm)):
			for cov in cov_li:
				str_idx_subj = '{:05d}'.format(i+1)
				fd_ori = join(ds_fd, nm_ori, labNm, str_idx_subj, 'depthRaw', cov)
				fd_tar = join(ds_fd, nm_aug, labNm, str_idx_subj, 'depth_dn', cov)
				if not os.path.exists(fd_tar):
					os.makedirs(fd_tar)     # make denoised depth folder in aug SLP
				for j in range(45):
					pth_ori = join(fd_ori, '{:06d}.npy'.format(j+1))
					pth_tar = join(fd_tar, '{:06d}.npy'.format(j+1))
					# if if_chk and j % 5 == 0:
					# 	depth_dn = np.load(pth_tar)
					# 	plot_image_frame([depth_dn])
					if if_chk:
						if not os.path.exists(pth_ori):
							print('{} does not exist', pth_ori)
					else:
						p.apply_async(get_depth_dn, args=(pth_ori, pth_tar))

	p.close()   # end
	p.join()
	print('Total time cost {}'.format(time.time()- tm_st))


def run_get_depth_dn_MIMM(if_chk=False):        # work on MIMM dataset
	# if_chk: only check the generated image, not save it
	ds_fd = '/home/liu.shu/datasets/MIMMv2'
	if not if_chk:
		p = Pool(28)

	# list all depth images,  then loop
	fd_ori = osp.join(ds_fd, 'depthRaw')
	fd_tar = osp.join(ds_fd, 'depth_dn')
	if not os.path.exists(fd_tar):
		os.makedirs(fd_tar)
	depth_files = os.listdir(fd_ori)

	print("current cpu numbers :{}".format(cpu_count()))
	print('main pid: {}'.format(os.getpid()))
	tm_st = time.time()
	n_proc = -1     # how many to process
	for j, nm in tqdm(enumerate(depth_files), desc='process MIMM depth'):
		pth_ori = join(fd_ori, nm)
		pth_tar = join(fd_tar, nm)
		if n_proc > 0 and n_proc == j:
			break
		if if_chk and j % 5 == 0:
			depth_dn = np.load(pth_tar)
			# depth_dn = np.load(pth_ori)
			plot_image_frame([depth_dn])
		if if_chk:
			if not os.path.exists(pth_ori):
				print('{} does not exist', pth_ori)
		else:
			p.apply_async(get_depth_dn, args=(pth_ori, pth_tar, True))  # down sample it 1920 /4
	if not if_chk:
		p.close()  # end
		p.join()
	print('Total time cost {}'.format(time.time() - tm_st))

if __name__ == '__main__':
	# run_get_depth_dn_SLP(if_chk=False)
	# run_get_depth_dn_MIMM(if_chk=True)
	run_get_depth_dn_MIMM(if_chk=False)

