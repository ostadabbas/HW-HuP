# for figure generations
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 6,
        }
matplotlib.rc('font', **font)

def get_d_shift_hist(if_abs=True):
	'''
	get the histogram plot
	:return:
	'''
	out_fd = 'rst_figs'

	if not osp.exists(out_fd):
		os.makedirs(out_fd)

	dt_pth = 'data/dataset_extras/h36m_train_2.npz'
	dt_in = np.load(dt_pth)
	S = dt_in['S']      # 24 jts  N x 24 x 3
	S_dp = dt_in['S_dp']

	# print('S shape', S.shape)   # 31476 x 24 x4  (visible)

	S_z = S[::10,:, 2]      # 3148 x 24
	S_dp_z = S_dp[::10,:, 2]

	pbar = tqdm(8)

	# print('S_z shape, type', S_z.shape, type(S_z))
	if if_abs:
		z_diff = np.abs(S_z - S_dp_z) *1000
	else:
		z_diff = S_z - S_dp_z *1000

	print('z_diff min max', z_diff.min(), z_diff.max()) # 0 to 1000


	z_diff_jt_li = []       # the joint list of diffs, ankle, knee, hip, wrist elbow, shoulder , head and total

	name_li = [ 'Ankle', 'Knee', 'Hip', 'Wrist', 'Elbow', 'Shoulder', 'Head', 'Total']

	# err: only integer scalr array
	z_diff_jt_li = [
		np.concatenate([z_diff[:, 0], z_diff[:, 5]]),
		np.concatenate([z_diff[:, 1], z_diff[:, 4]]),
		np.concatenate([z_diff[:, 2], z_diff[:, 3]]),
		np.concatenate([z_diff[:, 6], z_diff[:, 11]]),
		np.concatenate([z_diff[:, 7], z_diff[:, 10]]),
		np.concatenate([z_diff[:, 8], z_diff[:, 9]]),
		z_diff[:, 18],
	]
	z_diff_jt_li.append(np.concatenate(z_diff_jt_li))   # all together

	# all subplots, add to list,
	for i, diff in enumerate(z_diff_jt_li):
		ax = plt.subplot(2, 4, i+1)
		plt.hist(z_diff_jt_li[i], density=True)
		plt.title(name_li[i])
		plt.xlim([0, 600])
		plt.ylim([0, 0.01])
		plt.xlabel('3D error in depth (mm)')
		plt.ylabel("density")
		plt.tight_layout()


	plt.savefig(osp.join(out_fd, 'd_err_hist.pdf'))
	np.save(osp.join(out_fd, 'diff_rst.npy'), np.array(z_diff_jt_li))

if __name__ == '__main__':
	get_d_shift_hist()