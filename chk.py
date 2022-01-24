'''
mainly to check the files of a project
'''
import numpy as np
import pickle as pkl
import os.path as osp
import os
import constants
import cv2

# 3DPW
# fd = r'S:\ACLab\datasets\3DPW\sequenceFiles\train'
# nm = 'courtyard_arguing_00.pkl'
# path = pth.join(fd,nm)
# seq = pkl.load(open(path, 'rb'), encoding='latin1')
# print(seq.keys())   # trans_60Hz list 2 people, , cam, poses, img_frame_ids, betas_clothed, sequence(name), poses_60hz , betas, cam_poses, campose_valid, genders, trans, poses2d, texture_maps(empty)

# SPIN data
# pth = 'data/vertex_texture.npy'
# pth = 'data/cube_parts.npy' # 100 x 92 x 17  x 3
# data_in = np.load(pth)  # size 1x13776 x2 x2 x2 x 3
# print(data_in.size)

def gen_demo_imgs(n=5, npz_nm = '3dpw_test', ds_fd = '/home/liu.shu/datasets/3DPW'):
	'''
	gen images with text on jt, to show order
	'''

	out_fd = osp.join('tmp', npz_nm)
	if not osp.exists(out_fd):
		os.makedirs(out_fd)

	# draw color
	color = (255, 0, 0)
	thickness = 2
	fontScale = 1
	font = cv2.FONT_HERSHEY_SIMPLEX

	pth_in = osp.join('data/dataset_extras', npz_nm+'.npz')
	dt_in = np.load(pth_in)
	img_nms = dt_in['imgname']
	parts = dt_in['part_coco']

	for i in range(n):
		img_nm = img_nms[i]
		img_base_nm = osp.split(img_nm)[1]  # only base name
		img_pth = osp.join(ds_fd, img_nm)
		out_pth = osp.join(out_fd, img_base_nm)
		part = parts[i] # 17 x3
		img = cv2.imread(img_pth)
		for j, jt in enumerate(part):
			if jt[2]:   # if visible
				org = (int(jt[0]), int(jt[1]))
				img = cv2.putText(img, '{:d}'.format(j), org, font, fontScale, color)

		cv2.imwrite(out_pth, img)

if __name__ == '__main__':
	gen_demo_imgs()