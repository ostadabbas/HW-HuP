'''
combine image grids
'''
import cv2
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import pyrender as pyr

def cb_imgs_uc(li_exp_fd, li_view=('f_RGB', 's_ptc'), N=540, out_fd='logs/cmb_uc', ld_RGB_fd='logs/SLP_demos/RGB-jt_uncover'):
	'''
	combine the images of uncovered version.
	:param li_exp: the list of exp to be combined
	:param li_view: the list of the type of the images to be combined default for f_RGB and s_ptc (side point cloud.
	:param N: total number of the images
	:param log_fd: the folder holds all the exps
	:param out_fd: where to save the output combined images
	:param ld_RGB_fd: the leading RGB with the joints added
	:return:
	'''
	if not osp.exists(out_fd):
		os.makedirs(out_fd)

	for i in tqdm(range(0,N,3), desc='combine result image grids'):
		pth = osp.join(ld_RGB_fd, '{:05d}.jpg'.format(i))   # lead RGB
		li_img = [cv2.imread(pth)]
		for exp in li_exp_fd:
			for view in li_view:
				pth = osp.join(exp, 'vis', '{:05d}_{}.jpg'.format(i, view))       # 00294_f_RGB.jpg
				img = cv2.imread(pth)
				li_img.append(img)
		# cb all images in a list
		img_cb = np.hstack(li_img)
		out_pth = osp.join(out_fd, '{:05d}.jpg'.format(i))
		cv2.imwrite(out_pth, img_cb)        # save the h concatenated

def cb_imgs_allC(exp_ptn='SLP_3D_vis_d_{}_e30', N=1620, step=3*3, out_fd='logs/cb_allC'):
	'''
	:param exp_ptn: the experiment name pattern
	:param N : total number of samples  total 1620 for SLP eval
	:param step: the step to sample in case the result is generated with step df 3
	combine all cover images  of one approache
	RGB_jt,  depth,  IR, PM,  white
	white,   rst. ..
	white,  rst       fill later one mod
	:return:
	'''
	mod_li = ['depth', 'IR', 'PM', 'stk']
	if not osp.exists(out_fd):
		os.makedirs(out_fd)

	# first row
	img_white = np.ones([224, 224, 3])*255
	img_text = cv2.imread('tmp/sideViewText.png')
	for i in tqdm(range(0,N, step)):      # 9 save the neighboring i+1 i+2
		# first row
		for j in [0,1,2]:
			idx = i+j
			imNm = '{:05d}.jpg'.format(idx)

			RGB_jt = cv2.imread(osp.join('logs/SLP_demos', 'RGB-jt_uncover', '{:05d}.jpg'.format(int(i/3))))  # always the base idx
			img_li = [RGB_jt]     # for initial one
			for mod in mod_li[:-1]:     # SLP_demos/depth/{:05d}.jpg
				img_pth = osp.join('logs', 'SLP_demos', mod, imNm)    # different cov
				img_li.append(cv2.imread(img_pth))
			img_stk = cv2.imread(osp.join('logs','SLP_demos','stk_467', imNm))
			img_stk = cv2.resize(img_stk, (224, 224))       # to std
			# img_li.append(img_white.copy())
			img_li.append(img_stk)
			img_r1 = np.hstack(img_li)      # dim 1st 1,  2nd 3

			# get row two
			img_RGB = cv2.imread(osp.join('logs', 'SLP_demos', 'RGB', imNm))  #the allC version
			img_li = [img_RGB]      # the RGB cov version
			for mod in mod_li:  # SLP_demos/depth/{:05d}.jpg
				img_pth = osp.join('logs', exp_ptn.format(mod), 'vis', '{:05d}_f_RGB.jpg'.format(idx))
				img = cv2.imread(img_pth)
				img_li.append(img)
			img_r2 = np.hstack(img_li)

			# get row 3 side ptc
			img_li = [img_text.copy()]
			for mod in mod_li:  # SLP_demos/depth/{:05d}.jpg
				img_pth = osp.join('logs', exp_ptn.format(mod), 'vis', '{:05d}_s_ptc.jpg'.format(idx))
				img_li.append(cv2.imread(img_pth))
			img_r3 = np.hstack(img_li)
			img_rst = np.vstack([img_r1, img_r2, img_r3])
			pth_sv = osp.join(out_fd, imNm)
			cv2.imwrite(pth_sv, img_rst)        # save for each idx

# trimesh box primitives.box
def cmb_ds_imgs(exp_nm='infant_camL200_ft_e640', dsNm='MIMM', freq=1, crop_suffix='', vid_fd='logs/vid', n_proc=-1):
	'''
	combine the ds  crop( jt) , fitting_f,  fitting_s to a row.  from the exp folder
	:param exp_nm:
	:param dsNm:
	:param n_proc: how many frames to process , default -1 for all
	:return:
	:history: 210928,  add gen video at the same time
	'''
	out_fd = osp.join('logs', '{}_cmb_{}'.format(dsNm, crop_suffix))
	if not osp.exists(out_fd):
		os.makedirs(out_fd)
	vis_fd = osp.join('logs', exp_nm, 'vis')
	crop_fd = osp.join('logs', '{}_crops'.format(dsNm))
	if crop_suffix:
		crop_fd = crop_fd + '_' + crop_suffix
	img_li = os.listdir(crop_fd)
	n = len(img_li)

	# for vid part
	if vid_fd:  # if gen vis
		if not osp.exists(vis_fd):
			os.makedirs(vid_fd)
		vid_pth = osp.join(vid_fd, osp.split(crop_fd)[1] + '.avi')
		img_crop = cv2.imread(osp.join(crop_fd,img_li[0]))
		h, w, c = img_crop.shape
		out = cv2.VideoWriter(vid_pth, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w*3, h))   # 3 combine
		print('gen vid with path {}'.format(vid_pth))

	for i in tqdm(range(0,n,freq), desc='combine infant images'):
		if n_proc>0 and i>n_proc:       #- dbg
			break
		li_img = []
		nm = '{:05d}'.format(i)
		img_crop = cv2.imread(osp.join(crop_fd, '{}.jpg'.format(nm)))
		li_img.append(img_crop)
		img_f = cv2.imread(osp.join(vis_fd, '{}_f_RGB.jpg'.format(nm))) # not read in?
		li_img.append(img_f)
		img_s_ptc = cv2.imread(osp.join(vis_fd, '{}_s_ptc.jpg'.format(nm)))
		li_img.append(img_s_ptc)
		img_cb = np.hstack(li_img)      # 3 dim to 1 dim
		out_pth = osp.join(out_fd, '{}.jpg'.format(nm))
		cv2.imwrite(out_pth, img_cb)
		if vid_fd:
			out.write(img_cb)

	out.release()

def genVid(img_fd='logs/h36m_cmb', out_fd='logs/vid'):
	print('generating video for {}'.format(img_fd))
	if not osp.exists(out_fd):
		os.makedirs(out_fd)
	vid_pth = osp.join(out_fd, osp.split(img_fd)[1]+'.avi')     # base + .avi

	file_nm_li = os.listdir(img_fd)
	file_nm_li.sort()
	pth0 = osp.join(img_fd, file_nm_li[0])
	img0 = cv2.imread(pth0)
	h, w, c = img0.shape

	out = cv2.VideoWriter(vid_pth, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))

	for nm in tqdm(file_nm_li):
		pth = osp.join(img_fd, nm)
		img = cv2.imread(pth)
		out.write(img)

	out.release()

if __name__ == '__main__':
	# li_exp = ['SLP_2D_e30', 'SLP_3D_e30', 'SLP_3D_vis_d_e30', 'SLP_3D_vis_d_op0_e30']       # give the exp names here
	# li_exp = [osp.join('logs', nm) for nm in li_exp]
	# cb_imgs_uc(li_exp, N=540) #2 for debug otherwise  540

	# combine all
	# cb_imgs_allC(N=1620)

	# infant
	# cmb_ds_imgs(dsNm='SyRIP')
	# cmb_ds_imgs(dsNm='3DPW')
	# cmb_ds_imgs(exp_nm='3dpw_eval', dsNm='3dpw', freq=3)        # 3dpw
	# cmb_ds_imgs(exp_nm='h36m_2d_ft_e4', dsNm='h36m', freq=1, crop_suffix='s50')
	cmb_ds_imgs(exp_nm='h36m_2d_ft_e4', dsNm='h36m', freq=1, crop_suffix='s5')
	# genVid()  # default h36m
	# genVid(img_fd='logs/3dpw_cmb')  # 3dpw