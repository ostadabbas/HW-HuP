'''
generate the figures. generate the changing video.
requires smpl and torch
'''
import numpy as np
import config
import torch
from models import hmr, SMPL
import constants
from utils.renderer import Renderer
import cv2
from tqdm import tqdm
import os.path as osp
import os
import imageio

def gen_vid_poses(poses, betas, cams, aroundys, sv_nm= 'tmp/demo.avi',n_frame=60):
	'''
	from the starting and ending point to render the images
	:param poses: st and end pose
	:param cams: st and end cam
	:param sv_nm: the vid to be saved
	:param n_frame: how many frames to be saved
	:return:
	'''
	# SMPL ,  render_vis
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
	                   create_transl=False).to(device)
	render_vis = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)
	# interpolate the poses to list , to bch, to rotmat
	pose_li = np.linspace(poses[0], poses[1], n_frame)    # 60 x 72
	beta_li = np.linspace(betas[0], betas[1], n_frame)
	cam_li = np.linspace(cams[0], cams[1], n_frame)
	around_li = np.linspace(aroundys[0], aroundys[1],n_frame)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(sv_nm, fourcc, 30.0, (224, 224))
	# with no grad, rend the vertices out inbatch
	with torch.no_grad():
		betas_tch = torch.from_numpy(beta_li).to(device).float()
		pose_tch = torch.from_numpy(pose_li).to(device).float()

		pred_output = smpl_neutral(betas=betas_tch, body_pose=pose_tch[:, 1:], global_orient=pose_tch[:, 0].unsqueeze(1), pose2rot=True)  # to render beta, pose, global_orient
		pred_vertices = pred_output.vertices
		pred_camera = torch.from_numpy(cam_li)
		camera_translation_bch = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * constants.FOCAL_LENGTH / (
					constants.IMG_RES * pred_camera[:, 0] + 1e-9)], dim=-1)
	# loop result vertices batch, render image, add to the vid file
	img_white = np.ones([224,224,3])
	for i, pred_v in tqdm(enumerate(pred_vertices), desc='rend shapes...'):  # loop bch
		# if if_svImg:    # only the first in batch will be saved, otherwise loop it
		# Calculate camera parameters for rendering
		camera_translation_t = camera_translation_bch[i].cpu().numpy()  # bch 1st
		pred_vertices0 = pred_v.cpu().numpy()  # single sample
		aroundy = around_li[i]   # ith
		center = pred_vertices0.mean(axis=0)
		rot_vertices = np.dot((pred_vertices0 - center), aroundy) + center  # rotate body
		img_shape, valid_mask, rend_depth = render_vis(rot_vertices, camera_translation_t, img_white)  # 0 ~1, use white one
		img_cv = (255 * img_shape[:, :, ::-1]).astype(np.uint8)
		out.write(img_cv)  # to BGR 255
	out.release()

def s_pose_evolve():
	'''
	Generate the pose evolve video from natural to hws to slp.
	:return:
	'''
	pose0 = np.zeros(72).reshape([-1, 3])  # 24 x3
	betas0 = np.zeros(10)
	pth_ptn = 'logs/{}/eval_rst.npz'
	idx = 294  # the frame to demo
	# SPIN pose
	exp = 'SLP_2D_e30'  # fine tuned SPIN
	pth = pth_ptn.format(exp)
	dt_in = np.load(pth)
	pose_spin = dt_in['pose'][idx].reshape([-1, 3])  # -1.3 ` 0.7 in rad
	betas_spin = dt_in['betas'][idx]
	cam_spin = dt_in['camera'][idx]  # just use this one

	# update the pose 0 to be same as  spin
	pose0[0, :] = pose_spin[0, :]

	# HW-Hup
	exp = 'SLP_3D_vis_d_e30'  # fine tuned SPIN
	pth = pth_ptn.format(exp)
	dt_in = np.load(pth)
	pose_hw = dt_in['pose'][idx].reshape([-1, 3])
	betas_hw = dt_in['betas'][idx]
	cam_hw = dt_in['camera'][idx]  # just use this one

	## rotation control
	aroundys0 = [cv2.Rodrigues(np.array([0., 0., 0.]))[0], cv2.Rodrigues(np.array([0., 0., 0.]))[0]]
	aroundys = [cv2.Rodrigues(np.array([0., 0., 0.]))[0], cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]]
	aroundys90 = [cv2.Rodrigues(np.array([0., np.radians(90.), 0.]))[0],
	              cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]]

	# gen_vid_poses([pose0, pose_spin], [betas0, betas_spin], [cam_spin, cam_spin], sv_nm='tmp/n2spin.avi')
	# gen_vid_poses([pose_spin, pose_spin], [betas_spin, betas_spin], [cam_spin, cam_spin], aroundys=aroundys,  sv_nm='tmp/spin2side.avi')
	gen_vid_poses([pose_spin, pose_hw], [betas_spin, betas_spin], [cam_spin, cam_spin], aroundys=aroundys90,
	              sv_nm='tmp/spin2hw_side.avi')

def gen_gif(input, n=130, out_fd='logs/vid'):
	'''
	from the input folder/movie generate the gift files.
	:param input:
	:param n: the frame number to process
	:return:
	'''
	if not osp.exists(out_fd):
		os.makedirs(out_fd)

	if osp.isdir(input):
		print('not emplemented yet')
		return -1
	else:
		vidcap = cv2.VideoCapture(input)
		success, img = vidcap.read()
		count = 0
		base_nm = osp.splitext(osp.split(input)[1])[0]
		out_pth = osp.join(out_fd, base_nm+'.gif')
		pbar=tqdm(total=n, desc='generate gift from video {}'.format(input))
		with imageio.get_writer(out_pth, mode='I') as writer:
			while success:
				if count>=n:
					break
				success, img = vidcap.read()
				img = img[:,:,::-1]
				writer.append_data(img)
				count+=1
				pbar.update()
		vidcap.release()

if __name__ == '__main__':
	gen_gif('logs/vid/alex_stair.avi')