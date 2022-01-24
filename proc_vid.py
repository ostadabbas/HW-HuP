'''
process video with selected exp model.
save it out to vids folders
'''

import torch
from torchvision.transforms import Normalize

import numpy as np
import argparse
from tqdm import tqdm

import os.path as osp
from glob import glob

import config
import constants
from utils.imutils import uncrop, crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from utils.part_utils import PartRenderer

from utils.renderer import Renderer
from models import hmr, SMPL
import cv2

action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting',
                'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']




def rgb_processing(rgb_img, center, scale, rot, flip, pn=None):
	"""Process rgb image and do augmentation. c,h,w ,0 ~1 """
	rgb_img = crop(rgb_img, center, scale,
	               [constants.IMG_RES, constants.IMG_RES], rot=rot)
	# flip the image
	if flip:
		rgb_img = flip_img(rgb_img)
	if pn is not None:  # will only affect RGB as depth no pn
		for i in range(rgb_img.shape[2]):  # int not subscrible
			rgb_img[:, :, i] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, i] * pn[i]))
	rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
	return rgb_img


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--vid_file', default='~/datasets/vid/infants/baby1.mp4')
	parser.add_argument('--sv_suffix', default='', help='the save video suffix')
	parser.add_argument('--exp_fd', default='logs/h36m_2d_ft_e4', help='the exp folder to read the files')
	parser.add_argument('--out_fd', default='logs/vid', help='the video output folder')
	parser.add_argument('--bb', nargs='+', default=[], type=int, help='the bounding box , default with use whole image')
	parser.add_argument('--checkpoint', default='', help='the checkpoint, pretrained')

	args = parser.parse_args()
	exp_fd = args.exp_fd

	if_dbg = True
	# GET MODEL
	model = hmr(config.SMPL_MEAN_PARAMS)
	if args.checkpoint:
		checkpoint = torch.load(args.checkpoint)
		print('load ckpt from {}'.format(checkpoint))
	else:  # read from the exp folder
		ckpt_fd = osp.join(args.exp_fd, 'checkpoints')
		li_pt = glob(osp.join(ckpt_fd, "*.pt"))
		ckpt_pth = sorted(li_pt)[-1]  # the latest ckpt
		checkpoint = torch.load(ckpt_pth)
		print('load ckpt from {}'.format(ckpt_pth))
	model.load_state_dict(checkpoint['model'], strict=False)
	model.eval()

	# GET read in video
	vidcap = cv2.VideoCapture(args.vid_file)
	n_frm = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=n_frm, desc='processing video {}'.format(args.vid_file))
	count = 0
	success, img = vidcap.read()        # first for testing size
	h, w, c = img.shape  # get the raw shape
	if not success:
		print('video reading failed')
		quit(-1)

	# write in video file gen , std res 224
	res_std = 224
	vid_out_shp = (res_std*3, res_std)

	vid_nm = osp.splitext(osp.split(args.vid_file)[1])[0] # the base name
	sv_suffix = args.sv_suffix
	if sv_suffix:
		vid_nm = '_'.join(vid_nm, sv_suffix)
	vid_pth = osp.join(args.out_fd, vid_nm+'.avi')
	vid_writer = cv2.VideoWriter(vid_pth, cv2.VideoWriter_fourcc(*'DIVX'), 15, vid_out_shp)

	# to device
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model.to(device)
# smpl model in
	if 'infant' in exp_fd:
		smpl_pth = osp.join(config.SMPL_MODEL_DIR, 'SMIL.pkl')
		t_smil = torch.tensor([0.0, 0, -0.46], requires_grad=False).cuda()  # no grad
		s_smil = 2.75
		if_cam_rct = True
	else:
		smpl_pth = config.SMPL_MODEL_DIR
		t_smil = 0
		s_smil = 1.
		if_cam_rct = False

	smpl_neutral = SMPL(smpl_pth,
	                    create_transl=False).to(device)
	# bb or not
	bb = args.bb
	print('bounding box ul, br', bb)
	if not bb:
		bb = [0, 0, w-1, h-1]
	bb = np.array(bb)       # to 4 ele vec
	bb_w, bb_h = bb[2]-bb[0], bb[3]-bb[1]  # assume to be 200
	scale = max(bb_h, bb_w)*0.9/200
	ct = np.array([
		(bb[2] + bb[0]) / 2., (bb[3] + bb[1]) / 2.
	])

	# other settings
	renderer = PartRenderer()
	render_vis = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)
	# Regressor for H36m joints
	J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
	# for aug default
	flip = 0  # flipping
	pn = np.ones(3)  # per channel pixel-noise
	rot = 0  # rotation
	sc = 1  # scaling

	means = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	if 'IR' in exp_fd:
		means = [0.1924838]*3
		std = [0.077975444]*3

	normalize_img = Normalize(mean=means, std=std)

	# loop video
	while success:        # debug false
		# img_ori = img
		# crop
		if 0 and count>120:
			break
		# img = img[:,:,::-1] # to SLP similar
		img = rgb_processing(img, ct, sc * scale, rot, flip, pn)    # BGR-> RGB
		img = torch.from_numpy(img).float().to(device)  # to cuda

		img_RGB = img       # for show
		img_normed = normalize_img(img)    # to torch
		# print('img_normed range min max', img_normed.min(), img_normed.max())
		images = img_normed.unsqueeze(0)  # fake batch

		with torch.no_grad():
			pred_rotmat, pred_betas, pred_camera = model(images)  # pose, shape, camera(ortho projection), z, x, y ?
			if if_cam_rct:
				pred_camera += t_smil
				pred_camera[:, 0] *= s_smil  # 64 match 3 non singular dimension

			pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
			                           global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
			pred_vertices = pred_output.vertices
		camera_translation_bch = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * constants.FOCAL_LENGTH / (
					constants.IMG_RES * pred_camera[:, 0] + 1e-9)], dim=-1)

		# loop batch
		pred_t = pred_vertices[0]   # temp from batch first 1
		img_RGB_t = img_RGB.cpu().numpy()  # to image format
		img_RGB_t = np.transpose(img_RGB_t, (1, 2, 0))  # 0 ~1, BGR
		img_ori = img_RGB_t *255
		img_rd = img_RGB_t[:, :, ::-1]        # if BGR input
		# img_rd = img_RGB_t
		camera_translation_t = camera_translation_bch[0].cpu().numpy()
		pred_vertices0 = pred_t.cpu().numpy()  # single sample
		# Render side views
		aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]  # x, y ,z   right, up ,  outward
		center = pred_vertices0.mean(axis=0)
		rot_vertices = np.dot((pred_vertices0 - center), aroundy) + center  # rotate body
		trans = camera_translation_t.copy()  # ptc, x,y z to world, x, -y , -z
		trans[1] *= -1  # y opposite direction.
		rot_ptc = None
		img_shape, _, _ = render_vis(pred_vertices0, camera_translation_t, img_rd, ptc=None)  # 0 ~1 only if want to have ptc on front view
		img_shape_side, _, _ = render_vis(rot_vertices, camera_translation_t, np.ones_like(img_rd), ptc=None)
		img_shape = 255 * img_shape[:, :, ::-1]
		img_shape_side = 255 * img_shape_side[:,:,::-1]
		img_cmb = np.hstack([img_ori, img_shape, img_shape_side])
		img_cmb = img_cmb.astype(np.uint8)  # cv ask for u8?
		vid_writer.write(img_cmb)
		count+=1
		pbar.update()
		success, img = vidcap.read()    # read in new

	# clean up
	vidcap.release()
	vid_writer.release()
	pbar.close()
