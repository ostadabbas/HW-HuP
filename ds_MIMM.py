'''
clean up the MIMM dataset to /RGB /depth/   MIMM5006_00037.jpg  format
ori: MIMM5015-IR-2016-10-17_rgb_13670.jpg
modeling infant motor  movement from co,.
gen folder
MIMMv2
`-rgb
`-depth
`-depthRaw
annotations_all.json
annotations_train.json
annotations_valid.json

ori, images[{},  {}...]
annotations,
categories
'''

import os
import os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import utils.utils as ut_t
from utils.vis import vis_kps, vis_keypoints

MIMM_tar_fd = '/scratch/liu.shu/datasets/MIMMv2'      # for MIMM only
SYRIP_tar_fd = '/scratch/liu.shu/datasets/SyRIPv2'

def gen_MIMM():
	ds_fd = '/scratch/liu.shu/datasets/MIMM'

	dct_subj= {}    # if not in then add the subj in ther folder
	if not osp.exists(MIMM_tar_fd):
		os.makedirs(osp.join(MIMM_tar_fd, 'RGB'))  # image
		os.makedirs(osp.join(MIMM_tar_fd, 'depth'))    # only for visualization purpose
		os.makedirs(osp.join(MIMM_tar_fd, 'depthRaw'))     #

	anno_pth = osp.join(ds_fd, 'annotations', 'MIMM_keypoints_validate_mimm_1050.json') # split from all set
	with open(anno_pth, 'r') as f:
		dt_in = json.load(f)

	# N = 1050
	i_stop = -1 # no stop
	for i, img in tqdm(enumerate(dt_in['images']), desc='process the MIMM dataset', total=1050):
		if i_stop>0 and i >= i_stop:
			break
		f_nm = img['file_name']
		ori_f_nm = img['original_file_name']
		# simplyf ori name read in depth , read in RGB, transfer save out, change image name to ori_s
		ori_f_nm_s = ori_f_nm[:9] + ori_f_nm[-9:-4]  # simple version , no jpg MIMM5015-13670
		subj_nm = ori_f_nm[:8]      # no bar
		if subj_nm not in dct_subj:
			dct_subj[subj_nm] = i       # keep the first index
		pth_rgb = osp.join(ds_fd, 'images', 'validate_mimm_1050', f_nm)
		rgb = cv2.imread(pth_rgb)
		pth_sv = osp.join(MIMM_tar_fd, 'RGB', ori_f_nm_s + '.jpg')
		cv2.imwrite(pth_sv, rgb)
		# depth
		h = 1080
		w = 1920
		with open(osp.join(ds_fd, 'images', 'depth_data', ori_f_nm_s +'.txt'), 'r') as f:
			line_s = f.read().splitlines()
			depth_list = [elem for elem in line_s if elem != '']
			depth_list = list(map(int, depth_list))
			depthRaw = np.reshape(depth_list, (h, w))
		# norm image,  min to max  0 ~ 2500  only for vis
			depth = ut_t.normImg_clip(depthRaw, (0,3000), False)
		pth_sv = osp.join(MIMM_tar_fd, 'depthRaw', ori_f_nm_s + '.npy')
		np.save(pth_sv, depthRaw)
		pth_sv = osp.join(MIMM_tar_fd, 'depth', ori_f_nm_s + '.jpg')
		cv2.imwrite(pth_sv, depth)
		# update the dict file name
		img['file_name'] = ori_f_nm_s +'.jpg'      # for RGB
		img['depth_name'] = ori_f_nm_s + '.npy'

	print("total subj", len(dct_subj))
	print("subj st idx", dct_subj)
	# print("first  2 record of images", dt_in['images'][:2])
	pth_sv = osp.join(MIMM_tar_fd, 'anno_all.json')
	with open(pth_sv, 'w') as f:
		json.dump(dt_in, f)     # updated json  filename depth_name
		f.close()
	# for subj
	pth_sv = osp.join(MIMM_tar_fd, 'subj_list.json')
	with open(pth_sv, 'w') as f:
		json.dump(dct_subj, f)
		f.close()

def cmb_SyRIP():
	'''
	combine the SyRIP ds, with consistent names.
	S+R 1200 + 500 validation, with 1000 syn + 100 + 400 real ,  100 for valid
	combine all first, then split  accordingly.
	change id and image name
	:return:
	'''
	ds_fd = '/scratch/liu.shu/datasets'
	tar_fd = osp.join(ds_fd, 'SyRIPv2')
	img_tar_fd = osp.join(tar_fd, 'RGB')
	if not osp.exists(img_tar_fd):
		os.makedirs(img_tar_fd)

	src_fd = '/scratch/liu.shu/datasets/SyRIP/'
	# anno1 , loop save, dict, enumerate, as base for next.
	anno1_pth = osp.join(src_fd, 'annotations/200R_1000S/person_keypoints_train_infant.json')
	anno2_pth = osp.join(src_fd, 'annotations/validate500/person_keypoints_validate_infant.json')
	with open(anno1_pth, 'r') as f:
		dt_in = json.load(f)
		f.close()
	with open(anno2_pth, 'r') as f:
		dt_in2 = json.load(f)
		f.close()
	# combine
	# print('before combine dt_in', len(dt_in['images']))
	n_part1 = len(dt_in['images'])  # how many entries in there, at this entry use annother target folder
	# sort by id
	dt_in['images'].sort(key=lambda x: x['id'])     # sort by id
	dt_in['annotations'].sort(key=lambda x: x['id'])     # sort by id
	dt_in2['images'].sort(key=lambda x: x['id'])  # sort by id
	dt_in2['annotations'].sort(key=lambda x: x['id'])  # sort by id

	dt_in['images'] += dt_in2['images']
	dt_in['annotations'] += dt_in2['annotations']
	# print("after combine, the dt_in length", len(dt_in['images']))    # 1200 to 1700
	n_total = len(dt_in['images'])
	n_chk = -1

	for i, item in tqdm(enumerate(dt_in['images']), desc='organizing SyRIP...',total=n_total):
		if n_chk>0 and i>= n_chk:
			break
		f_nm = item['file_name']
		if i<n_part1:       #  at 1200 to valid folder
			img_src_pth = osp.join(src_fd, 'images/1200', f_nm)
		else:       # for valid part
			img_src_pth = osp.join(src_fd, 'images/validate_500', f_nm)

		f_tar_nm = 'image_{:05d}.jpg'.format(i)
		img_tar_pth = osp.join(img_tar_fd, f_tar_nm)        # save as new name
		rgb = cv2.imread(img_src_pth)
		if 1:   # if save images or not
			cv2.imwrite(img_tar_pth, rgb)
		item['file_name'] = f_tar_nm    # change name
		item['id'] = i
		dt_in['annotations'][i]['id']=i # update annotation to save one

	json_tar_pth= osp.join(tar_fd, 'anno_all.json')
	with open(json_tar_pth, 'w') as f:
		json.dump(dt_in, f)
		f.close()
		print("anno saved to {}".format(json_tar_pth))


def split_MIMM(n_split=834, ds_tar_fd='/scratch/liu.shu/datasets/MIMMv2'):
	'''
	split the annotations to train and test  0:834  train, 834:1050 valid MIMM
	syrip last 100 for test.
	834 for the MIMM set left last 2 persons.
	:return:
	'''
	anno_pth = osp.join(ds_tar_fd, 'anno_all.json')
	with open(anno_pth, 'r') as f:
		dt_in = json.load(f)
		f.close()
	# n_split = 834       # the 2nd subj
	dt_in['images'].sort(key=lambda x: x['id'])
	dt_in['annotations'].sort(key=lambda x: x['id'])    # in order

	train_db = {
		'images': dt_in['images'][:n_split],
		'annotations': dt_in['annotations'][:n_split],
		'categories': dt_in['categories']       # single entry keep all
	}
	sv_pth = osp.join(ds_tar_fd, 'anno_train.json')
	with open(sv_pth, 'w') as f:
		json.dump(train_db, f)
		f.close()
		print('save train anno to {}'.format(sv_pth))

	valid_db = {
		'images': dt_in['images'][n_split:],
		'annotations': dt_in['annotations'][n_split:],
		'categories': dt_in['categories']
	}
	sv_pth = osp.join(ds_tar_fd, 'anno_valid.json')
	with open(sv_pth, 'w') as f:
		json.dump(valid_db, f)
		f.close()
		print('save valid anno to {}'.format(sv_pth))

def vis_ds(ds_fd='/scratch/liu.shu/datasets/SyRIPv2'):
	'''
	visualize the keypoints of ds and save it out
	:param ds_fd:
	:return:
	'''
	tar_fd = osp.join(ds_fd, 'vis')
	if not osp.exists(tar_fd):
		os.makedirs(tar_fd)

	anno_pth = osp.join(ds_fd, 'anno_all.json')
	with open(anno_pth, 'r') as f:
		dt_in = json.load(f)
		f.close()
	image_rcds = dt_in['images']

	n_chk = -1
	skl = np.array(dt_in['categories'][0]['skeleton'])-1    # 0 index
	n = len(image_rcds)
	for i, item in tqdm(enumerate(image_rcds),desc='visulaize...', total=n):
		if n_chk>0 and i>=n_chk:
			break
		f_nm= item['file_name']
		img_pth = osp.join(ds_fd, 'RGB', f_nm)
		kps = dt_in['annotations'][i]['keypoints']
		kps = np.array(kps).reshape([17,3])
		img =cv2.imread(img_pth)
		# img_vis = vis_kps(img, kps[:,:2], kps[:, 2], [skl])
		img_vis = vis_keypoints(img, kps, skl)
		if 0:       # debug vis
			cv2.imshow('test', img_vis)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		tar_pth = osp.join(tar_fd, f_nm)
		cv2.imwrite(tar_pth, img_vis)




def ex_loc():
	'''
	for the local test of the files
	:return:
	'''
	# test the json content
	# pth = r'S:\ACLab\datasets\MIMM_demo\MIMM_keypoints_train_mimm.json'
	# with open(pth, 'r') as f:
	# 	dt_in = json.load(f)
	# 	f.close()
	# print(dt_in.keys())
	# print(dt_in['images'][0])
	# print(dt_in['annotations'][0])
	# print(dt_in['categories'][0])

	pth = r'S:\ACLab\datasets\MIMM_demo\MIMM5006-00037.txt'
	h= 1080
	w = 1920
	with open(pth, 'r') as f:
		line_s = f.read().splitlines()
		depth_list = [elem for elem in line_s if elem != '']
		depth_list = list(map(int, depth_list))
		depthRaw = np.reshape(depth_list, (h, w))

	# hist, bins = np.histogram(depthRaw, bins=50)
	plt.hist(depthRaw)      # roughly  800 ~ 2500 , other small cluster
	plt.show()

def update_anno():
	'''
	update the anno all files to remove the '__' to single '_'
	:return:
	'''
	anno_pth = osp.join(MIMM_tar_fd, 'anno_all.json')
	with open(anno_pth, 'r') as f:
		dt_in = json.load(f)
		f.close()
	images = dt_in['images']
	n_chk = -1
	for i, item in tqdm(enumerate(images)):
		if n_chk>0 and i >= n_chk:
			break
		ori_nm = item['original_file_name']
		item['original_file_name'] = ori_nm.replace('__', '_')

	# for i in range(n_chk+2):
	# 	print('after updateing', images[i]['original_file_name'])
	# 	print('in dt', dt_in['images'][0]['original_file_name'])
	with open(anno_pth, 'w') as f:
		json.dump(dt_in, f)
		f.close()
	print('updated anno saved to {}'.format(anno_pth))

def chk_depth(if_update=False):
	'''
	for all 3d_kp, the depth and vis should be be 0 at the same time.
	index vis, if any 0 then print
	:return:
	'''
	anno_pth = osp.join(MIMM_tar_fd, 'anno_all.json')
	with open(anno_pth, 'r') as f:
		dt_in = json.load(f)
		f.close()
	annos = dt_in['annotations']
	n_chk = -1
	li_zero_depth = []
	for i, anno in enumerate(annos[0:]):
		if n_chk > 0 and i >= n_chk:
			break
		kp_3d = np.array(anno['3d_keypoints']).reshape([-1, 4])
		depth = kp_3d[:, 2]      # all 17 joints
		vis = kp_3d[:, 3]

		vis_flt = np.zeros_like(vis)
		vis_flt[depth>0] = 1.
		# print('depth before filter', vis)
		if if_update:
			kp_3d[:, 3] = vis*vis_flt       # has to poitn to orig kp
			anno['3d_keypoints'] = kp_3d.flatten().tolist() # update the kp 3d  in annot
		idx_pres = vis > 0  # 1 or 2    updated vis
		depth_pres = depth[idx_pres]
		# print("depth pres", depth_pres)
		if not np.all(depth_pres):  # if not all non 0
			li_zero_depth.append((i, dt_in['images'][i]['file_name']))
			# print('idx {} has vis point as 0'.format(i))
			# print("3d_kp is")
			# print(kp_3d)
	print('the problem frame with 0 depth jt', li_zero_depth)
	print("total", len(li_zero_depth))

	print('check dt file')
	print(dt_in['annotations'][4]['3d_keypoints'])
	if if_update:
		with open(anno_pth, 'w') as f:
			json.dump(dt_in, f)
			f.close()
		print('updated anno saved to {}'.format(anno_pth))


if __name__ == '__main__':
	# gen_MIMM()
	# cmb_SyRIP()
	# vis_ds()
	split_MIMM(ds_tar_fd=SYRIP_tar_fd, n_split=-100)    # last 100 for testing
	# ex_loc()
	# update_anno()
	# chk_depth(False)