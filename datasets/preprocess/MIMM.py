import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
# from spacepy import pycdf
import cdflib
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from os.path import join

# def MIMM_extract(dataset_path, out_path, c_path, protocol=1, extract_img=False, is_train=False):
def MIMM_extract(dataset_path, out_path, is_train=False):
    '''
    hist: 21-4-18,  add the  predicted head points, counted as visible if score >0.5
    :param dataset_path:
    :param out_path:
    :param is_train:
    :return:
    '''
    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]  # convert to 17 joints, 1st 5 are face

    # joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]     # convert to 17 joints
    # right left hip  2, 3  , visible to 0 for
    # bbox expansion factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, depthnames_, Ss_dp_ = [], [], [], [], [], [], []
    # no openpose make part and Ss same projection

    # read int the anno,  depending on the train or not
    # users in validation set

    # json annotation file
    if is_train:
        split = 'train'
    else:
        split = 'valid'

    json_path = os.path.join(dataset_path,
                             'anno_{}.json'.format(split))
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img  # {0: {'file_name':xx, 'RGB':....}
    n_chk = -1
    for i, annot in tqdm(enumerate(json_data['annotations']), desc='gen SyRIP db for SPIN...'):
        # keypoints processing
        if n_chk > 0 and i >= n_chk:
            break
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17, 3))
        keypoints[keypoints[:, 2] > 0, 2] = 1
        # gen the pred key poitns
        pred_keypoints = annot['pred_keypoints']
        pred_keypoints = np.reshape(pred_keypoints, (17, 3))
        pred_keypoints[pred_keypoints[:, 2] > 0.5, 2] = 1       # use 0.5 score to make visible or not
        # check if all major body joints are annotated
        if sum(keypoints[5:, 2] > 0) < 12:
            continue
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name']) # direct index
        img_name_full = join('RGB', img_name)  # relative from ds folder to images
        depth_nm = str(imgs[image_id]['depth_name'])
        depth_nm_full = join('depth_dn', depth_nm)  # relative from ds folder to images

        # keypoints
        part = np.zeros([24, 3])        # 24 part
        part[joints_idx] = keypoints  # 24 joints, put the gt 17 in, 2, 3 vis to 0 , add openpose jt
        # update the face parts
        part[joints_idx[:5]] = pred_keypoints[:5]       # first 5 only
        # add pseudo pelvis 14
        part[14] = (part[2] +part[3])/2.
        part[14, 2] = part[2, 2]*part[3,2]   # show the visibility

        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        scale = scaleFactor * max(bbox[2], bbox[3]) / 200

        kp3d_vis = annot['3d_keypoints']    # should man made a pelvis here,
        kp3d_vis = np.reshape(kp3d_vis, (17,4))
        kp3d = kp3d_vis[:, :3]
        kp3d /= 1000.
        vis = kp3d_vis[:, 3]
        vis[vis>0] =1
        pelvis_3d = (kp3d_vis[12]+ kp3d_vis[11])/2.      #  coco  r,l hip 12,11
        pelvis_3d[3] = vis[12]*vis[11]
        kp3d -= pelvis_3d[:3]

        # kp3d[:,3][kp3d[:,3]>0] = 1  # present then 1
        S24 = np.zeros([24, 4])
        S24[joints_idx, :3] = kp3d      # should be 0
        S24[joints_idx, 3] = vis
        S24[14,3] = pelvis_3d[3]     # pseudo

        S24_dp = np.zeros([24, 4])
        S24_dp[joints_idx, :3] = kp3d
        S24_dp[joints_idx, 3] = vis
        S24_dp[14, 3] = pelvis_3d[3]  # should be 0 only keep vis

        # debug show
        if not n_chk < 0:
            print('id {} part and S24'.format(i))
            print(part)
            # print(S24)
            # print('depth name', depth_nm_full)
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)  # gt 17
        # depth name , Ss Ss_dp
        depthnames_.append(depth_nm_full)
        Ss_.append(S24)
        Ss_dp_.append(S24_dp)


    # store the data struct
    print('{} data length'.format(split), len(imgnames_))
    if n_chk < 0:
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        if is_train:
            out_file = os.path.join(out_path, 'MIMM_train.npz')
        else:
            out_file = os.path.join(out_path, 'MIMM_valid.npz')
        print("file saved to {}".format(out_file))
        np.savez(out_file, imgname=imgnames_,
                 center=centers_,
                 scale=scales_,
                 S=Ss_,
                 part=parts_,
                 S_dp=Ss_dp_,
                 depthname=depthnames_)


if __name__ == '__main__':
    ds_pth = '/scratch/liu.shu/datasets/MIMMv2'
    out_pth = '/scratch/liu.shu/codesPool/SPIN/data/dataset_extras'
    MIMM_extract(ds_pth, out_pth, is_train=False)