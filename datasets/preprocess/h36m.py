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

def h36m_extract(dataset_path, out_path, protocol=1, extract_img=False):

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_  = [], [], [], [], []

    # users in validation set test,  train 2,3,5  .. .  eval:  how to determine the action
    user_list = [9, 11]

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # depth path

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in tqdm(seq_list, desc='processing sub {}'.format(user_i)):  # all cdf file list with folder in front?

            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')     # SittingDown 2.54138969.cdf
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            # poses_3d = pycdf.CDF(seq_i)['Pose'][0]
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]     # use another library

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                if not os.path.exists(imgs_path):
                    os.makedirs(imgs_path)
                vidcap = cv2.VideoCapture(vid_file)
                success, image = vidcap.read()

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):        #
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break
                    # depth =  extract .

                # check if you can keep this frame
                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):  # only protol 1 is saved or else  for p2
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    # depth name j2d - . d3d
                    # check d3d if 0 ?   openpose  s9 s11 ,
                    
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)

                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]     # should be pixel
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.  # scale is 0.9 of the bb

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.    # in m
                    S17 = Sall[h36m_idx]    # 17 joints from original 30 jts
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24, 4])     # 24 x 4  all 0 , except the  known joint 1
                    S24[global_idx, :3] = S17   # put into 24 joints order
                    S24[global_idx, 3] = 1

                    # store data, depth, 3d_dp (
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    Ss_.append(S24)     # rooot centered action_id
                    # no 3D gt:
                    #  ours:  2D ,  depth,  3D_dp ( 2d+depth : ptc recover) vis!,
                    # Ss_dp.  depth, part, vis,
                    #  Ss all true vis
                    # Ss_dp   some wrong, some occlusigon 3 filters  (occlu+ depth)  out of range
                    #  part (2d),  occlusion 2 filter  out of range
                    # train  limite the iteration.  (65)  random shuffle  trainIter
                    # train:
                    # stage1  I,   super:  2D +  X_dp +simplify ( beta, pose) ,  shape( vertices wht0)
                    # stage2  +  L_depth
                    #  options:  if_depth True ,   epoch_dp  1 ,   trainIter   (100: 900) shuffle
                    # summary_step  20

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 
        'h36m_valid_protocol%d.npz' % protocol)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       S=Ss_)       # save z auto to array?


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_fd', default='/scratch/liu.shu/datasets/Human36M_raw', help='the input of the dataset')
    parser.add_argument('--out_fd', default='data/dataset_extras', required=False, help='Path to input image')
    parser.add_argument('--extract_img', action='store_true', help='if save the image')
    args = parser.parse_args()
    print('if extract img', args.extract_img)
    h36m_extract(args.ds_fd, args.out_fd, extract_img=args.extract_img)