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

def h36m_extract(dataset_path, out_path, c_path, protocol=1, extract_img=False, is_train=False, sample_rt=5):
    # count missing depth for joints
    num_arr = np.zeros((2, 24)) # row 1: number of missing depth, row 2: number of total frames, 24 joints
    
    # convert joints to main 17 jointx.
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    # h36m  pelvis, right h, right knee, right_ankle , left_hip...  the 17 joints in position of the global 24 jts
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]


    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, depthnames_, Ss_dp_ = [], [], [], [], [], [], []

    # users in validation set
    if is_train:
        user_list = [1, 5, 6, 7, 8] # , 5, 6, 7, 8
    else:
        user_list = [9, 11] # 11
        
    # load camera parameters
    with open(c_path) as f:
        c_para = json.load(f)
    c_intr = c_para['intrinsics']

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')
        # path with depth
        depth_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'depth_pure')   ############################# add depth and 2D
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        
        
        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort() 
    
        test = 0
        for seq_i in tqdm(seq_list, desc='processing sub {}'.format(user_i)):  # all cdf file list with folder in front?
            # sequence info
            seq_name = seq_i.split('/')[-1]     # Directions.54138969.cdf，  Discussion 1.55011271.cdf
            action, camera, _ = seq_name.split('.')     # SittingDown 2.54138969.cdf
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue
                
            # specific camera calibration parameters
            c = c_intr[camera]['calibration_matrix']
            f_x = c[0][0]
            f_y = c[1][1]
            c_x = c[0][2]
            c_y = c[1][2]
            f_d = [f_x, f_y]
            c_d = [c_x, c_y]

            # 3D pose file
            # poses_3d = pycdf.CDF(seq_i)['Pose'][0]
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]     # use another library

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = cdflib.CDF(pose2d_file)['Pose'][0] 
            
            # depth file
            depth_file = os.path.join(depth_path, seq_name.replace('cdf', 'mat'))
            with h5py.File(depth_file, 'r') as f:
                depth_h5py = [f[element[0]][:] for element in f['Feat']] #  f['Feat'] = N x { hxw }    h5 {key1:... , key2:...} feature for key , 
                # N x hxw 
            
            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            # bbox_h5py = h5py.File(bbox_file)
            with h5py.File(bbox_file, 'r') as f:
                bbox_h5py = [f[element[0]][:] for element in f['Masks']]

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))

                if is_train:
                    flag = 'train'
                else:
                    flag = 'valid'
                imgs_path = os.path.join(out_path, flag + '_images')
                if not os.path.exists(imgs_path):
                    os.makedirs(imgs_path)
                    
                depths_path = os.path.join(out_path, flag + '_depth')
                if not os.path.exists(depths_path):
                    os.makedirs(depths_path)

                vidcap = cv2.VideoCapture(vid_file)
                success, image = vidcap.read()
                print(success)
                print(image.shape)
                
            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()      # image and depth not match, possibly the 3d is not correct
                    if not success:
                        break

                # check if you can keep this frame   save frame every 25 iter
                if frame_i % sample_rt == 0 and (protocol == 1 or camera == '60457274'):  # only protol 1 is saved or else  for p2
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    '''                  
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)                  
                    '''
                    
                    # read GT bounding box
                    # mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    mask = bbox_h5py[frame_i].T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]     # should be pixel
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.  # scale is 0.9 of the bb
                    
                    # read GT 2D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partall[h36m_idx]
                    part = np.zeros([24,3])     # 24 version
                    part[global_idx, :2] = part17       # fill the 24 with the jt
                    part[global_idx, 2] = 1
                    

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.    # in m h36m17 
                    S17 = Sall[h36m_idx]    # 17 joints from original 30 jts
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24, 4])     # 24 x 4  all 0 , except the  known joint 1
                    S24[global_idx, :3] = S17   # put into 24 joints order
                    S24[global_idx, 3] = 1

                    # read depth
                    depth = depth_h5py[frame_i].T

                    # get Proxy 3D pose. Estimate from original 17 2D jts and depth
                    jt_25d_dp = np.zeros([part17.shape[0], 3])     # 17 joints
                    jt_3d_dp = np.zeros([part17.shape[0], 3])     # 17 joints
                    jt_25d_dp[:, :2] = part17
                    jt_3d_dp[:, :2] = part17
                     
                    jt_near = (part17+0.5).astype(int)
                    
                    # handle out range of image case
                    for idx in range(jt_near.shape[0]):
                        x = jt_near[idx, 0]
                        y = jt_near[idx, 1]
                        if x < depth.shape[1] and x >= 0 and y < depth.shape[0] and y >= 0:
                            jt_25d_dp[idx, 2] = depth[jt_near[idx, 1], jt_near[idx, 0]].astype(float)
                        else:
                            jt_25d_dp[idx, 2] = 0.0
                    
                    # pix2cam
                    jt_3d_dp[:, 0] = (jt_25d_dp[:, 0] - c_d[0]) / f_d[0] * jt_25d_dp[:, 2]
                    jt_3d_dp[:, 1] = (jt_25d_dp[:, 1] - c_d[1]) / f_d[1] * jt_25d_dp[:, 2]
                    jt_3d_dp[:, 2] = jt_25d_dp[:, 2] # first 3 columns
                  
                    jt_3d_dp -= jt_3d_dp[0] # root-centered
                    S24_dp = np.zeros([24, 4])     # 24 x 4  all 0 , except the  known joint 1
                    S24_dp[global_idx, :3] = jt_3d_dp/1000.   # put into 24 joints order
                    S24_dp[global_idx, 3] = 1

                    
                    # check if depth value is valid for 24 joints
                    for i in range(part.shape[0]):
                        num_arr[1, i] += 1
                        x = part[i, 0]+0.5
                        y = part[i, 1]+0.5

                        if int(x) < depth.shape[1] and int(x) >= 0 and int(y) < depth.shape[0] and int(y) >= 0:
                            if part[i, 2] != 0 and depth[int(y), int(x)] == 0.0:
                                print('invalid depth!')
                                print(i)
                                print(imgname)
                                #print(part)
                                num_arr[0, i] += 1
                                S24_dp[i, 3] = 0    # depth 0 invisible
                                # part[i, 2] = 0  # 2D only filter range in image    
                                #print(part)
                                #print(S24_dp)
                                                      
                        else:
                            print('out of range!')
                            print(i)
                            print(imgname)
                            num_arr[0, i] += 1
                            S24_dp[i, 3] = 0    # out of range invisible
                            part[i, 2] = 0    
                            
              
                    # save depth
                    depth_name = imgname[:-4] + '.npy'
                    if extract_img:
                        depth_out = os.path.join(depths_path, depth_name)
                        np.save(depth_out, depth)

                    # store data, depth, 3d_dp 
                    if is_train:
                        flag = 'train'
                    else:
                        flag = 'valid'
                    imgnames_.append(os.path.join(flag + '_images', imgname))
                    depthnames_.append(os.path.join(flag + '_depth', depth_name))
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    Ss_.append(S24)     # root centered
                    Ss_dp_.append(S24_dp)
                    
                    
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    if is_train:
        out_file = os.path.join(out_path, 'h36m_train_s{}.npz'.format(sample_rt))
    else:
        out_file = os.path.join(out_path, 'h36m_valid_protocol{}_s{}.npz'.format(protocol, sample_rt))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       S=Ss_,           # real
                       part=parts_,
                       S_dp=Ss_dp_,     # the dp version, additional dp and depthname
                       depthname=depthnames_)       # save z auto to array?
    print(num_arr)                 
    np.save(os.path.join(out_path, 'train_missing_depth.npy'), num_arr)
    print('saved!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_fd', default='/scratch/liu.shu/datasets/Human36M_raw', help='the input of the dataset')
    parser.add_argument('--c_path', default='/scratch/fu.n/SPIN_slp/h36m/camera-parameters.json', help='the path of camera parameters json file')
    parser.add_argument('--out_fd', default='data/dataset_extras', required=False, help='Path to input image')
    # parser.add_argument('--extract_img', required=False, action='store_true', help='if save the image')
    args = parser.parse_args()
    # print('if extract img', args.extract_img)
    # h36m_extract(args.ds_fd, args.out_fd, extract_img=args.extract_img)
    h36m_extract(args.ds_fd, args.out_fd, args.c_path, protocol = 1, extract_img=False, is_train=False)