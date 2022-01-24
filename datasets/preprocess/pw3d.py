import os
import os.path as osp
import cv2
import numpy as np
import pickle
import argparse
import sys
sys.path.append(osp.join(sys.path[0], '..', '..'))  # relative to current one
# print(sys.path)
import utils.utils as ut
from tqdm import tqdm

'''
history:
210924: update the part to global 24 index with visibility  
'''
def pw3d_extract(dataset_path, out_path):

    # scale factor
    scaleFactor = 1.2
    if_dbg = 0

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_, genders_ = [], [], []
    parts_coco_ = []
    parts_ = []

    # coco to 24
    # global_idx = [19, 12, 8, 7, 6, 9, 10, 11, 2, 1, 3, 11, 12, 13, 21, 20, 23, 22]  # the openpose order
    global_idx = []  # the openpose order
    idx_valid = [8,5,2,1,4,7,21,19,17, 16, 18, 20, 12, 15, 0]         # the valid joint

    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'test')
    files = [os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) if f.endswith('.pkl')]    # maybe the file randomeness

    # go through all the .pkl files

    # get regressor

    for filename in tqdm(files):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']       # N x 72
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool) # indicate which cam pose aligned to image , maybe 0 failed
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            cam_intrinsics = data['cam_intrinsics']
            jointPositions = data['jointPositions']    #  24 x 3  xyz

            # get through all the people in the sequence
            # print('smpl pose shape', smpl_pose.shape)
            for i in range(num_people):
                if if_dbg and i>0:
                    break
                valid_pose = smpl_pose[i][valid[i]] #
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                # valid_cam_intrinsics = cam_intrinsics[valid[i]]
                # print('valid[i] shape', valid[i].shape)
                valid_j3ds = jointPositions[i][valid[i]]   # scalar indices only， jp dim0 dim 2?
                # print('valid pose shape', valid_pose.shape) # (939,u72)

                gender = genders[i]
                # consider only valid frames
                for valid_i in range(valid_pose.shape[0]):
                    part = valid_keypoints_2d[valid_i,:,:].T
                    j3d = valid_j3ds[valid_i].reshape([-1, 3])[idx_valid]   # to the 24 x3 format , to only 15 valid jts

                    if if_dbg and valid_i> 2:
                        break

                    # 2D global, open ose version
                    # part_openpose = part.copy()     # openpose 17  order
                    # part_g = np.zeros([24, 3])  # 24 version
                    # part_g[global_idx] = part_openpose  # fill the 24 with the jt
                    # part_g[global_idx, 2] = 1

                    part = part[part[:,2]>0,:]  # filterd the empty one
                    bbox = [min(part[:,0]), min(part[:,1]),
                        max(part[:,0]), max(part[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                    
                    # transform global pose
                    pose = valid_pose[valid_i]
                    # print('pose shape', pose.shape)
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    T = valid_global_poses[valid_i][:3, -1]     #
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]        # only change global

                    # 2D projects
                    j3d_mono = np.einsum('ij,kj->ki', extrinsics[:3, :3], j3d) + T # 24 x3  -> 2
                    # j3d_mono = j3d  # assume already in camera space
                    j2d = np.einsum('ij,kj->ki', cam_intrinsics, j3d_mono)
                    j2d = j2d[:,:2]/j2d[:, 2:]  # las dim
                    part_g = np.zeros([24, 3])  # assume all visible
                    part_g[:15,:2] = j2d      # replace
                    part_g[:15,2] = 1       # all visible for 1st 15

                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)
                    # parts_coco_.append(part_coco)
                    parts_.append(part_g)

    print('totally {} processed'.format(len(imgnames_)))
    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        '3dpw_test.npz')
    if not if_dbg:   # debug not save
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           pose=poses_,
                           shape=shapes_,
                           gender=genders_,
                           part = parts_,
                 # part_coco=parts_coco_,
                 )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_fd', default='/home/liu.shu/datasets/3DPW', help='the input of the dataset')
    parser.add_argument('--out_fd', default='data/dataset_extras', help='Path to input image')
    args = parser.parse_args()

    pw3d_extract(args.ds_fd, args.out_fd)
