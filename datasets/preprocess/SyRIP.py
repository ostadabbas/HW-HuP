import os
from os.path import join
import sys
import json
import numpy as np
from tqdm import tqdm
# from .read_openpose import read_openpose

def coco_extract(dataset_path, out_path):
    '''
    no open pose data, SyRIP version
    :param dataset_path:
    :param out_path:
    :return:
    '''
    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]     # convert to 17 joints
    # joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]     # convert to 17 joints
    # right left hip  2, 3  , visible to 0 for
    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 
                             'annotations/200R_1000S/',
                             'person_keypoints_train_infant.json')
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img       # {0: {'file_name':xx, 'RGB':....}
    n_chk = -1
    for i, annot in tqdm(enumerate(json_data['annotations']), desc='gen SyRIP db for SPIN...'):
        # keypoints processing
        if n_chk>0 and i>=n_chk:
            break
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated
        if sum(keypoints[5:,2]>0) < 12:
            continue
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join('images/1200/', img_name)      # relative from ds folder to images
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints        # 24 joints, put the gt 17 in, 2, 3 vis to 0 , add openpose jt
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200
        # read openpose detections, no read openpose
        # json_file = os.path.join(openpose_path, 'coco',
        #     img_name.replace('.jpg', '_keypoints.json'))
        # openpose = read_openpose(json_file, part, 'coco')
        # update only the  hip to openpose , then clean the part vis
        openpose = np.zeros([25, 3])
        # r,l hip  op 9, 12  ,  part 2, 3
        openpose[9] = part[2]
        openpose[12] = part[3]
        part[[2,3], 2] = 0     # clean up the vis for hip
        # debug show
        if not n_chk<0:
            print('id {} op and part'.format(i))
            print(openpose)
            print(part)
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)             # gt 17
        openposes_.append(openpose)     # openpose  25 correct detection

    # store the data struct
    print('valid data length', len(imgnames_))
    if n_chk<0:
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'SyRIP_train.npz')
        print("file saved to {}".format(out_file))
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           part=parts_,
                           openpose=openposes_)


def SyRIPv2_extract(dataset_path, out_path, is_train=False):
    '''
    coco format, SyRIP version
    :param dataset_path:
    :param out_path:
    :return:
    '''
    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]  # convert to 17 joints
    # joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]     # convert to 17 joints
    # right left hip  2, 3  , visible to 0 for
    # bbox expansion factor
    scaleFactor = 1.2

    # json annotation file
    if is_train:
        split = 'train'
    else:
        split = 'valid'

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path,
                             'anno_{}.json'.format(split))
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img  # {0: {'file_name':xx, 'RGB':....}
    n_chk = -1
    N= len(imgs)
    for i, annot in tqdm(enumerate(json_data['annotations']), desc='gen SyRIP db for SPIN...', total=N):
        # keypoints processing
        if n_chk > 0 and i >= n_chk:
            break
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17, 3))
        keypoints[keypoints[:, 2] > 0, 2] = 1
        # check if all major body joints are annotated
        if sum(keypoints[5:, 2] > 0) < 12:      # if not all joints visible.
            continue
        # image name
        # image_id = annot['image_id']
        image_id = annot['id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join('RGB', img_name)  # relative from ds folder to images
        # keypoints
        part = np.zeros([24, 3])
        part[joints_idx] = keypoints  # 24 joints, put the gt 17 in, 2, 3 vis to 0 , add openpose jt
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        scale = scaleFactor * max(bbox[2], bbox[3]) / 200

        openpose = np.zeros([25, 3])
        # r,l hip  op 9, 12  ,  part 2, 3
        openpose[9] = part[2]
        openpose[12] = part[3]
        part[[2, 3], 2] = 0  # clean up the vis for hip
        # debug show
        if not n_chk < 0:
            print('id {} op and part'.format(i))
            print(openpose)
            print(part)
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)  # gt 17
        openposes_.append(openpose)  # openpose  25 correct detection

    # store the data struct
    print('valid data length', len(imgnames_))
    if n_chk < 0:
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'SyRIP_{}.npz'.format(split))
        print("file saved to {}".format(out_file))
        np.savez(out_file, imgname=imgnames_,
                 center=centers_,
                 scale=scales_,
                 part=parts_,
                 openpose=openposes_)


if __name__ == '__main__':
    # coco_extract('/scratch/liu.shu/datasets/SyRIP', '/scratch/liu.shu/codesPool/SPIN/data/dataset_extras')
    SyRIPv2_extract('/scratch/liu.shu/datasets/SyRIPv2', '/scratch/liu.shu/codesPool/SPIN/data/dataset_extras', is_train=True)