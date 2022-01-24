'''
generate the image crops for later combination.
logs/SLP_demos/RGB to save the crops
'''
from utils import TrainOptions
# from train import Trainer
import json
import numpy as np
import os.path as osp
from datasets import SLP_RD, SPIN_ADP
from tqdm import tqdm
import utils.utils as ut_t
import os
import os.path as osp
import cv2
from utils import vis
import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

# full loop with all cover condition in sequence
def get_SLP_crops(if_sMod=False, if_stk=False):
    '''
    single mod will be saved to 224.
    stk will map the depth coordiante
    :param if_sMod:
    :param if_stk:
    :return:
    '''
    options = TrainOptions().parse_args()

    ## SLP check
    cov = 'cover2'  # each time only single modalities

    class opt_t:
        SLP_fd = ''
        sz_pch = 224
        fc_depth = 50.  # down scale depth
        cov_li = ['uncover',
                  'cover1',
                  'cover2'
                  ]  # process only uncovered

    labNms = [
        'danaLab',
        # 'simLab'
    ]
    fd = 'logs/SLP_demos'
    # for local
    # fd = r'S:\ACLab\rst_model\HWS\SLP_demos'
    # nms = ['RGB', 'depth', 'IR', 'PM', 'RGB-jt']
    nms = ['RGB', 'depth', 'IR', 'PM']
    # if if_stk:
    #     nms = nms[1:]
    # nms = ['_'.join((nm, cov)) for nm in nms]
    li_fdd = []
    for nm in nms:  # make all the sub folders
        fdd = osp.join(fd, nm)
        li_fdd.append(fdd)
        if not osp.exists(fdd):
            os.makedirs(fdd)

    skl_lr = [
        [[3, 4], [4, 5], [9, 10], [10, 11]],
        [[2, 1], [1, 0], [8, 7], [7, 6]],
    ]
    color = [
        # (0, 255, 255),
        (0, 255, 0),
        (0, 0, 255)]  # B,R
    i_break = -1  # debug for break
    tarMod = 'RGB' # to generate in what coordinate
    for labNm in labNms:
        opt_t.SLP_fd = osp.join('/home/liu.shu/datasets', 'SLP', labNm)
        # opt_t.SLP_fd = osp.join(r'S:\ACLab\datasets', 'SLP', labNm)
        SLP_rd = SLP_RD(opt_t, phase='test', if_3d_dp=False, rt_hn=options.ratio_head_neck, if_SPIN_db=False)

        N = SLP_rd.n_smpl
        # N = 2   # for quick  train
    for i in tqdm(range(N), 'saving out SLP jt patches...'):
            if i_break > 0 and i >= i_break:
                break
            RGB, jt, bb = SLP_rd.get_array_joints(i, 'RGB')  # square bb
            jt_2d = jt[:, :2].copy()
            jt_vis = np.ones(len(jt_2d))

            IR = SLP_rd.get_array_A2B(i, 'IR', tarMod)
            depth = SLP_rd.get_array_A2B(i, 'depth', tarMod)
            PM = SLP_rd.get_array_A2B(i, 'PM', tarMod)
            IR_c = cv2.applyColorMap(IR, getattr(cv2, SLP_rd.dct_clrMap['IR']))
            depth_c = cv2.applyColorMap(depth, getattr(cv2, SLP_rd.dct_clrMap['depth']))
            PM_c = cv2.applyColorMap(PM, getattr(cv2, SLP_rd.dct_clrMap['PM']))
            # make stk make RGB gt  # not working for the   , combine later
            # pth= osp.join(fd, 'stk_{}'.format(cov), '{:05d}.jpg'.format(i))
            # stk = vis.stk_IR_D_PM(depth, IR, PM, pth=pth)    # stk image save out directly
            if if_stk:
                pth = osp.join(fd, 'stk', '{:05d}.jpg'.format(i))
                vis.stk_IR_D_PM(depth, IR, PM, pth=None)

            li_img = [
                # RGB[:,:,::-1], depth, IR, PM, RGB[:,:,::-1]
                RGB[:,:,::-1],
                # depth_c, IR_c, PM_c      # RGB to BGR
            ]
            # all images,  aligned, draw the l, r limbs,  then gen patch , save
            if if_sMod:
                for j, img in enumerate(li_img):  # j mod
                    img, trans = ut_t.generate_patch_image(img, bb, 0, 1, 0, 0, sz_std=(224, 224))  # to RGB
                    if j == 4:  # only make RGB
                        for k in range(len(jt_2d)):  # 2d first for boneLen calculation
                            jt_2d[k, 0:2] = ut_t.trans_point2d(jt_2d[k, 0:2], trans)  # to pix:patch under input_shape size
                        img = vis.vis_kps(img, jt_2d, jt_vis, skl_lr, color)    # back to BGR
                    img = img[:, :, ::-1].astype(np.uint8)  # float RGB to u8 bgr
                    pth = osp.join(li_fdd[j], '{:05d}.jpg'.format(i))  # j pth, ith image
                    cv2.imwrite(pth, img)

def get_SLP_stk():
    '''
    get the SLP stacked vesion of all modalities. all to stck with the same names as the.
    can't work on dis.  distored with x elongated.
    xxx not working from the plotly
    :return:
    '''
    # fd = 'logs/SLP_demos' # discovery
    fd = r'S:\ACLab\rst_model\HWS\SLP_demos'
    fdd = osp.join(fd, 'stk')
    if not osp.exists(fdd):
        os.makedirs(fdd)

    mods = ['depth', 'IR', 'PM']
    file_li = os.listdir(osp.join(fd, 'depth'))
    # print(len(file_li))
    N = len(file_li)
    N= 1   # for test purpose, test one single directly show it
    for i in tqdm(range(N), desc='combining into stk'):
        nm = '{:05d}.jpg'.format(i)
        img_li = []
        for mod in mods:
            pth = osp.join(fd, mod, nm)
            # print('path is ', pth)
            img = cv2.imread(pth)
            # print('img shape', img.shape)
            img_li.append(img)
        out_pth = osp.join(fdd, nm)
        vis.stk_IR_D_PM(img_li[0], img_li[1], img_li[2], bb=[0,0,224,224], pth=None)        # plotly can't stack color

def rgb_processing(self, rgb_img, center, scale, rot, flip, pn=None):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale,
                   [constants.IMG_RES, constants.IMG_RES], rot=rot)
    # flip the image
    if flip:
        rgb_img = flip_img(rgb_img)
    if pn is not None:  # will only affect RGB as depth no pn , pixel noise
        for i in range(rgb_img.shape[2]):  # int not subscrible
            rgb_img[:, :, i] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, i] * pn[i]))
    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
    return rgb_img

def get_ds_crops(ds_nm='MIMM', split='valid', crop_suffix='', freq=1):
    '''
    from the ds name, read the db,  get RGB, crop, plot limb, save to logs/dsNm_crop
    :param ds_nm:
    :return:
    '''
    db_pth = 'data/dataset_extras/{}_{}.npz'.format(ds_nm, split)
    data = np.load(db_pth)
    imgnames = data['imgname']
    scales = data['scale']
    centers = data['center']
    keypoints_gts = data['part']     # 2d, update hip to visible
    img_fd = config.DATASET_FOLDERS[ds_nm]

    out_fd = osp.join('logs/{}_crops'.format(ds_nm))
    if crop_suffix:
        out_fd = out_fd+'_'+crop_suffix
    if not osp.exists(out_fd):
        os.makedirs(out_fd)
    n = len(imgnames)        # sample numbers
    # n = 100  # for test purpose
    # df the skeleton and the colors, left and right
    skl_lr = np.array([
        [[3, 4], [4, 5], [9, 10], [10, 11]],
        [[2, 1], [1, 0], [8, 7], [7, 6]],
    ])
    color = [
        # (0, 255, 255),
        (0, 255, 0),
        (0, 0, 255)]  # B,R


    for i in tqdm(range(0, n, freq), desc='gen {} crops...'.format(ds_nm)):
        imgNm = osp.join(img_fd, imgnames[i])
        img = cv2.imread(imgNm)[:, :, ::-1].copy().astype(np.float32)   # RGB?
        orig_shape = np.array(img.shape)[:2]  # iamge ref before assignment
        s = scales[i]
        c = centers[i]
        rgb_img = crop(img, c, s,
                       [constants.IMG_RES, constants.IMG_RES], rot=0)
        kpt = keypoints_gts[i].copy()
        kpt[[2, 3], 2] = 1      # update the hip to visible
        # print('kpt shape', kpt.shape)
        for j in range(kpt.shape[0]):   # outside the 24
            kpt[j, 0:2] = transform(kpt[j, 0:2] + 1, c, s,      # 24 out side 24
                               [constants.IMG_RES, constants.IMG_RES], rot=0)
        img = vis.vis_kps(rgb_img, kpt[:,:2], kpt[:,2], skl_lr, color)  # back to BGR
        img = img[:, :, ::-1].astype(np.uint8)  # float RGB to u8 bgr
        pth = osp.join(out_fd, '{:05d}.jpg'.format(i))
        cv2.imwrite(pth, img)

if __name__ == '__main__':
    # get_SLP_crops(if_sMod=True) # to get crop save single modes
    # get_SLP_stk()
    # get_ds_crops(ds_nm='SyRIP')
    # get_ds_crops(ds_nm='3dpw', split='test')
    # get_ds_crops(ds_nm='h36m', split='valid_vis_2', crop_suffix='s50')     # first part used to gen the crop names,, original s50
    get_ds_crops(ds_nm='h36m', split='valid_protocol1_s5', crop_suffix='s5')     # first part used to gen the crop names,