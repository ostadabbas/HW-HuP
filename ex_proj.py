from utils import TrainOptions
# from train import Trainer
import json
import numpy as np
import os.path as osp
from datasets import SLP_RD, SPIN_ADP
from datasets.base_dataset import BaseDataset
from tqdm import tqdm

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    # trainer = Trainer(options)
    # trainer.train()

    # pth_3d_dp = 'data/SLP_danaLab_3d_dp_h36.json'
    # with open(pth_3d_dp, 'r') as f:
    #     dataIn= np.array(json.load(f))
    #     f.close()
    # print('dataIN shpae', dataIn.shape)

    ## SLP check
    # class opt_t:
    #     SLP_fd = ''
    #     sz_pch = 224
    #     fc_depth = 50.  # down scale depth
    #     cov_li = ['uncover',
    #               'cover1',
    #               # 'cover2'
    #               ]   # process only uncovered
    # labNms = [
    #     'danaLab',
    #     # 'simLab'
    # ]
    # for labNm in labNms:
    #     opt_t.SLP_fd = osp.join('/home/liu.shu/datasets', 'SLP', labNm)
    #     SLP_rd = SLP_RD(options, phase='all', if_3d_dp=True, rt_hn=options.ratio_head_neck)
    #     # SLP_fd = SPIN_ADP(options, 'SLP', SLP_rd, ignore_3d=options.ignore_3d, is_train=True)
    #     # dt = SLP_fd[0]
    #     img, jt, bb = SLP_rd.get_array_joints(0, mod='RGB')
    #     jt = jt* 224/1024
    #     print('neck head', jt[-2:])     # 63,50 ;  63, 27.4
    #     h36_hd = np.array([111.78341, 29.433624])
    #     rt = (h36_hd[1] - jt[12,1])/(jt[13,1]-jt[12,1])
    #     print('h36m head ratio is', rt)


        # print(dt['keypoints'])
        # print(dt['pose_3d'])
        # print(dt['has_smpl'])
        # print(dt['has_depth'])
        # print(dt['scale'])
        # print(dt['center'])
        # print(dt['orig_shape'])
        # print(dt['is_flipped'])
        # print(dt['rot_angle'])
        # print(dt['gender'])
        # print(dt['sample_index'])
        # print(dt['dataset_name'])
        # print('check {}'.format(labNm))
        # print('img name first 10', SLP_rd.db_SPIN['imgname'][:10])
        # print('pthDesc li', SLP_rd.pthDesc_li[:10])
        # print('the 3d dp is')
        # print(SLP_rd.li_joints_3d_dp[0][0])
        # RGB, jt, bb = SLP_rd.get_array_joints(0, mod='RGB')
        # for i in tqdm(range(SLP_rd.n_smpl), 'SLP sanity check for depth_dn to RGB'):
        #     depth_dn = SLP_rd.get_array_A2B(i, 'RGB', 'RGB')
        # print('RGB min, max', RGB.min(), RGB.max())


    # check the static fits
    # fNm = 'lspet_fits.npy'
    # pth = 'logs/st_fit/checkpoints/SLP_danaLab_fits.npy'
    # dt_in = np.load(pth)
    # fNm = 'SLP_danaLab_fits.npy'
    # dt_in = np.load(osp.join('data', 'static_fits', fNm))
    # print('dt shape', dt_in.shape)
    # print(dt_in[:3])

    # fits_0 = np.zeros([102*45, 82])
    # pth_fits = osp.join('data', 'static_fits', 'SLP_danaLab_fits.npy')  # only for dana full set
    # np.save(pth_fits, fits_0)

    ## check base ds
    ds = BaseDataset(options, 'SyRIP', use_augmentation=False, is_train=True)       # test 98 train 1588?  filtered
    print('ds len', len(ds))
    # rst = ds[0]
    # rst = ds[5]
    # rst = ds[10]
    # rst = ds[15]
    # rst = ds[5]

    # ids = [0,
    #        20,30,40,50,60
    #        ]
    # for id in ids:
    #     rst = ds[id]