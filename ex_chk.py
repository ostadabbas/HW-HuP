'''
check certain data
'''
# import cdflib
import os.path as osp
import torch

## h36m cdf
# fd = r'S:\ACLab\datasets\human3.6M\sub1\S1\MyPoseFeatures'
# # fdd = 'D3_Positions_mono'
# fdd = 'D2_Positions'
# file = 'Directions 1.54138969.cdf'
# poses_3d = cdflib.CDF(seq_i)['Pose'][0]     # use another library
# seems to be dict
# pth = osp.join(fd, fdd, file)
# dt_in = cdflib.CDF(pth)
# pose = dt_in['Pose'].reshape((-1,3))
# print('pose shape', pose.shape)     # 3d_mono 44256 x3    2d  44256 x2  no visible infor
# print("cdf infor", dt_in.attinq(attribute=None))  # not working
# print('dt_in', dict(dt_in).items())


## check the the epoch number of check point
# pth = 'data/model_checkpoint.pt'
# pth = 'logs/h36m_2d_d10_e12/checkpoints/2021_03_17-02_34_48.pt'   # fei's
# pth = 'logs/h36m_2d_d10_e12/checkpoints/2021_03_17-23_40_33.pt' # just to 10 is good e10 iter 97
# pth = 'logs/h36m_2d_d10_e12/checkpoints/2021_03_17-23_11_46.pt' # just to 10 is good , e10 iter 0
# pth = 'logs/h36m_2d1_d35_e40/checkpoints/2021_03_22-03_33_52.pt' #  34 iter 320
# pth = 'logs/h36m_2d1_d35_e40/checkpoints/2021_03_22-05_44_25.pt' #  35 iter 320
# pth = 'logs/infant_camL_ft_e130/checkpoints/2021_04_20-11_50_28.pt'
pth = 'logs/infant_camL_ft_e540/checkpoints/2021_04_21-05_15_11.pt'
checkpoint = torch.load(pth)
print("{} at epoch {}, iter {}".format(pth, checkpoint['epoch'], checkpoint['batch_idx']))

## local read the depth txt compare
# fd = r'S:\ACLab\datasets\MIMM_demo'
# rgb_nm = 'train00001.jpg'
