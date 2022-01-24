from utils import TrainOptions
from train import Trainer
import warnings
import os.path as osp
import numpy as np
warnings.filterwarnings("ignore")

def fits_from_epoch():
    # run a single epoch to save the fit to the  experiment folder
    options = TrainOptions().parse_args()
    options.run_smplify = True  # must ran simplify to get better shape and pose
    trainer = Trainer(options, phase='all')
    trainer.get_static_dict()  # train only one epoch will update the  dict

def gen_zero_fits(dsNm, N):
    '''
    gen the pure zero matrix by given the name and length
    :param name:
    :param N:   how many items
    :return:
    '''
    pth = osp.join('data/static_fits', dsNm+'_fits.npy')
    fits = np.zeros([N, 82])
    np.save(pth, fits)
    print('Static fits saved to {}'.format(pth))

if __name__ == '__main__':

    # trainer.train()
    # trainer.summary_writer.close()
    gen_zero_fits('SyRIP', 1700)
    gen_zero_fits('MIMM', 1050)