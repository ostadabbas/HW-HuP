"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = '/scratch/liu.shu/datasets/Human36M_raw/'
LSP_ROOT = ''
LSP_ORIGINAL_ROOT = ''
LSPET_ROOT = ''
MPII_ROOT = '/scratch/liu.shu/datasets/MPII/'
COCO_ROOT = ''
MPI_INF_3DHP_ROOT = ''
PW3D_ROOT = '/scratch/liu.shu/datasets/3DPW/'
UPI_S1H_ROOT = ''
SLP_danaLab = '/scratch/liu.shu/datasets/SLP/danaLab'
SLP_simLab = '/scratch/liu.shu/datasets/SLP/simLab'
# SYRIP_ROOT = '/scratch/liu.shu/datasets/SyRIP'
SYRIP_ROOT = '/scratch/liu.shu/datasets/SyRIPv2'
MIMM_ROOT = '/scratch/liu.shu/datasets/MIMMv2'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {
    # 'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
    'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_vis_2.npz'),  # 50 hz
    # 'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1_s5.npz'),  # 5 hz
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                    'SyRIP': join(DATASET_NPZ_PATH, 'SyRIP_valid.npz'),
                    'MIMM': join(DATASET_NPZ_PATH, 'MIMM_valid.npz')
                  },

                  {     # for train
                      # 'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                      'h36m': join(DATASET_NPZ_PATH, 'h36m_train_vis_2.npz'), # 50 hz
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                      'SyRIP': join(DATASET_NPZ_PATH, 'SyRIP_train.npz'),
                      'MIMM': join(DATASET_NPZ_PATH, 'MIMM_train.npz')
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                   'SyRIP': SYRIP_ROOT,
                   'MIMM': MIMM_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
