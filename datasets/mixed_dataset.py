"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp'] # full set
        if options.train_ds == 'h36m_2d':

            self.dataset_list = [
                'h36m',
                # 'lsp-orig',
                'mpii',     # only lsp fit the 2d
                # 'lspet',
                # 'coco',
                # 'mpi-inf-3dhp'
            ]
            self.dataset_dict = {
                'h36m': 0,
                # 'lsp-orig': 1,
                'mpii': 2,
                # 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5
            }

        elif options.train_ds == 'h36m':
            self.dataset_list = [
                'h36m',
            ]
            self.dataset_dict = {
                'h36m': 0,
            }
        elif options.train_ds == 'infant':
            self.dataset_list = [
                'MIMM',
                'SyRIP'
            ]
            self.dataset_dict = {
                'MIMM': 6,
                'SyRIP': 7
            }
        else:
            raise KeyError('not implemented train ds {}'.format(options.train_ds))
        print('{} ds with dataset'.format(options.train_ds), self.dataset_list)

        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]    # has_3d single for ds
        total_length = sum([len(ds) for ds in self.datasets])

        self.length = max([len(ds) for ds in self.datasets])        # longest ds
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        # original ds partition
        # self.partition = [.3, .6*len(self.datasets[1])/length_itw,
        #                   .6*len(self.datasets[2])/length_itw,
        #                   .6*len(self.datasets[3])/length_itw,
        #                   .6*len(self.datasets[4])/length_itw,
        #                   0.1]
        # single ds
        # self.partition = [1.0]        # original full

        if len(self.dataset_list)<2:        # single ds
            self.partition = [1.0]  # original full
        else:
            length_itw = sum([len(ds) for ds in self.datasets[1:]])
            self.partition = [.4, .6 * len(self.datasets[1]) / length_itw]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(6):
            if p <= self.partition[i]:  # cumulative, first time past, the  partition
                return self.datasets[i][index % len(self.datasets[i])]  #  the ds item

    def __len__(self):
        return self.length
