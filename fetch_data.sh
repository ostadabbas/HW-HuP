#!/bin/bash

# Script that fetches all necessary data for training and eval
# here is the original SPIN data. You can keep them if you wanna work on other related datasets.
# Model constants etc.
wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz
# Initial fits to start training
wget http://visiondata.cis.upenn.edu/spin/static_fits.tar.gz && tar -xvf static_fits.tar.gz --directory data && rm -r static_fits.tar.gz
# List of preprocessed .npz files for each dataset
wget http://visiondata.cis.upenn.edu/spin/dataset_extras.tar.gz && tar -xvf dataset_extras.tar.gz --directory data && rm -r dataset_extras.tar.gz
# Pretrained checkpoint
wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data

# GMM prior from vchoutas/smplify0x
wget https://github.com/vchoutas/smplify-x/raw/master/smplifyx/prior.py -O smplify/prior.py

# Here is for the HW-HuP specific data files
# get the staitic fits of h36m and SLP_danaLab static fits
wget http://www.coe.neu.edu/Research/AClab/HW-HuP/h36m_fits.npy -P data/static_fits
wget http://www.coe.neu.edu/Research/AClab/HW-HuP/SLP_danaLab_fits.npy -P data/static_fits

# get the SLP annotation in npz format, npz for SPIN format,  json for the dynamic SLP reading as SLP can have flexible combination of modalities which differs from common dataset.
wget http://www.coe.neu.edu/Research/AClab/HW-HuP/SLP_danaLab_3d_dp_h36_hn0.8.json -P data/
wget http://www.coe.neu.edu/Research/AClab/HW-HuP/SLP_danaLab_SPIN_db_hn0.8.npz -P data/

# get the h36m annotation npz with the visnet detection result and 3d_dp estimation.
wget http://www.coe.neu.edu/Research/AClab/HW-HuP/h36m_vis_2.zip && unzip h36m_vis_2.zip -d data/dataset_extras && rm h36m_vis_2.zip
