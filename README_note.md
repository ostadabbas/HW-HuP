# SPIN - SMPL oPtimization IN the loop
Code repository for the paper:  
**Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop**  
[Nikos Kolotouros](https://www.seas.upenn.edu/~nkolot/)\*, [Georgios Pavlakos](https://www.seas.upenn.edu/~pavlakos/)\*, [Michael J. Black](https://ps.is.mpg.de/~black), [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)  
ICCV 2019  
[[paper](https://arxiv.org/pdf/1909.12828.pdf)] [[project page](https://www.seas.upenn.edu/~nkolot/projects/spin/)]

![teaser](teaser.png)

## todo
generate the dict for both ds.  
update the configs for folder. 
denoise the ds.  
gen the ds npz file.  joint_3d, and 3d_dp both for MIMM. 
SyRIP single for train,   
MIMM should split for train and valid parts 

after npz db,  do evaluation, then save the dict.
should train openpose in train (MPII npz has openpose item) 

## note 
The MIMM and SyRIP has filtered out the one without all major limbs, so can limit the heavy occlusion this way. 
MIMM  834 split train and valid 
Syrip last 100 for valid. 

epoch save,  each end save epoch+1, next read directly use that one.  

keypoints_3d, visible.  some depth is 0 , so  make it invisible.  

h36m_2d_ft_e4  is not result for 3dpw 

h36m vis gen
/scratch/fu.n/SPIN_slp/vis_2d_gt.py

SLP read RGB-0~1, so 


## Running note  
tensorboard should be 1.14 or higher 
can't find past, install future. 
tensorboard TBL not found, uninstall "tensorboard-plugin-wit==1.6.0.post3"   
can't write summary: utils/base_trainer.py   summary_writter(dir, flush_secs=5) 
human-body-prior 0.9.3.0 requires torch==1.1.0, but 1.1.0 pyrender not working 
depth, neural-renderer in mm 
pyrender in m  
use cdflib. 

discovery, module 
 discovery/12.02 ?
 anaconda3/2020.12  
 cuda/9.0 
 
 gpu node 
 k40m  

plotly for vis. 
'WebGLRenderingContext' error on cluster.  they use container for this.

all other mods will hold full cover version [IR, depth, PM]

result for all cover conditions will be calculated from the total result list (divide by 3 parts) 
 

## command 
eval: 
from pretrained SPIN 
`python eval1.py --name SPIN_rls --checkpoint data/model_checkpoint.pt`
or 
evaluate experiment 
`python eval1.py --name SPIN_rls `
output:  eval_rst.npz
```
|--vis
|   \-- {:05d}_f.jpg  # front view image render
     -- {:05d}_s.jpg # side view 
    -- {:05d}_s_ptc.jpg side view with point cloud (ptc). 
```  
run tensorboard

`tensorboard --logdir logs/infant_ft_e10/tensorboard/`

## code updating 
map the hip to op hip.  kp direct to 49 format. give openpose wht 1.  
3d, map pred op_hip to ori_hip. 

## files 
SLP_{}_3d_dp_h36_RGB.json  [danaLab| simLab], save the 3d dp in RGB coord. 
SLP_danaLab_SPIN_db_hn0.8.npz hr0.8 the ratio neck to head.  
To generate,  `python datasets/preprocess/get_3d_dp.py` to get the SLP json file with the `3d_dp` annotation from smoothed depth.  
`python get_dbn_SLP_SPIN.py`  generate the SPIN compatible database `.npz`. 

h36m_train_vis_2.npz  vis has visnet result in.  
2 means save every 50 frames.  

## coordinate 
pred_camera, in world coordinate [z, x, y ] 
camera_trans -> neural render
ct(flip_y) ->  pyrender # may different coordinate
 
depth recover:  
depth-> camera, (x, -y , -z_c ) in cam_rw (rotation world) 
depth_flip, y,  (x,y, -z_c)_cam
z -> align to smpl (0 centered), so z_c depth. 
cam_t flip y->  -x_c, -y_c, z_c  (x is flipped NMR)
depth - cam_t = x_cam+x_c, y_cam+y_c, 0 ( z centered, x, y in world coordinate)  

### exp 
SPIN_rls  SPIN_rls qualitative eval on SLP 
ex1     training ex original 
train_SLP_ex
h36m_2d_d<#>_e<#>  # for the h36m + mpii   depth at and epoch at    
```
|<ds>_<config> --| tensorboard
|              --|  checkpoints 

```

## Installation instructions
We suggest to use the [docker image](https://hub.docker.com/r/chaneyk/spin) we provide that has all dependencies
compiled and preinstalled. Alternatively you can create a `python3` virtual environment and install all the relevant dependencies as follows:

```
virtualenv spin -p python3
source spin/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you choose to use a virtual environment, please look at the [instructions](https://pyrender.readthedocs.io/en/latest/install/index.html) for installing pyrender. 

After finishing with the installation, you can continue with running the demo/evaluation/training code.
In case you want to evaluate our approach on Human3.6M, you also need to manually install the [pycdf package of the spacepy library](https://pythonhosted.org/SpacePy/pycdf.html) to process some of the original files. If you face difficulties with the installation, you can find more elaborate instructions [here](https://stackoverflow.com/questions/37232008/how-read-common-data-formatcdf-in-python).

## Fetch data
We provide a script to fetch the necessary data for training and evaluation. You need to run:
```
./fetch_data.sh
```
The GMM prior is trained and provided by the original [SMPLify work](http://smplify.is.tue.mpg.de/), while the implementation of the GMM prior function follows the [SMPLify-X work](https://github.com/vchoutas/smplify-x). Please respect the license of the respective works.

Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de) for training and running the demo code, while the [male and female models](http://smpl.is.tue.mpg.de) will be necessary for evaluation on the 3DPW dataset. Please go to the websites for the corresponding projects and register to get access to the downloads section. In case you need to convert the models to be compatible with python3, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

## Final fits
We also release the improved fits that our method produced at the end of SPIN training. You can download them from [here](http://visiondata.cis.upenn.edu/spin/spin_fits.tar.gz). Each .npz file contains the pose and shape parameters of the SMPL model for the training examples, following the order of the training .npz files. For each example, a flag is also included, indicating whether the quality of the fit is acceptable for training (following an automatic heuristic based on the joints reprojection error).

## Run demo code
To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file  ```im1010_shape.png``` shows the overlayed reconstruction of the model on the image.  We also render a side view, saved in ```im1010_shape_side.png```.

## Run evaluation code
Besides the demo code, we also provide code to evaluate our models on the datasets we employ for our empirical evaluation. Before continuing, please make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).

Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```

You can also save the results (predicted SMPL parameters, camera and 3D pose) in a .npz file using ```--result=out.npz```.

For the MPI-INF-3DHP dataset specifically, we include evaluation code only for MPJPE (before and after alignment). If
you want to evaluate on all metrics reported in the paper you should use the official MATLAB test code provided with the
dataset together with the saved detections.

## Run training code
Due to license limitiations, we cannot provide the SMPL parameters for Human3.6M (recovered using [MoSh](http://mosh.is.tue.mpg.de)). Even if you do not have access to these parameters, you can still use our training code using data from the other datasets. Again, make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).

Example usage:
```
python3 train.py --name train_example --pretrained_checkpoint=data/model_checkpoint.pt --run_smplify
```
You can view the full list of command line options by running `python3 train.py --help`. The default values are the ones used to train the models in the paper.
Running the above command will start the training process. It will also create the folders `logs` and `logs/train_example` that are used to save model checkpoints and Tensorboard logs.
If you start a Tensborboard instance pointing at the directory `logs` you should be able to look at the logs stored during training.

## Citing
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:

	@Inproceedings{kolotouros2019spin,
	  Title          = {Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop},
	  Author         = {Kolotouros, Nikos and Pavlakos, Georgios and Black, Michael J and Daniilidis, Kostas},
	  Booktitle      = {ICCV},
	  Year           = {2019}
	}
