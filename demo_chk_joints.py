"""
simply use neutral pose and shape (T shape) to show joints locations.
user input:
to indicate the SMPL or SMIL,
indicate the joint index according to the constants
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation



parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str,default='examples/image_000002.png', help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

if __name__ == '__main__':
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model, point to the SMIL model
    # smpl_fd = config.SMPL_MODEL_DIR # original
    smpl_fd = 'data/smpl/SMIL.pkl'      # point to infant directly
    smil = SMPL(smpl_fd,
                batch_size=1,
                create_transl=False).to(device)
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)


    # Preprocess input image and generate predictions
    img, norm_img = process_image(args.img, args.bbox, args.openpose, input_res=constants.IMG_RES)  # 0~1,  c first, channel to RGB
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        pred_rotmat[:, 1:,:] = torch.eye(3).cuda()     # keep global rot, need global
        pred_betas[:] = 0  # neutral shape
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)  # root centered rot only
        pred_vertices = pred_output.vertices    # 0 centered
        pred_joints = pred_output.joints

        pred_output_smil = smil(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1),
                           pose2rot=False)  # root centered rot only
        pred_vertices_smil = pred_output_smil.vertices  # 0 centered
        pred_joints_smil = pred_output_smil.joints
        
    # Calculate camera parameters for rendering
    # print("pred cam is", pred_camera)       # 0.86, 0.02,  0.13
    # pred_camera[0,0] = 2.4 # make nearer,  3 times larger
    # pred_camera[0,2] = -0.13 # make nearer y more distance , pyrender y up?
    # change later for smil
    # pelvis version
    # t_smil = torch.tensor([0.017, 0, -0.451]).cuda()
    # s_smil = 2.715
    # torso center version
    # t_smil = torch.tensor([0.05, 0, -0.58]).cuda()
    # s_smil = 2.87
    # neck version
    # t_smil = torch.tensor([0.78, 0, -0.75]).cuda()
    # s_smil = 2.715
    # 14 jt
    t_smil = torch.tensor([0.05, 0, -0.46]).cuda()      # z is scale should not touch
    s_smil = 2.75
    # even jt , no elbow and hands knees
    # t_smil = torch.tensor([0.04, 0, -0.44]).cuda()
    # s_smil = 2.78


    # pred_camera_smil = pred_camera + t_smil   # s, x, y
    pred_camera_smil = pred_camera.clone()   # s, x, y
    pred_camera_smil[:,2] += t_smil[2]   #  only change y no others
    pred_camera_smil[:, 0] *= s_smil

    print('pred camera', pred_camera)
    print('pred camera smil', pred_camera_smil)

    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1) # metric ,  pred_cam [-1, 1] res range
    camera_translation = camera_translation[0].cpu().numpy()
    camera_translation_smil = torch.stack([pred_camera_smil[:, 1], pred_camera_smil[:, 2],
                                      2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera_smil[:, 0] + 1e-9)],
                                     dim=-1)  # metric ,  pred_cam [-1, 1] res range
    camera_translation_smil = camera_translation_smil[0].cpu().numpy()

    pred_vertices = pred_vertices[0].cpu().numpy()
    jts_smpl = pred_joints[0].cpu().numpy()
    pred_vertices_smil = pred_vertices_smil[0].cpu().numpy()
    jts_smil = pred_joints_smil[0].cpu().numpy()

    # r,l hip  27,28 , r,l shd,  33,34,  neck 37,
    # pv_smpl = (jts_smpl[27] + jts_smpl[28])/2.
    # pv_smpl = jts_smpl[[27,28,33,34]].mean(axis=0)  # torso
    # pv_smpl = jts_smpl[[27,28]].mean(axis=0)  # pelvis
    # idx_even= [25,27,28,30, 33,34,37,38]    # no elbow , hand, knee
    idx_even= np.arange(25, 39)   # no elbow , hand, knee
    # pv_smpl = jts_smpl[25:39].mean(axis=0)    # 14 center too much arm higher
    pv_smpl = jts_smpl[idx_even].mean(axis=0)    # 14 center too much arm higher
    # pv_smpl = jts_smpl[[25:31,33,34,37,38]].mean(axis=0)    # 14 center
    # nk_smpl = jts_smpl[37]
    nk_smpl = jts_smpl[38]
    l_smpl= np.linalg.norm(nk_smpl - pv_smpl)
    # pv_smil = jts_smil[[27, 28, 33, 34]].mean(axis=0)
    # pv_smil = jts_smil[[27, 28]].mean(axis=0)
    # pv_smil = jts_smil[25: 39].mean(axis=0)
    pv_smil = jts_smil[idx_even].mean(axis=0)
    # nk_smil = jts_smil[37]
    nk_smil = jts_smil[38]      # use head
    l_smil = np.linalg.norm(nk_smil - pv_smil)

    print("torso and neck for smil and smpl")
    print(pv_smil, nk_smil)
    print(pv_smpl, nk_smpl)
    print('smil to smpl', pv_smpl-pv_smil)
    # print('smil to smpl, nk', nk_smpl-nk_smil)
    print("smil to smpl, scale", l_smpl/l_smil)

    img = img.permute(1,2,0).cpu().numpy()      # img 0 ~1


    # for 2d part
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)], dim=-1)
    camera_center = torch.zeros(batch_size, 2, device=device)  # center 0
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size,
                                                                                                             -1, -1),
                                               translation=pred_cam_t,
                                               focal_length=constants.FOCAL_LENGTH,
                                               camera_center=camera_center)
    pred_keypoints_2d_t = pred_keypoints_2d.detach().cpu().numpy()[0] + constants.IMG_RES/2.  # 25 + 13 = headtop 25+17 = jaw 25 + 18 = head, single image

    img_shape, _, _ = renderer(pred_vertices, camera_translation, np.ones_like(img))        # will be 0 ~ 1       with mask and depth
    img_shape_smil, _, _ = renderer(pred_vertices_smil, camera_translation_smil, np.ones_like(img))        # will be 0 ~ 1       with mask and depth

    # Render side views
    # aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]       # x, y ,z   z point to body direction?
    # center = pred_vertices.mean(axis=0)
    # rot_vertices = np.dot((pred_vertices - center), aroundy) + center   # rotate body
    # # Render non-parametric shape
    # img_shape_side, _, _ = renderer(rot_vertices, camera_translation, np.ones_like(img))  # white bg

    idx_care = np.arange(25, 39)  # rh_op, rs, thorax
    idx_care = np.concatenate([idx_care, np.arange(44, 49)])  # 14 joints (lsp) +  5 nose eye
    img_shape = 255 * img_shape[:, :, ::-1]   # to BGR255
    img_shape_smil = 255 * img_shape_smil[:, :, ::-1]   # to BGR255
    clr = (255, 0, 0)
    for i in idx_care:
        x = int(pred_keypoints_2d_t[i, 0])
        y = int(pred_keypoints_2d_t[i, 1])
        cv2.circle(img_shape, (x, y), 1, clr, thickness=-1)
    # Save reconstructions
    cv2.imwrite('examples/smpl_shape.png', img_shape)
    cv2.imwrite('examples/smil_shape.png', img_shape_smil)
    # cv2.imwrite(outfile + '_shape.png', img_shape)
    # cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
