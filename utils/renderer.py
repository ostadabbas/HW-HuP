import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import utils.utils as ut_t

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]       # hard to int, but I think ( res-1)/2. more accurate
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images, depth_ts=None, mask=None): # original cutoff
        # make ptc to n_bch x n_ptc x3  prepare outside, be aligned to depth
        # cutoff for the range to get rid of the ptc
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()   # to cpu
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1)) # should be -1 to 1
        rend_imgs = []
        if depth_ts is None or mask is None:
            if_ptc = False
            n_row = 2
        else:
            if_ptc = True   # if depth dn then ptc, then mask is used and valid
            depth_np = np.transpose(depth_ts.cpu().numpy(), (0, 2, 3, 1))
            n_row = 3
        for i in range(vertices.shape[0]):
            # print('image np min max', images_np[i].min(), images_np[i].max() )
            rend_img_t, valid_mask, rend_depth = self.__call__(vertices[i], camera_translation[i], images_np[i])   # call cam translationi directly
            rend_img = torch.from_numpy(np.transpose(rend_img_t, (2,0,1))).float()
            rend_imgs.append(images[i])     # original
            rend_imgs.append(rend_img)
            if if_ptc:  # if there is depth input, calc side, add to images
                # get estimated ptc
                valid_mask = valid_mask.squeeze()   # h x w
                aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0] # v * R times right?
                vt_t = vertices[i]
                center = vt_t.mean(axis=0)
                rot_vertices = np.dot((vt_t - center), aroundy) + center  # rotate body
                depth_dn = depth_np[i].squeeze()
                d_rend_valid = rend_depth[valid_mask]
                d_dn_valid = depth_dn[valid_mask]
                if len(d_rend_valid) > 0: # if there is valid point
                    # d_dn_mask = np.logical_and(depth_dn < cutoff, depth_dn > 0)
                    d_dn_mask = mask.cpu().numpy().squeeze()
                    depth_dn = depth_dn + (d_rend_valid.mean() - d_dn_valid.mean())  # the z direction,
                    # warn: mean of empty slice
                    ptc = ut_t.get_ptc_mask(depth_dn, [self.focal_length, ] * 2, mask=d_dn_mask)
                    ptc[:, 1] *= -1  # y flipped? point up?
                    trans = camera_translation[i].copy()  # ptc, x,y z to world, x, -y , -z, the ith sample
                    trans[1] *= -1  # y opposite direction.
                    ptc = ptc - trans  # camera coordinate, to world        19287 x3  -> 64 x 3
                    # get the ptc version front view
                    # img_shape, _, _ = self.__call__(vt_t, camera_translation, images_np[i],ptc=ptc)  # no front view needed
                    rot_ptc = np.dot((ptc - center), aroundy) + center
                else:
                    rot_ptc = None
                    print('non valid depth')
                img_shape_side, _, _ = self.__call__(rot_vertices, camera_translation[i], np.ones_like(images_np[i]), ptc=rot_ptc)  # white bg
                rend_img_side = torch.from_numpy(np.transpose(img_shape_side, (2, 0, 1))).float()
                rend_imgs.append(rend_img_side)

        rend_imgs = make_grid(rend_imgs, nrow=n_row)    # seems to be two columns
        return rend_imgs        #  torch images cxhxw

    def __call__(self, vertices, camera_translation, image, ptc=None):    # add additional points
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))
        camera_translation = camera_translation.copy()
        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        # add ptc if there is
        if ptc is not None:
            colors_ptc = np.vstack([np.array([0, 0, 255]),] * ptc.shape[0])
            # colors = colors.repeat([1, ptc.shape[0]]).T     # n x 3 color
            m = pyrender.Mesh.from_points(ptc, colors=colors_ptc)
            scene.add(m)

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA) # smpl only depth
        # if ptc is not None:
        #     color, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        # print('image range', image.min(), image.max())  # 0 1
        # print('color range', color.min(), color.max())
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img, valid_mask, rend_depth       # 0 ~1
