import torch
import numpy as np
import neural_renderer as nr
import config

from models import SMPL

class PartRenderer():
    """Renderer used to render segmentation masks and part segmentations.
    Internally it uses the Neural 3D Mesh Renderer
    """
    def __init__(self, focal_length=5000., render_res=224):
        # Parameters for rendering
        self.focal_length = focal_length
        self.render_res = render_res
        # We use Neural 3D mesh renderer for rendering masks and part segmentations
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=render_res,
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=False)
        self.faces = torch.from_numpy(SMPL(config.SMPL_MODEL_DIR).faces.astype(np.int32)).cuda()
        textures = np.load(config.VERTEX_TEXTURE_FILE)      # possibly color for each vertices for seg? 5 dim 2x2x2 for the linear intepolation of the face P=xA+yB+zC ( the coordinate of 3 vertices)
        self.textures = torch.from_numpy(textures).cuda().float()
        self.cube_parts = torch.cuda.FloatTensor(np.load(config.CUBE_PARTS_FILE))   #  what is cube parts

    def get_parts(self, parts, mask):
        """Process renderer part image to get body part indices."""
        bn,c,h,w = parts.shape
        mask = mask.view(-1,1)      # the color and alpha, flattened mask pixels
        parts_index = torch.floor(100*parts.permute(0,2,3,1).contiguous().view(-1,3)).long() #  n_pix x 3, 3 channel for 3D index of the cub?, the texuture will be encoding color for cube coordinate, then map the code to  seg 6 classes index?
        parts = self.cube_parts[parts_index[:,0], parts_index[:,1], parts_index[:,2], None] # 100 x 92 x 17 x 3  into r,g,b group? each limb grid into cube, then each point in a cube belongs to ?
        parts *= mask       # 0 or 1 ~6
        parts = parts.view(bn,h,w).long()
        return parts

    def __call__(self, vertices, camera):   # mask seems to be n_bch xh x w
        """Wrapper function for rendering process."""
        # Estimate camera parameters given a fixed focal length
        cam_t = torch.stack([camera[:,1], camera[:,2], 2*self.focal_length/(self.render_res * camera[:,0] +1e-9)],dim=-1)   # x, y, scale
        batch_size = vertices.shape[0]
        K = torch.eye(3, device=vertices.device)
        K[0,0] = self.focal_length 
        K[1,1] = self.focal_length 
        K[2,2] = 1
        K[0,2] = self.render_res / 2.
        K[1,2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)
        R = torch.eye(3, device=vertices.device)[None, :, :].expand(batch_size, -1, -1)
        faces = self.faces[None, :, :].expand(batch_size, -1, -1)   # -1 no change
        parts, depth, mask = self.neural_renderer(vertices, faces, textures=self.textures.expand(batch_size, -1, -1, -1, -1, -1), K=K, R=R, t=cam_t.unsqueeze(1))  # rgb depth and alpha
        parts = self.get_parts(parts, mask)
        return mask, parts, depth
