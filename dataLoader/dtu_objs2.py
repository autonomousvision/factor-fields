
import torch
import cv2 as cv
import numpy as np
import os
from glob import glob
from .ray_utils import *
from torch.utils.data import Dataset


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class DTUDataset(Dataset):
    def __init__(self, cfg, split='train', batch_size=4096, is_stack=None):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        # self.N_vis = N_vis
        self.split = split
        self.batch_size = batch_size
        self.root_dir = cfg.datadir
        self.is_stack = is_stack if is_stack is not None else 'train'!=split
        self.downsample = cfg.get(f'downsample_{self.split}')
        self.img_wh = (int(400 / self.downsample), int(300 / self.downsample))

        self.white_bg = False
        self.camera_dict = np.load(os.path.join(self.root_dir, 'cameras.npz'))

        self.read_meta()
        self.get_bbox()

    # def define_transforms(self):
    #     self.transform = T.ToTensor()

    def get_bbox(self):
        object_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
        object_bbox_max = np.array([ 1.0,  1.0,  1.0, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.root_dir, 'cameras.npz'))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.scene_bbox = [object_bbox_min[:3, 0].tolist(),object_bbox_max[:3, 0].tolist()]
        self.scene_bbox[0].append(0)
        self.scene_bbox[1].append(1)

    def gen_rays_at(self, intrinsic, c2w, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        W,H = self.img_wh
        tx = torch.linspace(0, W - 1, W // l)+0.5
        ty = torch.linspace(0, H - 1, H // l)+0.5
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        intrinsic_inv = torch.inverse(intrinsic)
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(c2w[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = c2w[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1).reshape(-1,3), rays_v.transpose(0, 1).reshape(-1,3)

    def read_meta(self):

        images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
        images_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in images_lis]) / 255.0
        # masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
        # masks_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in masks_lis])>128

        self.all_rgbs = torch.from_numpy(images_np.astype(np.float32)[...,[2,1,0]])  # [n_images, H, W, 3]
        # self.all_masks  = torch.from_numpy(masks_np>0)   # [n_images, H, W, 3]
        self.img_wh = [self.all_rgbs.shape[2],self.all_rgbs.shape[1]]

        # world_mat is a projection matrix from world to image
        n_images = len(images_lis)
        world_mats_np = [self.camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        self.scale_mats_np = [self.camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

        # W,H = self.img_wh
        self.all_rays = []
        self.intrinsics, self.poses = [],[]
        for img_idx, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, world_mats_np)):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic, c2w = load_K_Rt_from_P(None, P)

            c2w = torch.from_numpy(c2w).float()
            intrinsic = torch.from_numpy(intrinsic).float()
            intrinsic[:2] /= self.downsample

            self.poses.append(c2w)
            self.intrinsics.append(intrinsic)

            rays_o, rays_d = self.gen_rays_at(intrinsic,c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.intrinsics, self.poses = torch.stack(self.intrinsics), torch.stack(self.poses)

        # self.all_rgbs[~self.all_masks] = 1.0
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = self.all_rgbs.reshape(-1,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = self.all_rgbs.reshape(-1, *self.img_wh[::-1],3)  # (len(self.meta['frames]),h,w,3)

        self.sampler = SimpleSampler(np.prod(self.all_rgbs.shape[:-1]), self.batch_size)

    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        idx_rand = self.sampler.nextids() #torch.randint(0,len(self.all_rays),(self.batch_size,))
        sample = {'rays': self.all_rays[idx_rand], 'rgbs': self.all_rgbs[idx_rand]}
        return sample