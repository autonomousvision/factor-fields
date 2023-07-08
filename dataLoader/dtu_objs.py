
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

def fps_downsample(points, n_points_to_sample):
    selected_points = np.zeros((n_points_to_sample, 3))
    selected_idxs = []
    dist = np.ones(points.shape[0]) * 100
    for i in range(n_points_to_sample):
        idx = np.argmax(dist)
        selected_points[i] = points[idx]
        selected_idxs.append(idx)
        dist_ = ((points - selected_points[i]) ** 2).sum(-1)
        dist = np.minimum(dist, dist_)

    return selected_idxs

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

        train_scene_idxs = sorted(cfg.train_scene_list)
        test_scene_idxs = cfg.test_scene_list
        if len(train_scene_idxs)==2:
            train_scene_idxs = list(range(train_scene_idxs[0],train_scene_idxs[1]))
        self.scene_idxs = train_scene_idxs if self.split=='train' else test_scene_idxs
        print(self.scene_idxs)
        self.train_views = cfg.train_views
        self.scene_num = len(self.scene_idxs)
        self.test_index = test_scene_idxs
        # if 'test' == self.split:
        #     self.test_index = train_scene_idxs.index(test_scene_idxs[0])

        self.scene_path_list = [os.path.join(self.root_dir, f"scan{i}") for i in self.scene_idxs]
        # self.scene_path_list = sorted(glob(os.path.join(self.root_dir, "scan*")))

        self.read_meta()
        self.white_bg = False

    def read_meta(self):
        self.aabbs = []
        self.all_rgb_files,self.all_pose_files,self.all_intrinsics_files = {},{},{}
        for i, scene_idx in enumerate(self.scene_idxs):

            scene_path = self.scene_path_list[i]
            camera_dict = np.load(os.path.join(scene_path, 'cameras.npz'))

            self.all_rgb_files[scene_idx] = [
                os.path.join(scene_path, "image", f)
                for f in sorted(os.listdir(os.path.join(scene_path, "image")))
            ]

            # world_mat is a projection matrix from world to image
            n_images = len(self.all_rgb_files[scene_idx])
            world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
            scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
            object_scale_mat = camera_dict['scale_mat_0']
            self.aabbs.append(self.get_bbox(scale_mats_np,object_scale_mat))

            # W,H = self.img_wh
            intrinsics_scene, poses_scene = [], []
            for img_idx, (scale_mat, world_mat) in enumerate(zip(scale_mats_np, world_mats_np)):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsic, c2w = load_K_Rt_from_P(None, P)

                c2w = torch.from_numpy(c2w).float()
                intrinsic = torch.from_numpy(intrinsic).float()
                intrinsic[:2] /= self.downsample

                poses_scene.append(c2w)
                intrinsics_scene.append(intrinsic)

            self.all_pose_files[scene_idx] = np.stack(poses_scene)
            self.all_intrinsics_files[scene_idx] = np.stack(intrinsics_scene)

        self.aabbs[0][0].append(0)
        self.aabbs[0][1].append(self.scene_num)
        self.scene_bbox = self.aabbs[0]
        print(self.scene_bbox)
        if self.split=='test' or self.scene_num==1:
            self.load_data(self.scene_idxs[0],range(49))

    def load_data(self, scene_idx, img_idx=None):
        self.all_rays = []

        n_views = len(self.all_pose_files[scene_idx])
        cam_xyzs = self.all_pose_files[scene_idx][:,:3, -1]
        idxs = fps_downsample(cam_xyzs, min(self.train_views, n_views)) if img_idx is None else img_idx
        # if "test" == self.split:
        #     idxs = [item for item in list(range(n_views)) if item not in idxs]
        #     if len(idxs)==0:
        #         idxs = list(range(n_views))

        images_np = np.stack([cv.resize(cv.imread(self.all_rgb_files[scene_idx][idx]), self.img_wh) for idx in idxs]) / 255.0
        self.all_rgbs = torch.from_numpy(images_np.astype(np.float32)[..., [2, 1, 0]])  # [n_images, H, W, 3]

        for c2w,intrinsic in zip(self.all_pose_files[scene_idx][idxs],self.all_intrinsics_files[scene_idx][idxs]):
            rays_o, rays_d = self.gen_rays_at(torch.from_numpy(intrinsic).float(), torch.from_numpy(c2w).float())
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = self.all_rgbs.reshape(-1, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = self.all_rgbs.reshape(-1, *self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

            # self.sampler = SimpleSampler(np.prod(self.all_rgbs.shape[:-1]), self.batch_size)


        # def read_meta(self):
    #
    #     images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
    #     images_np = np.stack([cv.resize(cv.imread(im_name), self.img_wh) for im_name in images_lis]) / 255.0
    #     # masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
    #     # masks_np = np.stack([cv.resize(cv.imread(im_name),self.img_wh) for im_name in masks_lis])>128
    #
    #     self.all_rgbs = torch.from_numpy(images_np.astype(np.float32)[..., [2, 1, 0]])  # [n_images, H, W, 3]
    #     # self.all_masks  = torch.from_numpy(masks_np>0)   # [n_images, H, W, 3]
    #     self.img_wh = [self.all_rgbs.shape[2], self.all_rgbs.shape[1]]
    #
    #     # world_mat is a projection matrix from world to image
    #     n_images = len(images_lis)
    #     world_mats_np = [self.camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    #     self.scale_mats_np = [self.camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    #
    #     # W,H = self.img_wh
    #     self.all_rays = []
    #     self.intrinsics, self.poses = [], []
    #     for img_idx, (scale_mat, world_mat) in enumerate(zip(self.scale_mats_np, world_mats_np)):
    #         P = world_mat @ scale_mat
    #         P = P[:3, :4]
    #         intrinsic, c2w = load_K_Rt_from_P(None, P)
    #
    #         c2w = torch.from_numpy(c2w).float()
    #         intrinsic = torch.from_numpy(intrinsic).float()
    #         intrinsic[:2] /= self.downsample
    #
    #         self.poses.append(c2w)
    #         self.intrinsics.append(intrinsic)
    #
    #         rays_o, rays_d = self.gen_rays_at(intrinsic, c2w)
    #         self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
    #
    #     self.intrinsics, self.poses = torch.stack(self.intrinsics), torch.stack(self.poses)
    #
    #     # self.all_rgbs[~self.all_masks] = 1.0
    #     if not self.is_stack:
    #         self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
    #         self.all_rgbs = self.all_rgbs.reshape(-1, 3)
    #     else:
    #         self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
    #         self.all_rgbs = self.all_rgbs.reshape(-1, *self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
    #
    #     self.sampler = SimpleSampler(np.prod(self.all_rgbs.shape[:-1]), self.batch_size)

    def get_bbox(self, scale_mats_np, object_scale_mat):
        object_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
        object_bbox_max = np.array([ 1.0,  1.0,  1.0, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        # object_scale_mat = np.load(os.path.join(scene_path, 'cameras.npz'))
        object_bbox_min = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        return [object_bbox_min[:3, 0].tolist(),object_bbox_max[:3, 0].tolist()]
        # self.near_far = [2.125, 4.525]

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


    def __len__(self):
        return 1000000 #len(self.all_rays)

    def __getitem__(self, idx):
        idx = torch.randint(self.scene_num,(1,)).item()
        if self.scene_num >= 1:
            scene_name = self.scene_idxs[idx]
            img_idx = np.random.choice(len(self.all_rgb_files[scene_name]), size=6)
            self.load_data(scene_name, img_idx)

        idxs = np.random.choice(self.all_rays.shape[0], size=self.batch_size)

        return {'rays': self.all_rays[idxs], 'rgbs': self.all_rgbs[idxs], 'idx': idx}