import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


class BlenderDatasetSet(Dataset):
    def __init__(self, cfg, split='train'):

        # self.N_vis = N_vis
        self.root_dir = cfg.datadir
        self.split = split
        self.is_stack = False if 'train'==split else True
        self.downsample = cfg.get(f'downsample_{self.split}')
        self.img_wh = (int(800 / self.downsample), int(800 / self.downsample))
        self.define_transforms()

        self.rot = torch.tensor([[0.65561799, -0.65561799, 0.37460659],
                                 [0.73729737, 0.44876192, -0.50498052],
                                 [0.16296514, 0.60727077, 0.77760181]])

        self.scene_bbox = (np.array([[-1.0, -1.0, -1.0, 0], [1.0, 1.0, 1.0, 2]])).tolist()
        # self.scene_bbox = [[-0.8,-0.8,-0.22],[0.8,0.8,0.2]]
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0, 6.0]

        # self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        # self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0

        img_eval_interval = 1 #if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            c2w[:3,-1] /= 1.5
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            # rays_o, rays_d = rays_o@self.rot, rays_d@self.rot

            scene_id = torch.ones_like(rays_o[...,:1])*0
            self.all_rays += [torch.cat([rays_o, rays_d, scene_id], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

        #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1],3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    # def world2ndc(self,points,lindisp=None):
    #     device = points.device
    #     return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        rays = torch.cat((self.all_rays[idx],torch.tensor([0+0.5])),dim=-1)
        sample = {'rays': rays, 'rgbs': self.all_rgbs[idx]}
        return sample