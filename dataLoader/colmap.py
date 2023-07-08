import torch, cv2
from torch.utils.data import Dataset

from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


class ColmapDataset(Dataset):
    def __init__(self, cfg, split='train'):

        self.cfg = cfg
        self.root_dir = cfg.datadir
        self.split = split
        self.is_stack = False if 'train'==split else True
        self.downsample = cfg.get(f'downsample_{self.split}')
        self.define_transforms()
        self.img_eval_interval = 8

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])#np.eye(4)#
        self.read_meta()

        self.white_bg = cfg.get('white_bg')
        
        # self.near_far = [0.1,2.0]


    def read_meta(self):

        if os.path.exists(f'{self.root_dir}/transforms.json'):
            self.meta = load_json(f'{self.root_dir}/transforms.json')
            i_test = np.arange(0, len(self.meta['frames']), self.img_eval_interval)  # [np.argmin(dists)]
            idxs = i_test if self.split != 'train' else list(set(np.arange(len(self.meta['frames']))) - set(i_test))
        else:
            self.meta = load_json(f'{self.root_dir}/transforms_{self.split}.json')
            idxs = np.arange(0, len(self.meta['frames']))  # [np.argmin(dists)]
            inv_split = 'train' if self.split!='train' else 'test'
            self.meta['frames'] += load_json(f'{self.root_dir}/transforms_{inv_split}.json')['frames']
            print(len(self.meta['frames']),len(idxs))


        self.scale = self.meta.get('scale', 0.5)
        self.offset = torch.FloatTensor(self.meta.get('offset', [0.0,0.0,0.0]))
        # self.scene_bbox = (torch.tensor([[-6.,-7.,-10.0],[6.,7.,10.]])/5).tolist()
        # self.scene_bbox = [[-1., -1., -1.0], [1., 1., 1.]]

        # center, radius = torch.tensor([-0.082157, 2.415426,-3.703080]), torch.tensor([7.36916, 11.34958, 20.1616])/2
        # self.scene_bbox = torch.stack([center-radius, center+radius]).tolist()

        h, w = int(self.meta.get('h')), int(self.meta.get('w'))
        cx, cy = self.meta.get('cx'), self.meta.get('cy')
        self.focal = [self.meta.get('fl_x'), self.meta.get('fl_y')]

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.focal, center=[cx, cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        # self.intrinsics = torch.FloatTensor([[self.focal[0], 0, cx], [0, self.focal[1], cy], [0, 0, 1]])
        # self.intrinsics[:2] /= self.downsample

        poses = pose_from_json(self.meta, self.blender2opencv)
        poses, self.scene_bbox = orientation(poses, f'{self.root_dir}/colmap_text/points3D.txt')

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []

        self.img_wh = [w,h]

        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#

            frame = self.meta['frames'][i]
            c2w = torch.FloatTensor(poses[i])
            # c2w[:3,3] = (c2w[:3,3]*self.scale + self.offset)*2-1
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, frame['file_path'])
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)

            img = self.transform(img)
            if img.shape[0]==4:
                img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
            img = img.view(3, -1).permute(1, 0)
            self.all_rgbs += [img]


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1],3)  # (len(self.meta['frames]),h,w,3)

    def define_transforms(self):
        self.transform = T.ToTensor()


    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]

            sample = {'rays': rays,
                      'rgbs': img}
        return sample