import torch
import numpy as np
from torch.utils.data import Dataset

def N_to_reso(avg_reso, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    n_voxels = avg_reso**dim
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return torch.ceil((xyz_max - xyz_min) / voxel_size).long().tolist()

def load(path, split, dtype='points'):

    if 'grid' == dtype:
        sdf = torch.from_numpy(np.load(path).astype(np.float32))
        D, H, W = sdf.shape
        z, y, x = torch.meshgrid(torch.arange(0, D), torch.arange(0, H), torch.arange(0, W), indexing='ij')
        coordiante = torch.stack((x,y,z),-1).reshape(D*H*W,3)#*2-1 # normalize to [-1,1]
        sdf = sdf.reshape(D*H*W,-1)
        DHW = [D,H,W]
    elif 'points' == dtype:
        DHW = [640] * 3
        sdf_dict = np.load(path, allow_pickle=True).item()
        sdf = torch.from_numpy(sdf_dict[f'sdfs_{split}'].astype(np.float32)).reshape(-1,1)
        coordiante = torch.from_numpy(sdf_dict[f'points_{split}'].astype(np.float32))
        aabb = [[-1,-1,-1],[1,1,1]]
        coordiante = ((coordiante + 1) / 2 * (torch.tensor(DHW[::-1]))).reshape(-1,3)
        DHW = DHW[::-1]
    return coordiante, sdf, DHW

class SDFDataset(Dataset):
    def __init__(self, cfg, split='train'):

        datadir = cfg.datadir
        self.coordiante, self.sdf, self.DHW = load(datadir, split)

        [D, H, W] = self.DHW

        self.scene_bbox = [[0., 0., 0.], [W, H, D]]
        cfg.aabb = self.scene_bbox

    def __len__(self):
        return len(self.sdf)

    def __getitem__(self, idx):
        sample = {'rgb': self.sdf[idx],
                  'xy': self.coordiante[idx]}

        return sample