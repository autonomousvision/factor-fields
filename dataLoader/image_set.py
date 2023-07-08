import torch,cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

def srgb_to_linear(img):
	limit = 0.04045
	return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

def load(path, HW=512):
    suffix = path.split('.')[-1]

    if 'npy' == suffix:
        img = np.load(path)
        # img = 0.3*img[...,:1] + 0.59*img[...,1:2] + 0.11*img[...,2:]

    if img.shape[-2]!=HW:
        for i in range(img.shape[0]):
            img[i] = cv2.resize(img[i],[HW,HW])
    
    return img


class ImageSetDataset(Dataset):
    def __init__(self, cfg, batchsize, split='train', continue_sampling=False, HW=512, N=10, tolinear=True):

        datadir = cfg.datadir
        self.batchsize = batchsize
        self.continue_sampling = continue_sampling
        imgs = load(datadir,HW=HW)[:N]
        
            
        self.imgs = torch.from_numpy(imgs).float()/255
        if tolinear:
            self.imgs = srgb_to_linear(self.imgs)

        D,H,W = self.imgs.shape[:3]
        
        y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        self.coordinate = torch.stack((x,y),-1).float()+0.5

        self.imgs, self.coordinate = self.imgs.reshape(D,H*W,-1), self.coordinate.reshape(H*W,2)
        self.DHW = [D,H,W]

        self.scene_bbox = [[0., 0., 0.], [W, H, D]]
        cfg.aabb = self.scene_bbox
        # self.down_scale = 512.0/H


    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        D,H,W = self.DHW
        pix_idx = torch.randint(0,H*W,(self.batchsize,))
        img_idx = torch.randint(0,D,(self.batchsize,))
        

        if self.continue_sampling:
            coordinate = self.coordinate[pix_idx] +  torch.rand((self.batchsize,2)) - 0.5
            coordinate = torch.cat((coordinate,img_idx.unsqueeze(-1)+0.5),dim=-1)
            coordinate_tmp = (coordinate.view(1,1,1,self.batchsize,3))/torch.tensor([W,H,D])*2-1.0
            rgb = F.grid_sample(self.imgs.view(1,D,H,W,-1).permute(0,4,1,2,3),coordinate_tmp, mode='bilinear',
                                align_corners=False, padding_mode='border').reshape(self.imgs.shape[-1],-1).t()
            # coordinate[:,:2] *= self.down_scale
            sample = {'rgb': rgb,
                      'xy': coordinate}
        else:
            sample = {'rgb': self.imgs[img_idx,pix_idx],
                      'xy': torch.cat((self.coordinate[pix_idx],img_idx.unsqueeze(-1)+0.5),dim=-1)}
                      # 'xy': torch.cat((self.coordiante[pix_idx],img_idx.expand_as(pix_idx).unsqueeze(-1)),dim=-1)}



        return sample