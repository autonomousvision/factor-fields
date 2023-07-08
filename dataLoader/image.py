import torch,imageio,cv2
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

_img_suffix = ['png','jpg','jpeg','bmp','tif']

def load(path):
    suffix = path.split('.')[-1]
    if suffix in _img_suffix:
        img =  np.array(Image.open(path))#.convert('L')
        scale = 256.**(1+np.log2(np.max(img))//8)-1
        return img/scale
    elif 'exr' == suffix:
        return imageio.imread(path)
    elif 'npy' == suffix:
        return np.load(path)

    
def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

class ImageDataset(Dataset):
    def __init__(self, cfg, batchsize, split='train', continue_sampling=False, tolinear=False, HW=-1, perscent=1.0, delete_region=None,mask=None):

        datadir = cfg.datadir
        self.batchsize = batchsize
        self.continue_sampling = continue_sampling
        img = load(datadir).astype(np.float32)
        if HW>0:
            img = cv2.resize(img,[HW,HW])
            
        if tolinear:
            img = srgb_to_linear(img)
        self.img = torch.from_numpy(img)

        H,W = self.img.shape[:2]

        y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        self.coordiante = torch.stack((x,y),-1).float()+0.5

        n_channel = self.img.shape[-1]
        self.image =  self.img
        self.img, self.coordiante = self.img.reshape(H*W,-1), self.coordiante.reshape(H*W,2)
      
        # if continue_sampling:
        #     coordiante_tmp = self.coordiante.view(1,1,-1,2)/torch.tensor([W,H])*2-1.0
        #     self.img = F.grid_sample(self.img.view(1,H,W,-1).permute(0,3,1,2),coordiante_tmp, mode='bilinear', align_corners=True).reshape(self.img.shape[-1],-1).t()
            
            
        if 'train'==split:
            self.mask = torch.ones_like(y)>0
            if mask is not None:
                self.mask = mask>0
                print(torch.sum(mask)/1.0/HW/HW)
            elif delete_region is not None:
                
                if isinstance(delete_region[0], list):
                    for item in delete_region:
                        t_l_x,t_l_y,width,height = item
                        self.mask[t_l_y:t_l_y+height,t_l_x:t_l_x+width] = False
                else:
                    t_l_x,t_l_y,width,height = delete_region
                    self.mask[t_l_y:t_l_y+height,t_l_x:t_l_x+width] = False
            else:
                index = torch.randperm(len(self.img))[:int(len(self.img)*perscent)] 
                self.mask[:] = False
                self.mask.view(-1)[index] = True
            self.mask = self.mask.view(-1)
            self.image, self.coordiante = self.img[self.mask], self.coordiante[self.mask]
        else:
            self.image = self.img
            

        self.HW = [H,W]

        self.scene_bbox = [[0., 0.], [W, H]]
        cfg.aabb = self.scene_bbox
        #

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        H,W = self.HW 
        device = self.image.device
        idx = torch.randint(0,len(self.image),(self.batchsize,), device=device)
        
        if self.continue_sampling:
            coordinate = self.coordiante[idx] +  torch.rand((self.batchsize,2))-0.5
            coordinate_tmp = (coordinate.view(1,1,self.batchsize,2))/torch.tensor([W,H],device=device)*2-1.0
            rgb = F.grid_sample(self.img.view(1,H,W,-1).permute(0,3,1,2),coordinate_tmp, mode='bilinear', 
                                align_corners=False, padding_mode='border').reshape(self.img.shape[-1],-1).t()
            sample = {'rgb': rgb,
                      'xy': coordinate}
        else:
            sample = {'rgb': self.image[idx],
                      'xy': self.coordiante[idx]}

        return sample