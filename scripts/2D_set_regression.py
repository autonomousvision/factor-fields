import torch,imageio,sys,cmapy,time,os
import numpy as np
from tqdm import tqdm
# from .autonotebook import tqdm as tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch.nn.functional as F

sys.path.append('..')
from models.FactorFields import FactorFields 

from utils import SimpleSampler,TVLoss
from dataLoader import dataset_dict
from torch.utils.data import DataLoader

device = 'cuda'


def PSNR(a, b):
    if type(a).__module__ == np.__name__:
        mse = np.mean((a - b) ** 2)
    else:
        mse = torch.mean((a - b) ** 2).item()
    psnr = -10.0 * np.log(mse) / np.log(10.0)
    return psnr


@torch.no_grad()
def eval_img(aabb, reso, idx, shiftment=[0.5, 0.5, 0.5], chunk=10240):
    y = torch.linspace(0, aabb[0] - 1, reso[0])
    x = torch.linspace(0, aabb[1] - 1, reso[1])
    yy, xx = torch.meshgrid((y, x), indexing='ij')
    zz = torch.ones_like(xx) * idx

    idx = 0
    res = torch.empty(reso[0] * reso[1], train_dataset.imgs.shape[-1])
    coordiantes = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3) + torch.tensor(
        shiftment)  # /(torch.FloatTensor(reso[::-1])-1)*2-1
    for coordiante in tqdm(torch.split(coordiantes, chunk, dim=0)):
        feats, _ = model.get_coding_imgage_set(coordiante.to(model.device))
        y_recon = model.linear_mat(feats, is_train=False)

        res[idx:idx + y_recon.shape[0]] = y_recon.cpu()
        idx += y_recon.shape[0]
    return res.view(reso[0], reso[1], -1), coordiantes


@torch.no_grad()
def eval_img_single(aabb, reso, chunk=10240):
    y = torch.linspace(0, aabb[0] - 1, reso[0])
    x = torch.linspace(0, aabb[1] - 1, reso[1])
    yy, xx = torch.meshgrid((y, x), indexing='ij')

    idx = 0
    res = torch.empty(reso[0] * reso[1], train_dataset.img.shape[-1])
    coordiantes = torch.stack((xx, yy), dim=-1).reshape(-1, 2) + 0.5

    for coordiante in tqdm(torch.split(coordiantes, chunk, dim=0)):
        feats, _ = model.get_coding(coordiante.to(model.device))
        y_recon = model.linear_mat(feats, is_train=False)

        res[idx:idx + y_recon.shape[0]] = y_recon.cpu()
        idx += y_recon.shape[0]
    return res.view(reso[0], reso[1], -1), coordiantes


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def srgb_to_linear(img):
    limit = 0.04045
    return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)


def write_image_imageio(img_file, img, colormap=None, quality=100):
    if colormap == 'turbo':
        shape = img.shape
        img = interpolate(turbo_colormap_data, img.reshape(-1)).reshape(*shape, -1)
    elif colormap is not None:
        img = cmapy.colorize((img * 255).astype('uint8'), colormap)

    if img.dtype != 'uint8':
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255.0).astype(np.uint8)

    kwargs = {}
    if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
        if img.ndim >= 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        kwargs["quality"] = quality
        kwargs["subsampling"] = 0
    imageio.imwrite(img_file, img, **kwargs)


def interpolate(colormap, x):
    a = (x * 255.0).astype('uint8')
    b = np.clip(a + 1, 0, 255)
    f = x * 255.0 - a

    return np.stack([colormap[a][..., 0] + (colormap[b][..., 0] - colormap[a][..., 0]) * f,
                     colormap[a][..., 1] + (colormap[b][..., 1] - colormap[a][..., 1]) * f,
                     colormap[a][..., 2] + (colormap[b][..., 2] - colormap[a][..., 2]) * f], axis=-1)

base_conf = OmegaConf.load('../configs/defaults.yaml')
second_conf = OmegaConf.load('../configs/image_set.yaml')
cfg = OmegaConf.merge(base_conf, second_conf)

dataset = dataset_dict[cfg.dataset.dataset_name]
train_dataset = dataset(cfg.dataset,cfg.training.batch_size, split='train',N=600,tolinear=True,HW=512, continue_sampling=True)
train_loader = DataLoader(train_dataset,
              num_workers=8,
              persistent_workers=True,
              batch_size=None,
              pin_memory=True)


batch_size = cfg.training.batch_size
n_iter = cfg.training.n_iters

model = FactorFields(cfg, device)
tvreg = TVLoss()
print(model)
print('total parameters: ', model.n_parameters())

grad_vars = model.get_optparam_groups(lr_small=cfg.training.lr_small, lr_large=cfg.training.lr_large)
optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))  #

tv_loss = 0
loss_scale = 1.0
lr_factor = 0.1 ** (1 / n_iter)
pbar = tqdm(range(n_iter))
for (iteration, sample) in zip(pbar, train_loader):
    loss_scale *= lr_factor

    coordiantes, pixel_rgb = sample['xy'], sample['rgb']

    basis, coeff = model.get_coding_imgage_set(coordiantes.to(device))

    y_recon = model.linear_mat(basis, is_train=True)
    # y_recon = torch.sum(basis,dim=-1,keepdim=True)
    l2_loss = torch.mean((y_recon.squeeze() - pixel_rgb.squeeze().to(device)) ** 2)  # + 4e-3*coeff.abs().mean()

    # tv_loss = model.TV_loss(tvreg)
    loss = l2_loss  # + tv_loss*10

    # loss = loss * loss_scale
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if iteration%100==0:
    #     model.normalize_basis()

    psnr = -10.0 * np.log(l2_loss.item()) / np.log(10.0)
    if iteration % 100 == 0:
        pbar.set_description(
            f'Iteration {iteration:05d}:'
            + f' loss_dist = {l2_loss.item():.8f}'
            + f' tv_loss = {tv_loss:.6f}'
            + f' psnr = {psnr:.3f}'
        )

save_root = '../log/imageSet/ffhq_mlp_coeff_16_64_8_pe_linear_64_node_800/'
os.makedirs(save_root, exist_ok=True)
model.save(f'{save_root}/ckpt.th')