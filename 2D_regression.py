import torch,imageio,sys,time,os,cmapy,scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch.nn.functional as F

device = 'cuda'

sys.path.append('..')
from models.sparseCoding import sparseCoding

from dataLoader import dataset_dict
from torch.utils.data import DataLoader


def PSNR(a, b):
    if type(a).__module__ == np.__name__:
        mse = np.mean((a - b) ** 2)
    else:
        mse = torch.mean((a - b) ** 2).item()
    psnr = -10.0 * np.log(mse) / np.log(10.0)
    return psnr


def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


@torch.no_grad()
def eval_img(aabb, reso, shiftment=[0.5, 0.5], chunk=10240):
    y = torch.linspace(0, aabb[0] - 1, reso[0])
    x = torch.linspace(0, aabb[1] - 1, reso[1])
    yy, xx = torch.meshgrid((y, x), indexing='ij')

    idx = 0
    res = torch.empty(reso[0] * reso[1], train_dataset.img.shape[-1])
    coordiantes = torch.stack((xx, yy), dim=-1).reshape(-1, 2) + torch.tensor(
        shiftment)  # /(torch.FloatTensor(reso[::-1])-1)*2-1
    for coordiante in tqdm(torch.split(coordiantes, chunk, dim=0)):
        feats, _ = model.get_coding(coordiante.to(model.device))
        y_recon = model.linear_mat(feats, is_train=False)
        # y_recon = torch.sum(feats,dim=-1,keepdim=True)

        res[idx:idx + y_recon.shape[0]] = y_recon.cpu()
        idx += y_recon.shape[0]
    return res.view(reso[0], reso[1], -1), coordiantes


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


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

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    base_conf = OmegaConf.load('configs/defaults.yaml')
    cli_conf = OmegaConf.from_cli()
    second_conf = OmegaConf.load('configs/image.yaml')
    cfg = OmegaConf.merge(base_conf, second_conf, cli_conf)
    print(cfg)


    folder = cfg.defaults.expname
    save_root = f'/vlg-nfs/anpei/project/NeuBasis/ours/images/'

    dataset = dataset_dict[cfg.dataset.dataset_name]

    delete_region = [[290,350,48,48],[300,380,48,48],[180, 407, 48, 48], [223, 263, 48, 48], [233, 150, 48, 48], [374, 119, 48, 48], [4, 199, 48, 48], [180, 234, 48, 48], [173, 39, 48, 48], [408, 308, 48, 48], [227, 177, 48, 48], [46, 330, 48, 48], [213, 26, 48, 48], [90, 44, 48, 48], [295, 61, 48, 48]]
    continue_sampling = False

    psnrs,ssims = [],[]
    for i in  range(1,257):
        cfg.dataset.datadir = f'/vlg-nfs/anpei/dataset/Images/crop//{i:04d}.png'
        name = os.path.basename(cfg.dataset.datadir).split('.')[0]
        if os.path.exists(f'{save_root}/{folder}/{int(name):04d}.png'):
            continue


        train_dataset = dataset(cfg.dataset, cfg.training.batch_size, split='train',tolinear=True, perscent=1.0,HW=1024)#, continue_sampling=continue_sampling,delete_region=delete_region
        train_loader = DataLoader(train_dataset,
                      num_workers=2,
                      persistent_workers=True,
                      batch_size=None,
                      pin_memory=False)
        # train_dataset.img = train_dataset.img.to(device)

        cfg.model.out_dim = train_dataset.img.shape[-1]
        batch_size = cfg.training.batch_size
        n_iter = cfg.training.n_iters

        H,W = train_dataset.HW
        train_dataset.scene_bbox = [[0., 0.], [W, H]]
        cfg.dataset.aabb = train_dataset.scene_bbox

        model = sparseCoding(cfg, device)
        if 1==i:
            print(model)
            print('total parameters: ',model.n_parameters())

        # tvreg = TVLoss()
        # trainingSampler = SimpleSampler(len(train_dataset), cfg.training.batch_size)

        grad_vars = model.get_optparam_groups(lr_small=cfg.training.lr_small,lr_large=cfg.training.lr_large)
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))#


        loss_scale = 1.0
        lr_factor = 0.1 ** (1 / n_iter)
        # pbar = tqdm(range(10000))
        start = time.time()
        # for iteration in pbar:
        for (iteration, sample) in zip(range(10000),train_loader):
            loss_scale *= lr_factor

            # if iteration==5000:
            #     model.coeffs = torch.nn.Parameter(F.interpolate(model.coeffs.data, size=None, scale_factor=2.0, align_corners=True,mode='bilinear'))
            #     grad_vars = model.get_optparam_groups(lr_small=cfg.training.lr_small,lr_large=cfg.training.lr_large)
            #     optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))#
            #     model.set_optimizable(['mlp','basis'], False)

            coordiantes, pixel_rgb = sample['xy'], sample['rgb']
            feats,coeff = model.get_coding(coordiantes.to(device))
            # tv_loss = model.TV_loss(tvreg)

            y_recon = model.linear_mat(feats,is_train=True)
            # y_recon = torch.sum(feats,dim=-1,keepdim=True)
            loss = torch.mean((y_recon.squeeze()-pixel_rgb.squeeze().to(device))**2)


            psnr = -10.0 * np.log(loss.item()) / np.log(10.0)
            # if iteration%100==0:
            #     pbar.set_description(
            #                 f'Iteration {iteration:05d}:'
            #                 + f' loss_dist = {loss.item():.8f}'
            #                 # + f' tv_loss = {tv_loss.item():.6f}'
            #                 + f' psnr = {psnr:.3f}'
            #             )

            # loss = loss + tv_loss
            # loss = loss + torch.mean(coeff.abs())*1e-2
            loss = loss * loss_scale
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if iteration%100==0:
            #     model.normalize_basis()
        iteration_time = time.time()-start

        H,W = train_dataset.HW
        img,coordinate = eval_img(train_dataset.HW,[1024,1024])
        if continue_sampling:
            import torch.nn.functional as F
            coordinate_tmp = (coordinate.view(1,1,-1,2))/torch.tensor([W,H])*2-1.0
            img_gt = F.grid_sample(train_dataset.img.view(1,H,W,-1).permute(0,3,1,2),coordinate_tmp, mode='bilinear',
                                   align_corners=False, padding_mode='border').reshape(-1,H,W).permute(1,2,0)
        else:
            img_gt = train_dataset.img.view(H,W,-1)
        psnrs.append(PSNR(img.clamp(0,1.),img_gt))
        ssims.append(rgb_ssim(img.clamp(0,1.),img_gt,1.0))
        # print(PSNR(img.clamp(0,1.),img_gt),iteration_time)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(linear_to_srgb(img.clamp(0,1.)))

        print(i, psnrs[-1], ssims[-1])


        os.makedirs(f'{save_root}/{folder}',exist_ok=True)
        write_image_imageio(f'{save_root}/{folder}/{int(name):04d}.png',linear_to_srgb(img.clamp(0,1.)))
        np.savetxt(f'{save_root}/{folder}/{int(name):04d}.txt',[psnrs[-1],ssims[-1],iteration_time,model.n_parameters()])


