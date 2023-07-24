from tqdm.auto import tqdm
from omegaconf import OmegaConf
from models.FactorFields import FactorFields

import json, random,time
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]


@torch.no_grad()
def export_mesh(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['cfg']
    cfg.defaults.device = device
    model = FactorFields(cfg)
    model.load(ckpt)

    alpha, _ = model.getDenseAlpha()
    # scale, offset = 188.5277, np.array([-7.6229210e+00, -5.7847214e-01,  5.9724469e+02])
    # scale, offset = 231.0682, np.array([-28.461128, -13.186272, 641.4886])
    # torch.save(alpha,f'{args.ckpt[:-3]}_occ.th')
    convert_sdf_samples_to_ply(alpha.cpu(), f'{ckpt.defaults.ckpt[:-3]}.ply', bbox=model.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(cfg):
    # init dataset
    dataset = dataset_dict[cfg.dataset.dataset_name]
    test_dataset = dataset(cfg.dataset, split='test')
    white_bg = test_dataset.white_bg
    ndc_ray = cfg.dataset.ndc_ray

    if not os.path.exists(cfg.defaults.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(cfg.defaults.ckpt, map_location=device)
    model = FactorFields( ckpt['cfg'], device)
    model.load(ckpt)


    logfolder = os.path.dirname(cfg.defaults.ckpt)
    if cfg.exportation.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(cfg.dataset.datadir, split='train', is_stack=True)
        PSNRs_test = evaluation(train_dataset, model, render_ray, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {cfg.defaults.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if cfg.exportation.render_test:
        # model.upsample_volume_grid()
        os.makedirs(f'{logfolder}/{cfg.defaults.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, model, render_ray, f'{logfolder}/{cfg.defaults.expname}/imgs_test_all/',
                   N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        n_params = model.n_parameters()
        print(f'======> {cfg.defaults.expname} test all psnr: {np.mean(PSNRs_test)} n_params: {n_params} <========================')


    if cfg.exportation.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{cfg.defaults.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, model, c2ws, render_ray, f'{logfolder}/{cfg.defaults.expname}/imgs_path_all/',
                        N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

    if cfg.exportation.export_mesh:
        alpha, _ = model.getDenseAlpha()
        convert_sdf_samples_to_ply(alpha.cpu(), f'{logfolder}/{cfg.defaults.expname}.ply', bbox=model.aabb.cpu(),
                                   level=0.005)


def reconstruction(cfg):
    # init dataset
    dataset = dataset_dict[cfg.dataset.dataset_name]
    train_dataset = dataset(cfg.dataset, split='train')
    test_dataset = dataset(cfg.dataset, split='test')
    white_bg = train_dataset.white_bg
    ndc_ray = cfg.dataset.ndc_ray

    # init resolution
    upsamp_list = cfg.training.upsamp_list
    update_AlphaMask_list = cfg.training.update_AlphaMask_list

    if cfg.defaults.add_timestamp:
        logfolder = f'{cfg.defaults.logdir}/{cfg.defaults.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{cfg.defaults.logdir}/{cfg.defaults.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    cfg.dataset.aabb = aabb = train_dataset.scene_bbox

    if cfg.defaults.ckpt is not None:
        ckpt = torch.load(cfg.defaults.ckpt, map_location=device)
        cfg = ckpt['cfg']
        model = FactorFields(cfg, device)
        model.load(ckpt)
    else:
        model = FactorFields(cfg, device)
    print(model)

    grad_vars = model.get_optparam_groups(cfg.training.lr_small, cfg.training.lr_large)
    if cfg.training.lr_decay_iters > 0:
        lr_factor = cfg.training.lr_decay_target_ratio ** (1 / cfg.training.lr_decay_iters)
    else:
        cfg.training.lr_decay_iters = cfg.training.n_iters
        lr_factor = cfg.training.lr_decay_target_ratio ** (1 / cfg.training.n_iters)

    print("lr decay", cfg.training.lr_decay_target_ratio, cfg.training.lr_decay_iters)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    volume_resoList = torch.linspace(cfg.training.volume_resoInit, cfg.training.volume_resoFinal,
                                     len(cfg.training.upsamp_list)).ceil().long().tolist()
    reso_cur = N_to_reso(cfg.training.volume_resoInit**model.in_dim, model.aabb)
    nSamples = min(cfg.renderer.max_samples, cal_n_samples(reso_cur, cfg.renderer.step_ratio))

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    trainingSampler = SimpleSampler(allrays.shape[0], cfg.training.batch_size)


    start = time.time()
    pbar = tqdm(range(cfg.training.n_iters), miniters=cfg.defaults.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        rgb_map, depth_map, _ = render_ray(rays_train, model, chunk=cfg.training.batch_size,
                                        N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray, device=device,
                                        is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % cfg.defaults.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        if iteration % cfg.dataset.vis_every == cfg.dataset.vis_every - 1:
            PSNRs_test = evaluation(test_dataset, model, render_ray, f'{logfolder}/imgs_vis/',
                                    N_vis=cfg.dataset.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration in update_AlphaMask_list or iteration in cfg.training.shrinking_list:

            if volume_resoList[0] < 256:  # update volume resolution
                reso_mask = N_to_reso(volume_resoList[0]**model.in_dim, model.aabb)

            new_aabb = model.updateAlphaMask(tuple(reso_mask), is_update_alphaMask=iteration >= 1500)


            if iteration in cfg.training.shrinking_list:
                model.shrink(new_aabb)
                L1_reg_weight = cfg.training.L1_weight_rest

                grad_vars = model.get_optparam_groups(cfg.training.lr_small, cfg.training.lr_large)
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
                print("continuing L1_reg_weight", L1_reg_weight)

            if not cfg.dataset.ndc_ray and iteration == update_AlphaMask_list[0] and not cfg.dataset.is_unbound:
                # filter rays outside the bbox
                allrays, allrgbs = model.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], cfg.training.batch_size)


        if iteration in upsamp_list:
            n_voxels = volume_resoList.pop(0)
            reso_cur = N_to_reso(n_voxels**model.in_dim, model.aabb)
            nSamples = min(cfg.renderer.max_samples, cal_n_samples(reso_cur, cfg.renderer.step_ratio))
            model.upsample_volume_grid(reso_cur)

            grad_vars = model.get_optparam_groups(cfg.training.lr_small, cfg.training.lr_large)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
            torch.cuda.empty_cache()

    time_iter = time.time()-start
    print(f'=======> time takes: {time_iter} <=============')
    os.makedirs(f'{logfolder}/imgs_test_all',exist_ok=True)
    np.savetxt(f'{logfolder}/imgs_test_all/time.txt',[time_iter])
    model.save(f'{logfolder}/{cfg.defaults.expname}.th')

    if cfg.exportation.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, model, render_ray, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        n_params = model.n_parameters()
        print(f'======> {cfg.defaults.expname} test all psnr: {np.mean(PSNRs_test)} n_params: {n_params} <========================')


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    base_conf = OmegaConf.load('configs/defaults.yaml')
    path_config = sys.argv[1]
    cli_conf = OmegaConf.from_cli()
    second_conf = OmegaConf.load(path_config)
    cfg = OmegaConf.merge(base_conf, second_conf, cli_conf)

    if cfg.exportation.render_only and (cfg.exportation.render_test or cfg.exportation.render_path):
        render_test(cfg)
    elif cfg.exportation.export_mesh_only:
        export_mesh(cfg)
    else:
        reconstruction(cfg)