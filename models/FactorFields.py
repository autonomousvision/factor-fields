import torch, math
import torch.nn
import torch.nn.functional as F
import numpy as np
import time, skimage
from utils import N_to_reso, N_to_vm_reso


# import BasisCoding

def grid_mapping(positions, freq_bands, aabb, basis_mapping='sawtooth'):
    aabbSize = max(aabb[1] - aabb[0])
    scale = aabbSize[..., None] / freq_bands
    if basis_mapping == 'triangle':
        pts_local = (positions - aabb[0]).unsqueeze(-1) % scale
        pts_local_int = ((positions - aabb[0]).unsqueeze(-1) // scale) % 2
        pts_local = pts_local / (scale / 2) - 1
        pts_local = torch.where(pts_local_int == 1, -pts_local, pts_local)
    elif basis_mapping == 'sawtooth':
        pts_local = (positions - aabb[0])[..., None] % scale
        pts_local = pts_local / (scale / 2) - 1
        pts_local = pts_local.clamp(-1., 1.)
    elif basis_mapping == 'sinc':
        pts_local = torch.sin((positions - aabb[0])[..., None] / (scale / np.pi) - np.pi / 2)
    elif basis_mapping == 'trigonometric':
        pts_local = (positions - aabb[0])[..., None] / scale * 2 * np.pi
        pts_local = torch.cat((torch.sin(pts_local), torch.cos(pts_local)), dim=-1)
    elif basis_mapping == 'x':
        pts_local = (positions - aabb[0]).unsqueeze(-1) / scale
    # elif basis_mapping=='hash':
    #     pts_local = (positions - aabb[0])/max(aabbSize)

    return pts_local


def dct_dict(n_atoms_fre, size, n_selete, dim=2):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis. If n_atoms is
    not a perfect square, the returned dictionary will have ceil(sqrt(n_atoms))**2 atoms
    :param n_atoms:
        Number of atoms in dict
    :param size:
        Size of first patch dim
    :return:
        DCT dictionary, shape (size*size, ceil(sqrt(n_atoms))**2)
    """
    # todo flip arguments to match random_dictionary
    p = n_atoms_fre  # int(math.ceil(math.sqrt(n_atoms)))
    dct = np.zeros((p, size))

    for k in range(p):
        basis = np.cos(np.arange(size) * k * math.pi / p)
        if k > 0:
            basis = basis - np.mean(basis)

        dct[k] = basis

    kron = np.kron(dct, dct)
    if 3 == dim:
        kron = np.kron(kron, dct)

    if n_selete < kron.shape[0]:
        idx = [x[0] for x in np.array_split(np.arange(kron.shape[0]), n_selete)]
        kron = kron[idx]

    for col in range(kron.shape[0]):
        norm = np.linalg.norm(kron[col]) or 1
        kron[col] /= norm

    kron = torch.FloatTensor(kron)
    return kron


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[..., :-1]  # [N_rays, N_samples]
    return alpha, weights, T[..., -1:]


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPMixer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim=16,
                 num_layers=2,
                 hidden_dim=64, pe=0, with_dropout=False):
        super().__init__()

        self.with_dropout = with_dropout
        self.in_dim = in_dim + 2 * in_dim * pe
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pe = pe

        backbone = []
        for l in range(num_layers):
            if l == 0:
                layer_in_dim = self.in_dim
            else:
                layer_in_dim = self.hidden_dim

            if l == num_layers - 1:
                layer_out_dim, bias = out_dim, False
            else:
                layer_out_dim, bias = self.hidden_dim, True

            backbone.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=bias))

        self.backbone = torch.nn.ModuleList(backbone)
        # torch.nn.init.constant_(backbone[0].weight.data, 1.0/self.in_dim)

    def forward(self, x, is_train=False):
        # x: [B, 3]
        h = x
        if self.pe > 0:
            h = torch.cat([h, positional_encoding(h, self.pe)], dim=-1)

        if self.with_dropout and is_train:
            h = F.dropout(h, p=0.1)

        for l in range(self.num_layers):
            h = self.backbone[l](h)
            if l != self.num_layers - 1:  # l!=0 and
                h = F.relu(h, inplace=True)
                # h = torch.sin(h)
        # sigma, feat = h[...,0], h[...,1:]
        return h


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, num_layers=3, hidden_dim=64, viewpe=6, feape=2):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 3 + inChanel + 2 * viewpe * 3 + 2 * feape * inChanel
        self.num_layers = num_layers
        self.viewpe = viewpe
        self.feape = feape

        mlp = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_mlpC
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim, bias = 3, False  # 3 rgb
            else:
                out_dim, bias = hidden_dim, True

            mlp.append(torch.nn.Linear(in_dim, out_dim, bias=bias))

        self.mlp = torch.nn.ModuleList(mlp)
        # torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features):

        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]

        h = torch.cat(indata, dim=-1)
        for l in range(self.num_layers):
            h = self.mlp[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        rgb = torch.sigmoid(h)
        return rgb


class FactorFields(torch.nn.Module):
    def __init__(self, cfg, device):
        super(FactorFields, self).__init__()

        self.cfg = cfg
        self.device = device

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.n_scene, self.scene_idx = 1, 0

        self.alphaMask = None
        self.coeff_type, self.basis_type = cfg.model.coeff_type, cfg.model.basis_type

        self.setup_params(self.cfg.dataset.aabb)
        if self.cfg.model.coeff_type != 'none':
            self.coeffs = self.init_coef()

        if self.cfg.model.basis_type != 'none':
            self.basises = self.init_basis()

        out_dim = cfg.model.out_dim
        if 'vm' in self.coeff_type:
            in_dim = sum(cfg.model.basis_dims) * 3
        elif 'x' in self.cfg.model.basis_type:
            in_dim = len(
                cfg.model.basis_dims) * 2 * self.in_dim if self.cfg.model.basis_mapping == 'trigonometric' else len(
                cfg.model.basis_dims) * self.in_dim
        else:
            in_dim = sum(cfg.model.basis_dims)
        self.linear_mat = MLPMixer(in_dim, out_dim, num_layers=cfg.model.num_layers, hidden_dim=cfg.model.hidden_dim,
                                   with_dropout=cfg.model.with_dropout).to(device)

        if 'reconstruction' in cfg.defaults.mode:
            # self.cur_volumeSize = N_to_reso(cfg.training.volume_resoInit, self.aabb)
            # self.update_renderParams(self.cur_volumeSize)

            view_pe, fea_pe = cfg.renderer.view_pe, cfg.renderer.fea_pe
            num_layers, hidden_dim = cfg.renderer.num_layers, cfg.renderer.hidden_dim
            self.renderModule = MLPRender_Fea(inChanel=out_dim - 1, num_layers=num_layers, hidden_dim=hidden_dim,
                                              viewpe=view_pe, feape=fea_pe).to(device)

            self.is_unbound = self.cfg.dataset.is_unbound
            if self.is_unbound:
                self.bg_len = 0.2
                self.inward_aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).to(device)
                self.aabb = self.inward_aabb * (1 + self.bg_len)
            else:
                self.inward_aabb = self.aabb

            # self.freq_bands = torch.FloatTensor(cfg.model.freq_bands).to(device)
            self.cur_volumeSize = N_to_reso(cfg.training.volume_resoInit ** self.in_dim, self.aabb)
            self.update_renderParams(self.cur_volumeSize)

        print('=====> total parameters: ', self.n_parameters())

    def setup_params(self, aabb):

        self.in_dim = len(aabb[0]) - 1 if (
                    'images' == self.cfg.defaults.mode or 'reconstructions' == self.cfg.defaults.mode) else len(aabb[0])
        self.aabb = torch.FloatTensor(aabb)[:, :self.in_dim].to(self.device)

        self.basis_dims = self.cfg.model.basis_dims
        if 'reconstruction' not in self.cfg.defaults.mode:
            self.basis_reso = self.cfg.model.basis_resos if 'image' in self.cfg.defaults.mode else np.round(
                np.array(self.cfg.model.basis_resos) * (min(aabb[1][:self.in_dim]) + 1) / 1024.0).astype('int').tolist()
            self.T_basis = self.cfg.model.T_basis if self.cfg.model.T_basis>0 else sum(np.power(np.array(self.basis_reso), self.in_dim) * np.array(self.cfg.model.basis_dims))
            self.T_coeff = self.cfg.model.T_coeff if self.cfg.model.T_coeff>0 else self.cfg.model.total_params - self.T_basis
            self.T_coeff = self.T_coeff if self.T_coeff > 0 else 8 ** self.in_dim * sum(self.basis_dims)
            print(self.T_basis,self.T_coeff,self.T_coeff )

            if 'image' == self.cfg.defaults.mode:
                self.freq_bands = max(aabb[1][:self.in_dim]) / torch.FloatTensor(self.basis_reso).to(self.device)
            else:
                self.freq_bands = torch.FloatTensor(self.cfg.model.freq_bands).to(self.device)

            self.coeff_reso = N_to_reso(self.T_coeff // sum(self.basis_dims), self.aabb[:, :self.in_dim])[::-1]  # DHW
            self.n_scene = 1  # int(aabb[1][-1]) if 'images' == self.cfg.defaults.mode else 1

            if 'sdf' == self.cfg.defaults.mode:
                self.freq_bands *= 0.5
            elif 'images' == self.cfg.defaults.mode:
                self.coeff_reso = [aabb[1][-1]] + self.coeff_reso
                self.aabb = torch.FloatTensor(aabb).to(self.device)
            if 'vec' in self.coeff_type or 'cp' in self.coeff_type or 'vm' in self.coeff_type:
                self.coeff_reso = aabb[1]
        else:
            self.coeff_reso = N_to_reso(self.cfg.model.coeff_reso ** self.in_dim, self.aabb[:, :self.in_dim])[::-1]
            self.T_coeff = sum(self.cfg.model.basis_dims) * np.prod(self.coeff_reso)
            self.T_basis = self.cfg.model.total_params - self.T_coeff
            scale = self.T_basis / sum(
                np.power(np.array(self.cfg.model.basis_resos), self.in_dim) * np.array(self.cfg.model.basis_dims))
            scale = np.power(scale, 1.0 / self.in_dim)
            self.basis_reso = self.cfg.model.basis_resos if (
                        'vec' in self.basis_type or 'cp' in self.basis_type) else np.round(
                np.array(self.cfg.model.basis_resos) * scale).astype('int').tolist()
            # self.freq_bands = self.cfg.dataset.scene_reso / torch.FloatTensor(self.basis_resos).to(self.device)
            self.freq_bands = torch.FloatTensor(self.cfg.model.freq_bands).to(self.device) \
                if (
                        'reconstructions' == self.cfg.defaults.mode or 'x' in self.basis_type or 'vec' in self.basis_type or 'cp' in self.basis_type) else torch.FloatTensor(
                self.cfg.model.freq_bands).to(self.device) * (self.cfg.dataset.scene_reso / float(
                max(self.basis_reso)) / max(self.cfg.model.freq_bands))
            self.n_scene = int(aabb[1][-1]) if 'reconstructions' == self.cfg.defaults.mode else 1

        # print(self.coeff_reso,self.basis_reso,self.freq_bands)

    def init_coef(self):
        n_scene = self.n_scene if 'reconstructions' == self.cfg.defaults.mode or 'images' == self.cfg.defaults.mode else 1
        if 'hash' in self.coeff_type or 'grid' in self.coeff_type:
            coeffs = [
                self.cfg.model.coef_init * torch.ones((1, sum(self.basis_dims), *self.coeff_reso), device=self.device)
                for _ in range(n_scene)]
            coeffs = torch.nn.ParameterList(coeffs)
        elif 'cp' in self.coeff_type or 'vm' in self.coeff_type:
            coeffs = []
            for i in range(len(self.coeff_reso)):
                coeffs.append(self.cfg.model.coef_init * torch.ones(
                    (1, sum(self.basis_dims), max(256, self.coeff_reso[i]), n_scene), device=self.device))
            coeffs = torch.nn.ParameterList(coeffs)
        elif 'vec' in self.coeff_type:
            coeffs = self.cfg.model.coef_init * torch.ones(
                (1, sum(self.basis_dims), max(256, max(self.coeff_reso)), n_scene), device=self.device)
            coeffs = torch.nn.ParameterList([coeffs])
        elif 'mlp' in self.coeff_type:
            coeffs = torch.nn.ParameterList(
                [MLPMixer(self.in_dim, sum(self.basis_dims), num_layers=2, hidden_dim=64, pe=4).to(self.device) for _ in
                 range(n_scene)])
        return coeffs

    def init_basis(self):

        if 'hash' in self.basis_type:
            import tinycudann as tcnn
            n_levels = len(self.basis_reso)
            if 'reconstruction' not in self.cfg.defaults.mode:
                base_resolution_low, base_resolution_high = 32, int(max(self.aabb[1]).item()) // 2
                per_level_scale = torch.pow(self.freq_bands[-1] / self.freq_bands[0], 1.0 / n_levels).item()
            else:
                base_resolution_low = torch.round(self.basis_reso[0] * self.freq_bands[0]).long().item()
                per_level_scale = torch.pow(self.freq_bands[0] / self.freq_bands[-1], 1.0 / n_levels).item()
                base_resolution_high = torch.round(
                    self.basis_reso[n_levels // 2] * self.freq_bands[n_levels // 2]).long().item()
            log2_hashmap_size = np.round(np.log2(self.T_basis / 3 / np.mean(self.basis_dims)))
            # per_level_scale = torch.pow((self.basis_reso[-1] * self.freq_bands[-1])/(self.basis_reso[0] * self.freq_bands[0]),1.0/n_levels).item()
            # per_level_scale = torch.pow(self.freq_bands[0] / self.freq_bands[-1], 1.0 / n_levels).item()

            basises = []
            if len(self.basis_dims) == 1 or sum(self.basis_dims) > 32:
                encoding_config_low = {
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": sum(self.basis_dims) // n_levels,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution_low,
                    "per_level_scale": per_level_scale  # 1.25992 #1.38191 #
                }

                basises.append(tcnn.Encoding(
                    n_input_dims=self.in_dim,
                    encoding_config=encoding_config_low))
            else:
                encoding_config_low = {
                    "otype": "HashGrid",
                    "n_levels": n_levels // 2,
                    "n_features_per_level": min(16, self.basis_dims[0]),
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution_low,
                    "per_level_scale": per_level_scale  # 1.25992 #1.38191 #
                }

                encoding_config_high = {
                    "otype": "HashGrid",
                    "n_levels": n_levels // 2,
                    "n_features_per_level": self.basis_dims[-1],
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution_high,
                    "per_level_scale": per_level_scale  # 1.25992 #1.38191 #
                }

                basises = []
                basises.append(tcnn.Encoding(
                    n_input_dims=self.in_dim,
                    encoding_config=encoding_config_low))
                if self.basis_dims[0] == 32:
                    basises.append(tcnn.Encoding(
                        n_input_dims=self.in_dim,
                        encoding_config=encoding_config_low))
                basises.append(tcnn.Encoding(
                    n_input_dims=self.in_dim,
                    encoding_config=encoding_config_high))

            return torch.nn.ParameterList(basises)
        else:
            basises, coeffs, n_params_basis = [], [], 0
            # in_dim = self.in_dim if 'images' != self.cfg.defaults.mode else self.in_dim - 1
            for i, (basis_dim, reso) in enumerate(zip(self.basis_dims, self.basis_reso)):
                # reso_cur = N_to_reso(reso, aabb)[::-1]
                if 'mlp' in self.basis_type:
                    basises.append(MLPMixer(self.in_dim, basis_dim, num_layers=2, \
                                            hidden_dim=64, pe=4).to(self.device))
                elif 'grid' in self.basis_type:
                    basises.append(torch.nn.Parameter(dct_dict(int(np.power(basis_dim, 1. / self.in_dim) + 1), reso,
                                                               n_selete=basis_dim, dim=self.in_dim).reshape(
                        [1, basis_dim] + [reso] * self.in_dim).to(self.device)))
                    # basises.append(torch.nn.Parameter(torch.ones([1, basis_dim] + [reso] * self.in_dim).to(self.device)))
                elif 'vm' in self.basis_type:
                    reso_level = N_to_vm_reso(reso ** self.in_dim, self.aabb[:, :self.in_dim])
                    for i in range(len(self.matMode)):
                        mat_id_0, mat_id_1 = self.matMode[i]
                        basises.append(torch.nn.Parameter(
                            0.1 * torch.randn((1, basis_dim, reso_level[mat_id_1], reso_level[mat_id_0]),
                                              device=self.device)))
                elif 'cp' in self.basis_type:
                    for _ in range(self.in_dim - 1):
                        basises.append(torch.nn.Parameter(
                            0.1 * torch.randn((1, basis_dim, max(reso, 128), 1), device=self.device)))
                elif 'x' in self.basis_type:
                    continue
                    # basises.append(torch.nn.Parameter(0.1 * torch.randn((1, basis_dim, 1, 1), device=self.device)))
                # elif 'vec' in self.basis_type:
                #     for _ in range(self.in_dim):
                #         basises.append(torch.nn.Parameter(0.1 * torch.randn((1, basis_dim, reso*2, 1))))
        return torch.nn.ParameterList(basises)

    def get_coeff(self, xyz_sampled):
        N_points, dim = xyz_sampled.shape
        # print(xyz_sampled.shape, self.aabb)
        in_dim = self.in_dim
        # in_dim = self.in_dim + 1  if 'images' == self.cfg.defaults.mode else self.in_dim
        pts = self.normalize_coord(xyz_sampled).view([1, -1] + [1] * (dim - 1) + [dim])

        if self.coeff_type in 'hash':
            coeffs = self.coeffs(pts * 0.5 + 0.5).float()
        elif 'grid' in self.coeff_type:
            coeffs = F.grid_sample(self.coeffs[self.scene_idx], pts, mode=self.cfg.model.coef_mode, align_corners=False,
                                   padding_mode='border').view(-1, N_points).t()
        elif 'vec' in self.coeff_type:
            pts = pts.view(1, -1, 1, in_dim)
            idx = (self.scene_idx + 0.5) / self.n_scene * 2 - 1
            pts = torch.stack((torch.ones_like(pts[..., 0]) * idx, pts[..., 0]), dim=-1)
            coeffs = F.grid_sample(self.coeffs[0], pts, mode=self.cfg.model.coef_mode,
                                   align_corners=False, padding_mode='border').view(-1, N_points).t()
        elif 'cp' in self.coeff_type:
            pts = pts.squeeze(2)
            idx = (self.scene_idx + 0.5) / self.n_scene * 2 - 1
            pts = torch.stack((torch.ones_like(pts) * idx, pts), dim=-2)

            coeffs = F.grid_sample(self.coeffs[0], pts[..., 0], mode=self.cfg.model.coef_mode,
                                   align_corners=False, padding_mode='border').view(-1, N_points).t()
            for i in range(1, in_dim):
                coeffs = coeffs * F.grid_sample(self.coeffs[i], pts[..., i], mode=self.cfg.model.coef_mode,
                                                align_corners=False, padding_mode='border').view(-1, N_points).t()
        elif 'vm' in self.coeff_type:
            # pts = torch.flip(pts, dims=(-1,))
            pts = pts.squeeze(2)
            idx = (self.scene_idx + 0.5) / self.n_scene * 2 - 1
            pts = torch.stack((torch.ones_like(pts) * idx, pts), dim=-2)

            coeffs = []
            for i in range(in_dim):
                coeffs.append(F.grid_sample(self.coeffs[i], pts[..., self.vecMode[i]], mode=self.cfg.model.coef_mode,
                                            align_corners=False, padding_mode='border').view(-1, N_points).t())
            coeffs = torch.cat(coeffs, dim=-1)
        elif 'mlp' in self.coeff_type:
            coeffs = self.coeffs[self.scene_idx](pts.view(N_points, in_dim))
        elif 'hash' in self.coeff_type:
            coeffs = self.coeffs[self.scene_idx]((pts.view(N_points, in_dim) + 1) / 2)
        return coeffs

    def get_basis(self, x):
        N_points = x.shape[0]
        if 'images' == self.cfg.defaults.mode:
            x = x[..., :-1]

        if 'hash' in self.basis_type:
            x = (x - self.aabb[0]) / torch.max(self.aabb[1] - self.aabb[0])
            if len(self.basises) == 1:
                basises = self.basises[0](x).float()
            if len(self.basises) == 2:
                basises = torch.cat((self.basises[0](x), self.basises[1](x)), dim=-1).float()
            elif len(self.basises) == 3:
                basises = torch.cat((self.basises[0](x), self.basises[1](x), self.basises[2](x)), dim=-1).float()
        else:
            freq_len = len(self.freq_bands)
            xyz = grid_mapping(x, self.freq_bands, self.aabb[:, :self.in_dim], self.cfg.model.basis_mapping).view(1, *(
                        [1] * (self.in_dim - 1)), -1, self.in_dim, freq_len)
            basises = []
            for i in range(freq_len):
                if 'mlp' in self.basis_type:
                    basises.append(self.basises[i](xyz[..., i].view(-1, self.in_dim)))
                elif 'grid' in self.basis_type:
                    basises.append(
                        F.grid_sample(self.basises[i], xyz[..., i], mode=self.cfg.model.basis_mode,
                                      align_corners=True).view(-1, N_points).T)
                elif 'vm' in self.basis_type:
                    coordinate_mat = torch.stack((xyz[..., self.matMode[0], i], xyz[..., self.matMode[1], i],
                                                  xyz[..., self.matMode[2], i])).view(3, -1, 1, 2)
                    for idx_mat in range(self.in_dim):
                        basises.append(F.grid_sample(self.basises[i * self.in_dim + idx_mat], coordinate_mat[[idx_mat]],
                                                     align_corners=True).view(-1, x.shape[0]).t())
                elif 'cp' in self.basis_type:
                    for idx_axis in range(self.in_dim - 1):
                        coordinate_vec = torch.stack(
                            (torch.zeros_like(xyz[..., idx_axis + 1, i]), xyz[..., idx_axis + 1, i]), dim=-1).squeeze(2)
                        if 0 == idx_axis:
                            basises_level = F.grid_sample(self.basises[i * (self.in_dim - 1) + idx_axis],
                                                          coordinate_vec,
                                                          align_corners=True).view(-1, x.shape[0]).t()
                        else:
                            basises_level = basises_level * F.grid_sample(
                                self.basises[i * (self.in_dim - 1) + idx_axis], coordinate_vec,
                                align_corners=True).view(-1, x.shape[0]).t()
                            # basises_level = torch.cat((basises_level, F.grid_sample(
                            #     self.basises[i * (self.in_dim - 1) + idx_axis], coordinate_vec,
                            #     align_corners=True).view(-1, x.shape[0]).t()),dim=-1)
                    basises.append(basises_level)
                elif 'x' in self.basis_type:
                    basises.append(xyz[..., i].view(x.shape[0], -1))
            if isinstance(basises, list):
                basises = torch.cat(basises, dim=-1)
            if 'vm' in self.basis_type:  # switch order
                basises = basises.view(x.shape[0], freq_len, -1).permute(0, 2, 1).reshape(x.shape[0], -1)
        return basises

    @torch.no_grad()
    def normalize_basis(self):
        for basis in self.basises:
            basis.data = basis.data / torch.norm(basis.data, dim=(2, 3), keepdim=True)

    def get_coding(self, x):
        if self.cfg.model.coeff_type != 'none' and self.cfg.model.basis_type != 'none':
            coeff = self.get_coeff(x)
            basises = self.get_basis(x)
            # return basises, coeff
            return basises * coeff, coeff
            # return torch.cat((basises, coeff),dim=-1), coeff
        elif self.cfg.model.coeff_type != 'none':
            coeff = self.get_coeff(x)
            return coeff, coeff
        elif self.cfg.model.basis_type != 'none':
            basises = self.get_basis(x)
            return basises, basises

    # def get_coding_set(self, x):
    #     basis = self.get_basis(x[..., :-1])
    #     coeff = self.get_coeff(x)
    #     return basis*coeff, coeff

    def n_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        if 'fix' in self.cfg.model.basis_type:
            total -= self.T_basis
        return total

    def get_optparam_groups(self, lr_small=0.001, lr_large=0.02):
        grad_vars = []
        if self.cfg.training.linear_mat:
            grad_vars += [{'params': self.linear_mat.parameters(), 'lr': lr_small}]

        if 'none' != self.coeff_type and self.cfg.training.coeff:
            grad_vars += [{'params': self.coeffs.parameters(), 'lr': lr_large}]

        if 'fix' not in self.cfg.model.basis_type and 'none' != self.cfg.model.basis_type and self.cfg.training.basis:
            grad_vars += [{'params': self.basises.parameters(), 'lr': lr_large}]

            # elif isinstance(self.basises,list):
            # grad_vars += [{'params': self.basises, 'lr': lr_large}] if  self.coeff_type != 'mlp'  \
            #     else [{'params': self.basises.parameters(), 'lr': lr_large}]

        if 'reconstruction' in self.cfg.defaults.mode and self.cfg.training.renderModule:
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_small}]
        return grad_vars

    def set_optimizable(self, items, statue):
        for item in items:
            if item == 'basis' and self.cfg.model.basis_type != 'none':
                for item in self.basises:
                    item.requires_grad = statue
            elif item == 'coeff' and self.cfg.model.coeff_type != 'none':
                for item in self.basises:
                    item.requires_grad = statue
            elif item == 'proj':
                self.linear_mat.requires_grad = statue
            elif item == 'renderer':
                self.renderModule.requires_grad = statue

    def TV_loss(self, reg):
        total = 0
        for idx in range(len(self.basises)):
            total = total + reg(self.basises[idx]) * 1e-2
        return total

    def sample_point_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.cfg.dataset.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0, :self.in_dim] > rays_pts) | (rays_pts > self.aabb[1, :self.in_dim])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_point(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1, :self.in_dim] - rays_o) / vec
        rate_b = (self.aabb[0, :self.in_dim] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=0.05, max=1e3)
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = self.stepSize * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0, :self.in_dim] > rays_pts) | (rays_pts > self.aabb[1, :self.in_dim])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    # def sample_point(self, rays_o, rays_d, is_train=True, N_samples=-1):
    #     N_samples = N_samples if N_samples>0 else self.nSamples
    #
    #     N_inner,N_outer = 3*N_samples//4,N_samples//4
    #     b_inner = torch.linspace(0.05, 2, N_inner+1).to(self.device)
    #     b_outer = 2 / torch.linspace(1, 1/16, N_outer+1).to(self.device)
    #
    #     if is_train:
    #         rng = torch.rand((N_inner+N_outer),device=self.device)
    #         interpx = torch.cat([
    #             b_inner[1:]*rng[:N_inner] + b_inner[:-1]*(1-rng[:N_inner]),
    #             b_outer[1:]*rng[N_inner:] + b_outer[:-1]*(1-rng[N_inner:]),
    #         ])[None]
    #     else:
    #         interpx = torch.cat([
    #             (b_inner[1:] + b_inner[:-1]) * 0.5,
    #             (b_outer[1:] + b_outer[:-1]) * 0.5,
    #         ])[None]
    #
    #     rays_pts = rays_o[:,None,:] + rays_d[:,None,:] * interpx[...,None]
    #
    #     # norm = rays_pts.norm(dim=-1, keepdim=True)
    #     norm = rays_pts.abs().amax(dim=-1, keepdim=True)
    #     inner_mask = (norm<=1)
    #     rays_pts = torch.where(
    #         inner_mask,
    #         rays_pts,
    #         rays_pts / norm * ((1+self.bg_len) - self.bg_len/norm)
    #     )
    #
    #     # interpx = torch.norm(rays_pts - rays_o[:,None,:],dim=-1)
    #     return rays_pts, interpx, inner_mask.squeeze(-1)

    def sample_point_unbound(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples

        N_inner, N_outer = 3 * N_samples // 4, N_samples // 4
        b_inner = torch.linspace(0, 2, N_inner + 1).to(self.device)
        b_outer = 2 / torch.linspace(1, 1 / 16, N_outer + 1).to(self.device)

        if is_train:
            rng = torch.rand((N_inner + N_outer), device=self.device)
            interpx = torch.cat([
                b_inner[1:] * rng[:N_inner] + b_inner[:-1] * (1 - rng[:N_inner]),
                b_outer[1:] * rng[N_inner:] + b_outer[:-1] * (1 - rng[N_inner:]),
            ])[None]
        else:
            interpx = torch.cat([
                (b_inner[1:] + b_inner[:-1]) * 0.5,
                (b_outer[1:] + b_outer[:-1]) * 0.5,
            ])[None]

        rays_pts = rays_o[:, None, :] + rays_d[:, None, :] * interpx[..., None]

        # norm = rays_pts.norm(dim=-1, keepdim=True)
        norm = rays_pts.abs().amax(dim=-1, keepdim=True)
        inner_mask = (norm <= 1)
        rays_pts = torch.where(
            inner_mask,
            rays_pts,
            rays_pts / norm * ((1 + self.bg_len) - self.bg_len / norm)
        )

        # interpx = torch.norm(rays_pts - rays_o[:,None,:],dim=-1)
        return rays_pts, interpx, inner_mask.squeeze(-1)

    def normalize_coord(self, xyz_sampled):
        invaabbSize = 2.0 / (self.aabb[1] - self.aabb[0])
        return (xyz_sampled - self.aabb[0]) * invaabbSize - 1

    def basis2density(self, density_features):
        if self.cfg.renderer.fea2denseAct == "softplus":
            return F.softplus(density_features + self.cfg.renderer.density_shift)
        elif self.cfg.renderer.fea2denseAct == "relu":
            return F.relu(density_features + self.cfg.renderer.density_shift)

    @torch.no_grad()
    def cal_mean_coef(self, state_dict):
        if 'grid' in self.coeff_type or 'mlp' in self.coeff_type:
            key_list = []
            for item in state_dict.keys():
                if 'coeffs.0' in item:
                    key_list.append(item)

            for key in key_list:
                average = torch.zeros_like(state_dict[key])
                for i in range(self.n_scene):
                    item = key.replace('0', f'{i}', 1)
                    average += state_dict[item]
                    state_dict.pop(item, None)
                average /= self.n_scene
                state_dict[key] = average
        elif 'vec' in self.coeff_type:
            state_dict['coeffs.0'] = torch.mean(state_dict['coeffs.0'], dim=-1, keepdim=True)
        elif 'cp' in self.coeff_type or 'vm' in self.coeff_type:
            for i in range(3):
                state_dict[f'coeffs.{i}'] = torch.mean(state_dict[f'coeffs.{i}'], dim=-1, keepdim=True)

        return state_dict

    def save(self, path):
        ckpt = {'state_dict': self.state_dict(), 'cfg': self.cfg}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})

        # average the coeff for saving if batch training
        if 'reconstruction' in self.cfg.defaults.mode:
            ckpt['state_dict'] = self.cal_mean_coef(ckpt['state_dict'])
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device),
                                           alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])
        volumeSize = N_to_reso(self.cfg.training.volume_resoFinal ** self.in_dim, self.aabb)
        self.update_renderParams(volumeSize)

    def update_renderParams(self, gridSize):
        # print("aabb", self.aabb.view(-1))
        # print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        # self.invaabbSize = 2.0/self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(units) * self.cfg.renderer.step_ratio
        aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((aabbDiag / self.stepSize).item()) + 1
        # print("sampling step size: ", self.stepSize)
        # print("sampling number: ", self.nSamples)

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_renderParams(res_target)

        if self.cfg.dataset.dataset_name == 'google_objs' and self.n_scene == 1 and self.cfg.model.coeff_type == 'grid':
            coeffs = [
                F.interpolate(self.coeffs[0].data, size=None, scale_factor=1.3, align_corners=True, mode='trilinear')]
            self.coeffs = torch.nn.ParameterList(coeffs)
        # elif self.cfg.model.coeff_type =='vm':
        #     for i in range(len(self.vecMode)):
        #         vec_id = self.vecMode[i]
        #         mat_id_0, mat_id_1 = self.matMode[i]
        #
        #         self.basises[i] = torch.nn.Parameter(
        #             F.interpolate(self.basises[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]),
        #                           mode='bilinear',
        #                           align_corners=True))
        #         # self.coeffs[i] = torch.nn.Parameter(
        #         #     F.interpolate(self.coeffs[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        # coeffs = F.interpolate(self.coeffs.data, size=None, scale_factor=2.0, align_corners=True,mode='trilinear')
        # self.coeffs = torch.nn.Parameter(coeffs)
        # print(coeffs.shape)

        # basises = []
        # for basis in self.basises:
        #     basises.append(torch.nn.Parameter(
        #         F.interpolate(basis.data, scale_factor=1.3, mode='trilinear',align_corners=True)))
        #     print(basises[-1].shape)
        # self.basises = torch.nn.ParameterList(basises)
        # print(f'upsamping to {res_target}')

    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            # xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            feats, _ = self.get_coding(xyz_locs[alpha_mask])
            validsigma = self.linear_mat(feats, is_train=False)[..., 0]
            sigma[alpha_mask] = self.basis2density(validsigma)

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None, times=16):

        gridSize = self.gridSize.tolist() if gridSize is None else gridSize

        aabbSize = self.inward_aabb[1] - self.inward_aabb[0]
        units = aabbSize / (torch.LongTensor(gridSize).to(self.device) - 1)
        units_half = 1.0 / (torch.LongTensor(gridSize) - 1) * 0.5
        stepSize = torch.mean(units)

        samples = torch.stack(torch.meshgrid(
            [torch.linspace(units_half[0], 1 - units_half[0], gridSize[0]),
             torch.linspace(units_half[1], 1 - units_half[1], gridSize[1]),
             torch.linspace(units_half[2], 1 - units_half[2], gridSize[2])], indexing='ij'
        ), -1).to(self.device)
        dense_xyz = self.inward_aabb[0] * (1 - samples) + self.inward_aabb[1] * samples

        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for _ in range(times):
            for i in range(gridSize[2]):
                shiftment = (torch.rand(dense_xyz[i].shape) * 2 - 1).to(self.device) * (
                            units / 2 * 1.2) if times > 1 else 0.0
                alpha[i] += self.compute_alpha((dense_xyz[i] + shiftment).view(-1, 3),
                                               stepSize * self.cfg.renderer.distance_scale).view(
                    (gridSize[1], gridSize[0]))
        return alpha / times, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200), is_update_alphaMask=False):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = alpha.clamp(0, 1)[None, None]
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        # alpha[alpha>=self.alphaMask_thres] = 1
        # alpha[alpha<self.alphaMask_thres] = 0

        # filter floaters
        min_size = np.mean(alpha.shape[-3:]).item()
        alphaMask_thres = self.cfg.renderer.alphaMask_thres if is_update_alphaMask else 0.08
        if self.is_unbound:
            alphaMask_thres = 0.04
            alpha = (alpha >= alphaMask_thres).float()
        else:
            alpha = skimage.morphology.remove_small_objects(alpha.cpu().numpy() >= alphaMask_thres, min_size=min_size,
                                                            connectivity=1)
            alpha = torch.FloatTensor(alpha).to(self.device)
        # torch.save(alpha, '/home/anpei/code/TensoRF_draft//alpha.th')

        if is_update_alphaMask:
            self.alphaMask = AlphaGridMask(self.device, self.inward_aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)
        if not self.is_unbound:
            pad = (xyz_max - xyz_min) / 20
            xyz_min -= pad
            xyz_max += pad

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb

    @torch.no_grad()
    def shrink(self, new_aabb):
        print(f'=======> shrinking ...')
        # self.setup_params(new_aabb.tolist())
        # basises, coeffs =  self.init_basis(), self.init_coef()

        if self.cfg.model.coeff_type != 'none':
            del self.coeffs
            self.coeffs = self.init_coef()

        if self.cfg.model.basis_type != 'none':
            del self.basises
            self.basises = self.init_basis()

        self.aabb = self.inward_aabb = new_aabb
        self.cfg.dataset.aabb = self.aabb.tolist()
        self.update_renderParams(self.gridSize.tolist())

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()
        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        length_current = 0
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_point(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            # mask_filtered.append(mask_inbbox.cpu())
            length = torch.sum(mask_inbbox)
            all_rays[length_current:length_current + length], all_rgbs[length_current:length_current + length] = \
            rays_chunk[mask_inbbox].cpu(), all_rgbs[idx_chunk][mask_inbbox.cpu()]
            length_current += length

        # mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time() - tt} s. ray mask ratio: {length_current / N}')
        return all_rays[:length_current], all_rgbs[:length_current]

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if self.is_unbound:
            xyz_sampled, z_vals, inner_mask = self.sample_point_unbound(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                        N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], z_vals[:, -1:] - z_vals[:, -2:-1]), dim=-1)
        elif ndc_ray:
            xyz_sampled, z_vals, inner_mask = self.sample_point_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                    N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, inner_mask = self.sample_point(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        # if 'reconstructions' == self.cfg.defaults.mode:
        #     assert 7 == rays_chunk.shape[-1]
        #     self.scene_idx = int(rays_chunk[0, -1])
        # else:
        #     self.scene_idx = 0

        ray_valid = torch.ones_like(xyz_sampled[..., 0]).bool() if self.is_unbound else inner_mask
        if self.alphaMask is not None:
            alpha_inner_valid = self.alphaMask.sample_alpha(xyz_sampled[inner_mask]) > 0.5
            ray_valid[inner_mask.clone()] = alpha_inner_valid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        coeffs = torch.zeros((1, sum(self.cfg.model.basis_dims)), device=xyz_sampled.device)
        if ray_valid.any():
            # xyz_sampled = self.normalize_coord(xyz_sampled[ray_valid])
            feats, coeffs = self.get_coding(xyz_sampled[ray_valid])
            feat = self.linear_mat(feats, is_train=is_train)
            sigma[ray_valid] = self.basis2density(feat[..., 0])

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.cfg.renderer.distance_scale)
        app_mask = weight > self.cfg.renderer.rayMarch_weight_thres
        ray_valid_new = torch.logical_and(ray_valid, app_mask)
        app_mask = ray_valid_new[ray_valid]

        if app_mask.any():
            valid_rgbs = self.renderModule(viewdirs[ray_valid_new], feat[app_mask, 1:])
            rgb[ray_valid_new] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
        #     depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map, coeffs