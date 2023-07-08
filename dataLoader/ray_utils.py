import torch, re, json
import numpy as np
from torch import searchsorted
from kornia import create_meshgrid


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# from utils import index_point_feature
def depth2dist(z_vals, cos_angle):
    # z_vals: [N_ray N_sample]
    device = z_vals.device
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * cos_angle.unsqueeze(-1)
    return dists


def ndc2dist(ndc_pts, cos_angle):
    dists = torch.norm(ndc_pts[:, 1:] - ndc_pts[:, :-1], dim=-1)
    dists = torch.cat([dists, 1e10 * cos_angle.unsqueeze(-1)], -1)  # [N_rays, N_samples]
    return dists


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def get_ray_directions_blender(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]+0.5
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -torch.ones_like(i)],
                             -1)  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def ndc_rays_blender(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = (near - rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = 1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = 1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. - 2. * near / rays_o[..., 2]

    d0 = 1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = 1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = 2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def dda(rays_o, rays_d, bbox_3D):
    inv_ray_d = 1.0 / (rays_d + 1e-6)
    t_min = (bbox_3D[:1] - rays_o) * inv_ray_d  # N_rays 3
    t_max = (bbox_3D[1:] - rays_o) * inv_ray_d
    t = torch.stack((t_min, t_max))  # 2 N_rays 3
    t_min = torch.max(torch.min(t, dim=0)[0], dim=-1, keepdim=True)[0]
    t_max = torch.min(torch.max(t, dim=0)[0], dim=-1, keepdim=True)[0]
    return t_min, t_max


def ray_marcher(rays,
                N_samples=64,
                lindisp=False,
                perturb=0,
                bbox_3D=None):
    """
    sample points along the rays
    Inputs:
        rays: ()

    Returns:

    """

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    if bbox_3D is not None:
        # cal aabb boundles
        near, far = dda(rays_o, rays_d, bbox_3D)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not lindisp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    return xyz_coarse_sampled, rays_o, rays_d, z_vals


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale
    
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

def ndc_bbox(all_rays):
    near_min = torch.min(all_rays[...,:3].view(-1,3),dim=0)[0]
    near_max = torch.max(all_rays[..., :3].view(-1, 3), dim=0)[0]
    far_min = torch.min((all_rays[...,:3]+all_rays[...,3:6]).view(-1,3),dim=0)[0]
    far_max = torch.max((all_rays[...,:3]+all_rays[...,3:6]).view(-1, 3), dim=0)[0]
    print(f'===> ndc bbox near_min:{near_min} near_max:{near_max} far_min:{far_min} far_max:{far_max}')
    return torch.stack((torch.minimum(near_min,far_min),torch.maximum(near_max,far_max)))

def pose_from_json(meta, transpose):
    c2ws = []
    for frame in meta['frames']:
        c2ws.append(np.array(frame['transform_matrix'])@transpose)
    return np.stack(c2ws)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions

def normalize(x):
    return x / np.linalg.norm(x)

def rotation_up(poses):
    up = normalize(np.linalg.lstsq(poses[:, :3, 1],np.ones((poses.shape[0],1)))[0])
    rot = rotation_matrix_from_vectors(up,np.array([0.,1.,0.]))
    return rot

def search_orientation(points):
    from scipy.spatial.transform import Rotation as R
    bbox_sizes,rot_mats,bboxs = [],[],[]
    for y_angle in np.linspace(-45,45,15):
        rotvec = np.array([0,y_angle,0])/180*np.pi
        rot = R.from_rotvec(rotvec).as_matrix()
        point_orientation = rot@points
        bbox = np.max(point_orientation,axis=1) - np.min(point_orientation,axis=1)
        bbox_sizes.append(np.prod(bbox))
        rot_mats.append(rot)
        bboxs.append(bbox)
    rot = rot_mats[np.argmin(bbox_sizes)]
    bbox = bboxs[np.argmin(bbox_sizes)]
    return rot,bbox


def load_point_txt(path):
    points = []
    with open(path, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            err = float(els[7])
            if err > 0.25:
                continue
            points.append([float(els[1]), float(els[2]), float(els[3])])

    points = np.stack(points)

    # mask out outerliner with a sphere cover 95 percet poitns
    center = np.mean(points, axis=0)
    dist = np.sqrt(np.sum((points - center) ** 2, axis=1))
    radius_idx = np.argsort(dist)[:int(points.shape[0] * 0.95)]
    points = points[radius_idx]

    return points

p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

def orientation(poses, point_path=None):
    if point_path is not None:
        points = load_point_txt(point_path)
    else:
        points = poses[:,:3,-1]

    # np.savetxt('points_1.txt', points)
    # np.savetxt('poses_center_1.txt', poses[:, :3, -1])

    rot_up = rotation_up(poses)
    poses_new = rot_up[None] @ poses[:, :3]
    points = rot_up[:3, :3] @ points.T

    center = np.mean(poses_new[:,:3,-1], axis=0, keepdims=True)
    poses_new[:, :3, -1] -= center
    points -= center.T

    # scale = 1.5
    rot, bbox = search_orientation(poses_new[:, :3, -1].T)
    poses_new[:, :3, :4] = rot[None] @ poses_new[:, :3, :4]
    points = rot @ points
    aabb = np.stack((np.min(points,axis=1),np.max(points,axis=1)))
    poses_new[:, :3, -1] /= np.min(aabb[1]-aabb[0])
    points /= np.min(aabb[1] - aabb[0])
    aabb = (aabb/np.min(aabb[1]-aabb[0])*1.5).tolist()


    np.savetxt('points.txt', points.T)
    np.savetxt('poses_center.txt', poses_new[:, :3, -1])
    return p34_to_44(poses_new), aabb


def spherify_poses(poses, radus=1):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))*radus
    # print(poses_reset,center,rad)

    scale = 1. / rad
    poses_reset[:, :3, 3] *= scale
    # bds *= sc
    rad *= scale
    # print ('========>',poses_reset)

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    render_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        render_poses.append(p)

    render_poses = np.stack(render_poses, 0)

    # render_poses = np.concatenate([render_poses, np.broadcast_to(poses[0, :3, -1:], render_poses[:, :3, -1:].shape)], -1)
    # poses_reset = np.concatenate(
    #     [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, render_poses, scale
