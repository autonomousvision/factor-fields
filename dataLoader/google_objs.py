import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import glob
from scipy.spatial.transform import Rotation as R

from .ray_utils import *

def fps_downsample(points, n_points_to_sample):
    selected_points = np.zeros((n_points_to_sample, 3))
    selected_idxs = []
    dist = np.ones(points.shape[0]) * 100
    for i in range(n_points_to_sample):
        idx = np.argmax(dist).tolist()
        selected_points[i] = points[idx]
        print(idx)
        selected_idxs.extend([idx])
        dist_ = ((points - selected_points[i]) ** 2).sum(-1)
        dist = np.minimum(dist, dist_).tolist()

    return selected_idxs

############################# Get Spherical Path #############################

def pose_spherical_nerf(euler, radius=1.8, ep=1):
    c2ws_render = np.eye(4)
    c2ws_render[:3,:3] =  R.from_euler('xyz', euler, degrees=True).as_matrix()
    # 保留旋转矩阵的最后一列再乘个系数就能当作位置？
    c2ws_render[:3,3]  = c2ws_render[:3,:3] @ np.array([0.0,0.0,ep*radius])
    return c2ws_render

def nerf_video_path(c2ws, theta_range=10,phi_range=20,N_views=120,radius=1.3,ep=-1):
    c2ws = torch.tensor(c2ws)
    mean_position = torch.mean(c2ws[:,:3, 3],dim=0).reshape(1,3).cpu().numpy()
    rotvec = []
    for i in range(c2ws.shape[0]):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
    # 采用欧拉角做平均的方法求旋转矩阵的平均
    rotvec = np.mean(np.stack(rotvec), axis=0)
    render_poses = [pose_spherical_nerf(rotvec+np.array([angle,0.0,-phi_range]), radius=radius, ep=ep) for angle in np.linspace(-theta_range,theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([theta_range,0.0,angle]), radius=radius, ep=ep) for angle in np.linspace(-phi_range,phi_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([angle,0.0,phi_range]), radius=radius, ep=ep) for angle in np.linspace(theta_range,-theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([-theta_range,0.0,angle]), radius=radius, ep=ep) for angle in np.linspace(phi_range,-phi_range,N_views//4, endpoint=False)]

    return render_poses

def _interpolate_trajectory(c2ws, num_views: int = 300):
    """calculate interpolate path"""

    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    key_rots = Rotation.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(c2ws)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    render_c2ws = []
    for i in range(num_views):
        time = float(i) / num_views * (len(c2ws) - 1)
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_c2ws.append(c2w)
    return np.stack(render_c2ws, axis=0)

def google_objs_path(c2ws, N_views=150):
    positions = c2ws[:, :3, 3]
    selected_idxs = fps_downsample(positions, 3)
    selected_idxs.append(selected_idxs[0])
    return _interpolate_trajectory(c2ws[selected_idxs].numpy(), N_views)



class GoogleObjsDataset(Dataset):
    def __init__(self, cfg, split="train", batch_size=4096):

        # self.N_vis = N_vis
        self.cfg = cfg
        self.root_dir = cfg.datadir
        self.split = split
        self.batch_size = batch_size
        self.is_stack = False if "train" == split else True
        self.downsample = cfg.get(f"downsample_{self.split}")
        self.img_wh = (int(512 / self.downsample), int(512 / self.downsample))
        self.define_transforms()
        train_scene_idxs = sorted(cfg.train_scene_list)
        test_scene_idxs = cfg.test_scene_list
        if len(train_scene_idxs)==2:
            train_scene_idxs = list(range(train_scene_idxs[0],train_scene_idxs[1]))
        self.scene_idxs = train_scene_idxs if self.split=='train' else test_scene_idxs
        self.train_views = cfg.train_views
        self.scene_num = len(self.scene_idxs)

        if 'test' == self.split:
            self.test_index = train_scene_idxs.index(test_scene_idxs[0])


        # self.rot = torch.tensor([[0.65561799, -0.65561799, 0.37460659],
        #                          [0.73729737, 0.44876192, -0.50498052],
        #                          [0.16296514, 0.60727077, 0.77760181]])

        self.scene_bbox = [[-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, self.scene_num]]
        # self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.white_bg = True
        self.near_far = [0.4, 1.6]

        #################################

        # self.folder_path = datadir
        # self.num_source_views = args.num_source_views
        # self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.scene_path_list = sorted(glob.glob(os.path.join(self.root_dir, "*/")))

        all_rgb_files = []
        # all_depth_files = []
        all_pose_files = []
        all_intrinsics_files = []
        num_files = 250

        for i, scene_idx in enumerate(self.scene_idxs):

            scene_path = self.scene_path_list[scene_idx]
            # print(scene_idx,scene_path)

            rgb_files = [
                os.path.join(scene_path, "rgb", f)
                for f in sorted(os.listdir(os.path.join(scene_path, "rgb")))
            ]
            # depth_files = [os.path.join(scene_path, 'depth', f)
            #              for f in sorted(os.listdir(os.path.join(scene_path, 'depth')))]
            pose_files = [
                f.replace("rgb", "pose").replace("png", "txt") for f in rgb_files
            ]
            intrinsics_files = [
                f.replace("rgb", "intrinsics").replace("png", "txt") for f in rgb_files
            ]

            if (
                np.min([len(rgb_files), len(pose_files), len(intrinsics_files)])
                < num_files
            ):
                print(scene_path)
                continue

            all_rgb_files.append(rgb_files)
            # all_depth_files.append(depth_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        # self.all_depth_files = np.array(all_depth_files)[index]
        self.all_pose_files = np.array(all_pose_files)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

        if self.split=='test' or self.scene_num==1:
            self.read_meta()
        else:
            self.train_idxs = range(self.train_views)
        # self.define_proj_mat()

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):

        assert (
            len(self.all_rgb_files)
            == len(self.all_pose_files)
            == len(self.all_intrinsics_files)
        )

        self.all_image_paths = []
        self.all_poses = []
        self.all_rays = []
        self.all_rgbs = []
        for idx in range(len(self.all_rgb_files)):
            rgb_files = self.all_rgb_files[idx]
            pose_files = self.all_pose_files[idx]
            intrinsics_files = self.all_intrinsics_files[idx]

            intrinsics = np.loadtxt(intrinsics_files[0])
            index_3x3 = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10])
            self.intrinsics = intrinsics[index_3x3]

            w, h = self.img_wh
            self.focal = self.intrinsics[0]
            self.focal *= (
                self.img_wh[0] / 512
            )  # modify focal length to match size self.img_wh

            # ray directions for all pixels, same for all images (same H, W, focal)
            self.directions = get_ray_directions(
                h, w, [self.focal, self.focal]
            )  # (h, w, 3)
            self.directions = self.directions / torch.norm(
                self.directions, dim=-1, keepdim=True
            )
            self.intrinsics = torch.tensor(
                [[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]
            ).float()

            self.scene_image_paths = []
            self.scene_poses = []
            self.scene_rays = []
            self.scene_rgbs = []
            # self.downsample = 1.0

            img_eval_interval = (
                1  # if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
            )
            if "train" == self.split:
                cam_xyzs = []
                # for i in range(len(pose_files)):
                for i in range(100):
                    pose = np.loadtxt(pose_files[i])
                    cam_xyzs.append([pose[3], pose[7], pose[11]])
                cam_xyzs = np.array(cam_xyzs)
                idxs = fps_downsample(cam_xyzs, min(self.train_views, len(rgb_files)))
                self.train_idxs = idxs
                print("train idxs:", idxs)
            else:
                idxs = list(range(100, 200))


            for i in tqdm(
                idxs, desc=f"Loading data {self.split} ({len(idxs)})"
            ):  # img_list:#


                pose = np.loadtxt(pose_files[i])
                pose = torch.FloatTensor(pose).view(4, 4)
                self.scene_poses += [pose]

                image_path = rgb_files[i]
                self.scene_image_paths += [image_path]
                img = Image.open(image_path)

                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
                # img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.scene_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, pose)  # both (h*w, 3)
                # rays_o, rays_d = rays_o@self.rot, rays_d@self.rot
                self.scene_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

            self.scene_poses = torch.stack(self.scene_poses)

            views = 180
            radius = {183: 1.3, 199: 1.7, 298: 1.5, 467: 1.1, 957: 1.9, 244: 1.2, 963: 1.2, 527: 1.2, 681:1.9,948:1.2}
            self.render_path = google_objs_path(self.scene_poses, N_views=views)

            name = self.scene_path_list[self.scene_idxs[0]].split('/')[-2]
            np.save(f'{self.root_dir}/{name}_render_path.npy', self.render_path)
            # self.render_path = nerf_video_path(self.scene_poses, N_views=views, theta_range=45,phi_range=90, radius=radius[self.scene_idxs[0]],ep=-1)

            if not self.is_stack:
                self.scene_rays = torch.cat(
                    self.scene_rays, 0
                )  # (len(self.meta['frames])*h*w, 3)
                self.scene_rgbs = torch.cat(
                    self.scene_rgbs, 0
                )  # (len(self.meta['frames])*h*w, 3)

            #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
            else:
                self.scene_rays = torch.stack(
                    self.scene_rays, 0
                )  # (len(self.meta['frames]),h*w, 3)
                self.scene_rgbs = torch.stack(self.scene_rgbs, 0).reshape(
                    -1, *self.img_wh[::-1], 3
                )  # (len(self.meta['frames]),h,w,3)
                # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

            ######################## pre-generate and save in mem ########################

            self.all_image_paths.append(self.scene_image_paths)
            self.all_poses.append(self.scene_poses)
            self.all_rays.append(self.scene_rays)
            self.all_rgbs.append(self.scene_rgbs)

        self.all_rays = torch.cat(self.all_rays)
        self.all_rgbs = torch.cat(self.all_rgbs)

    def get_rays(self, idx):
        ######################## pre-generate and save in mem ########################
        return self.all_rays[idx]


    def get_rgbs(self, idx):

        ######################## pre-generate and save in mem ########################
        return self.all_rgbs[idx]


    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = (
            self.intrinsics.unsqueeze(0) @ torch.inverse(self.scene_poses)[:, :3]
        )

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def update_index(self):
        self.scene_idx = torch.randint(0, len(self.all_rgb_files), (1,)).item()

    def __len__(self):
        return 10000000 #len(self.all_rgb_files)

    def __getitem__(self, idx):
        #
        # self.update_index()
        # idx =  self.scene_idx #
        idx = idx % len(self.all_rgb_files)
        # idx = torch.randint(len(self.all_rgb_files), (1,)).item()

        ######################## generate rays on the fly ########################
        rgb_files = self.all_rgb_files[idx]
        pose_files = self.all_pose_files[idx]
        intrinsics_files = self.all_intrinsics_files[idx]

        intrinsics = np.loadtxt(intrinsics_files[0])
        index_3x3 = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10])
        intrinsics = intrinsics[index_3x3]

        w, h = self.img_wh
        focal = intrinsics[0]
        focal *= self.img_wh[0] / 512  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        directions = get_ray_directions(h, w, [focal, focal])  # (h, w, 3)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        intrinsics = torch.tensor(
            [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
        ).float()

        scene_poses = []
        scene_rays = []
        scene_image_paths = []
        scene_rgbs = []
        # downsample = 1.0

        sample_views = 5
        if self.scene_num>1:
            ids = np.random.choice(self.train_idxs, size=sample_views)

            for i in ids:
                image_path = rgb_files[i]
                scene_image_paths += [image_path]
                img = Image.open(image_path)

                idxs = torch.randint(0, w*h, (self.batch_size // sample_views,))

                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
                # img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                scene_rgbs += [img[idxs]]

                pose = np.loadtxt(pose_files[i])
                pose = torch.FloatTensor(pose).view(4, 4)
                scene_poses += [pose]

                rays_o, rays_d = get_rays(directions, pose)  # both (h*w, 3)
                scene_rays += [torch.cat([rays_o[idxs], rays_d[idxs]], 1)]  # (h*w, 6)

            scene_poses = torch.stack(scene_poses)
            if not self.is_stack:
                scene_rays = torch.cat(scene_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                scene_rgbs = torch.cat(scene_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            else:
                scene_rays = torch.stack(scene_rays, 0)  # (len(self.meta['frames]),h*w, 3)
                scene_rgbs = torch.stack(scene_rgbs, 0)  # (len(self.meta['frames]),h*w, 3)


            return {'rays': scene_rays, 'rgbs': scene_rgbs, 'idx': idx}
        else:
            idx_rand = torch.randint(0, len(self.all_rays), (self.batch_size,))
            return {'rays': self.all_rays[idx_rand], 'rgbs': self.all_rgbs[idx_rand]}


