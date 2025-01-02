# Copyright (c) 2024, Ziwen Chen.

import os
import json
import numpy as np
import PIL.Image as Image
import cv2
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        data_path_text = config.data.data_path
        data_folder = data_path_text.rsplit('/', 1)[0] 
        with open(data_path_text, 'r') as f:
            self.data_path = f.readlines()
        self.data_path = [x.strip() for x in self.data_path]
        self.data_path = [x for x in self.data_path if len(x) > 0]
        for i, data_path in enumerate(self.data_path):
            if not data_path.startswith("/"):
                self.data_path[i] = os.path.join(data_folder, data_path)

    def __len__(self):
        return len(self.data_path)

    def process_frames(self, frames, image_base_dir):
        resize_h = self.config.data.resize_h
        resize_w = self.config.data.resize_w
        patch_size = self.config.data.patch_size
        patch_size = patch_size * 2 ** len(self.config.model.get("merge_layers", [])) 
        square_crop = self.config.data.square_crop

        images = [Image.open(os.path.join(image_base_dir, frame["file_path"])) for frame in frames]
        images = np.stack([np.array(image) for image in images]) # (num_frames, H, W, 3)
        if resize_h == -1:
            resize_h = int(resize_w / images.shape[2] * images.shape[1])
        elif resize_w == -1:
            resize_w = int(resize_h / images.shape[1] * images.shape[2])
        resize_h = resize_h // patch_size * patch_size
        resize_w = resize_w // patch_size * patch_size
        images = np.stack([cv2.resize(image, (resize_w, resize_h)) for image in images]) # (num_frames, resize_h, resize_w, 3)
        if square_crop:
            min_size = min(resize_h, resize_w)
            # center crop
            start_h = (resize_h - min_size) // 2
            start_w = (resize_w - min_size) // 2
            images = images[:, start_h:start_h+min_size, start_w:start_w+min_size, :]
        images = images / 255.0
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() # (num_frames, 3, resize_h, resize_w)
        
        h = np.array([frame["h"] for frame in frames])
        w = np.array([frame["w"] for frame in frames])
        fx = np.array([frame["fx"] for frame in frames])
        fy = np.array([frame["fy"] for frame in frames])
        cx = np.array([frame["cx"] for frame in frames])
        cy = np.array([frame["cy"] for frame in frames])
        intrinsics = np.stack([fx, fy, cx, cy], axis=1) # (num_frames, 4)
        intrinsics[:, 0] *= resize_w / w
        intrinsics[:, 1] *= resize_h / h
        intrinsics[:, 2] *= resize_w / w
        intrinsics[:, 3] *= resize_h / h
        if square_crop:
            intrinsics[:, 2] -= start_w
            intrinsics[:, 3] -= start_h
        intrinsics = torch.from_numpy(intrinsics).float()

        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def __getitem__(self, idx):
        data_path = self.data_path[idx]
        data_json = json.load(open(data_path, 'r'))
        scene_name = data_json['scene_name']
        frames = data_json['frames']
        image_base_dir = data_path.rsplit('/', 1)[0]

        # read config
        input_frame_select_type = self.config.data.input_frame_select_type
        target_frame_select_type = self.config.data.target_frame_select_type
        num_input_frames = self.config.data.num_input_frames
        num_target_frames = self.config.data.num_target_frames
        target_has_input = self.config.data.target_has_input
        min_frame_dist = self.config.data.min_frame_dist
        max_frame_dist = self.config.data.max_frame_dist
        if min_frame_dist == "all":
            min_frame_dist = len(frames) - 1
            max_frame_dist = min_frame_dist
        min_frame_dist = min(min_frame_dist, len(frames) - 1)
        max_frame_dist = min(max_frame_dist, len(frames) - 1)
        assert min_frame_dist <= max_frame_dist
        if target_has_input:
            assert min_frame_dist >= max(num_input_frames, num_target_frames) - 1
        else:
            assert min_frame_dist >= num_input_frames + num_target_frames - 1
        frame_dist = np.random.randint(min_frame_dist, max_frame_dist + 1)

        # get frame range
        start_frame_idx = np.random.randint(0, len(frames) - frame_dist)
        end_frame_idx = start_frame_idx + frame_dist
        frame_idx = list(range(start_frame_idx, end_frame_idx + 1))

        # get target frames
        if target_frame_select_type == 'random':
            target_frame_idx = np.random.choice(frame_idx, num_target_frames, replace=False)
        elif target_frame_select_type == 'uniform':
            target_frame_idx = np.linspace(start_frame_idx, end_frame_idx, num_target_frames, dtype=int)
        elif target_frame_select_type == 'uniform_every':
            uniform_every = self.config.data.target_uniform_every
            target_frame_idx = list(range(start_frame_idx, end_frame_idx + 1, uniform_every))
            num_target_frames = len(target_frame_idx)
        else:
            raise NotImplementedError
        target_frame_idx = sorted(target_frame_idx)

        # get input frames
        if not target_has_input:
            frame_idx = [x for x in frame_idx if x not in target_frame_idx]
        if input_frame_select_type == 'random':
            input_frame_idx = np.random.choice(frame_idx, num_input_frames, replace=False)
        elif input_frame_select_type == 'uniform':
            input_frame_idx = np.linspace(0, len(frame_idx) - 1, num_input_frames, dtype=int)
            input_frame_idx = [frame_idx[i] for i in input_frame_idx]
        elif input_frame_select_type == 'kmeans':
            json_key = "fold_"+str(self.config.data.target_uniform_every)+"_kmeans_"+str(num_input_frames)+"_input"
            if json_key in data_json:
                input_frame_idx = data_json[json_key]
            else:
                from sklearn.cluster import KMeans
                w2cs = np.stack([np.array(frames[i]["w2c"]) for i in frame_idx])
                c2ws = np.linalg.inv(w2cs)
                cam_poses = c2ws[:, :3, 3]
                cam_dirs = c2ws[:, :3, 2]
                pos_dirs = np.concatenate([cam_poses, cam_dirs], axis=1)
                cluster_centers = KMeans(n_clusters=num_input_frames, random_state=0, n_init="auto").fit(pos_dirs).cluster_centers_ # (num_input, 6)
                input_frame_idx = []
                for center in cluster_centers:
                    dists = np.linalg.norm(pos_dirs - center, axis=1)
                    input_frame_idx.append(frame_idx[np.argmin(dists)])
                data_json[json_key] = sorted(input_frame_idx)
                # save json
                with open(data_path, 'w') as f:
                    json.dump(data_json, f, indent=4)
        else:
            raise NotImplementedError
        input_frame_idx = sorted(input_frame_idx)

        target_frames = [frames[i] for i in target_frame_idx]
        target_images, target_intr, target_c2ws = self.process_frames(target_frames, image_base_dir)

        input_frames = [frames[i] for i in input_frame_idx]
        input_images, input_intr, input_c2ws = self.process_frames(input_frames, image_base_dir)

        # noramlize input camera poses
        position_avg = input_c2ws[:, :3, 3].mean(0) # (3,)
        forward_avg = input_c2ws[:, :3, 2].mean(0) # (3,)
        down_avg = input_c2ws[:, :3, 1].mean(0) # (3,)
        # gram-schmidt process
        forward_avg = F.normalize(forward_avg, dim=0)
        down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
        right_avg = torch.cross(down_avg, forward_avg)
        pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
        pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)
        pos_avg_inv = torch.inverse(pos_avg)

        # scale scene size
        position_max = input_c2ws[:, :3, 3].abs().max()
        scene_scale = self.config.data.get("scene_scale", 1.0) * position_max
        scene_scale = 1.0 / scene_scale

        ret_dict = {
            "scene_name": scene_name,
            "input_images": input_images,
            "input_intr": input_intr,
            "input_c2ws": input_c2ws,
            "target_images": target_images,
            "target_intr": target_intr,
            "target_c2ws": target_c2ws,
            "pos_avg_inv": pos_avg_inv,
            "scene_scale": scene_scale,
        }

        return ret_dict

if __name__ == "__main__":
    # test dataset
    config = edict()
    config.data = edict()
    config.model = edict()
    config.data.data_path = "example_data/mydesk.txt"
    config.data.resize_h = 128
    config.data.resize_w = 416
    config.data.patch_size = 16
    config.data.square_crop = True
    config.data.input_frame_select_type = "kmeans"
    config.data.target_frame_select_type = "uniform_every"
    config.data.num_input_frames = 32
    config.data.num_target_frames = 8
    config.data.target_has_input = False
    config.data.min_frame_dist = "all"
    config.data.max_frame_dist = 64
    config.data.target_uniform_every = 8

    dataset = Dataset(config)
    print("dataset length:", len(dataset))

    for i in range(len(dataset)):
        data = dataset[i]
        print("scene_name:", data["scene_name"])
        print("input_images:", data["input_images"].shape)
        print("input_intr:", data["input_intr"].shape)
        print("input_c2ws:", data["input_c2ws"].shape)
        print("target_images:", data["target_images"].shape)
        print("target_intr:", data["target_intr"].shape)
        print("target_c2ws:", data["target_c2ws"].shape)
        print("pos_avg_inv:", data["pos_avg_inv"].shape)
        print("scene_scale:", data["scene_scale"])


