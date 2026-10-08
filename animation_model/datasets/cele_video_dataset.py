import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu


class CeleVideoDataset(Dataset):
    def __init__(
            self,
            video_dirs: list,            
            render_dirs: list,           
            image_size: int = 512,
            sample_frames: int = 24,
            sample_rate: int = 4,
            guids: list = ["render"],
            bbox_crop: bool = True,
            bbox_resize_ratio: tuple = (0.8, 1.2),
            aug_type: str = "Resize",
            select_face: bool = False,
            max_retry: int = 5,
    ):
        super().__init__()
        assert len(video_dirs) == len(render_dirs)
        self.video_dirs = video_dirs
        self.render_dirs = render_dirs
        self.image_size = image_size
        self.sample_frames = sample_frames
        self.sample_rate = sample_rate
        self.guids = guids
        self.bbox_crop = bbox_crop
        self.bbox_resize_ratio = bbox_resize_ratio
        self.aug_type = aug_type
        self.select_face = select_face
        self.max_retry = max_retry

        self.data_lst = self.generate_data_lst()
        self.pixel_transform, self.guid_transform = self.setup_transform()

    def generate_data_lst(self):
        data_lst = []
        for video_dir, render_dir in zip(self.video_dirs, self.render_dirs):
            valid = None
            for guid in self.guids:
                names = set(f for f in os.listdir(os.path.join(render_dir, guid))
                            if f.endswith(".mp4") and not f.endswith(".part.mp4"))
                valid = names if valid is None else valid & names
            for name in sorted(os.listdir(video_dir)):
                if name in valid:
                    data_lst.append((os.path.join(video_dir, name), render_dir, name))
        print(f"CeleVideoDataset: {len(data_lst)} videos")
        return data_lst

    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.image_size / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)

        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.image_size, self.image_size
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top

        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img

    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])

            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")

        else:
            pixel_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        return pixel_transform, guid_transform

    def set_clip_idx(self, video_length):
        clip_length = min(video_length, (self.sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        clip_idxes = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_frames, dtype=int
        ).tolist()
        return clip_idxes

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            ret_lst = []
            for img in images:
                if isinstance(img, list):
                    transformed_sub_images = [transform(sub_img) for sub_img in img]
                    sub_ret_tensor = torch.cat(transformed_sub_images, dim=0)  # (c*n, h, w)
                    ret_lst.append(sub_ret_tensor)
                else:
                    transformed_images = transform(img)
                    ret_lst.append(transformed_images)  # (c*1, h, w)
            ret_tensor = torch.stack(ret_lst, dim=0)  # (f, c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        video_path, render_dir, name = self.data_lst[idx]

        src_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        guid_vrs = [VideoReader(os.path.join(render_dir, guid, name), ctx=cpu(0), num_threads=1)
                    for guid in self.guids]
        video_length = min([len(src_vr)] + [len(vr) for vr in guid_vrs])

        for _ in range(self.max_retry):
            clip_idxes = self.set_clip_idx(video_length)
            guid_clips = [vr.get_batch(clip_idxes).asnumpy() for vr in guid_vrs]  # n x (f, H, W, 3)
            if all(clip.reshape(len(clip_idxes), -1).max(1).min() > 0 for clip in guid_clips):
                break

        ref_img_idx = random.randint(0, video_length - 1)
        ref_img_pil = Image.fromarray(src_vr[ref_img_idx].asnumpy())

        tgt_frames = src_vr.get_batch(clip_idxes).asnumpy()  # (f, H, W, 3) RGB
        h, w = tgt_frames.shape[1:3]

        tgt_vidpil_lst = [Image.fromarray(f) for f in tgt_frames]
        # guid frames: [[frame0: n_type x pil], [frame1: n x pil], ...]
        tgt_guid_vidpil_lst = []
        for f_i in range(len(clip_idxes)):
            tgt_guid_vidpil_lst.append(
                [Image.fromarray(np.ascontiguousarray(clip[f_i, :h, :w])) for clip in guid_clips]
            )

        state = torch.get_rng_state()
        tgt_vid = self.augmentation(tgt_vidpil_lst, self.pixel_transform, state)          # (f, 3, h, w)
        tgt_guid_vid = self.augmentation(tgt_guid_vidpil_lst, self.guid_transform, state)  # (f, 3*n, h, w)
        ref_img_vae = self.augmentation(ref_img_pil, self.pixel_transform, state)          # (3, h, w)
        prompt = 'A close up of a person.'

        sample = dict(
            tgt_vid=tgt_vid,
            tgt_guid_vid=tgt_guid_vid,
            ref_img=ref_img_vae,
            prompt=prompt,
        )

        return sample


if __name__ == '__main__':
    from torchvision.utils import save_image

    train_dataset = CeleVideoDataset(
        video_dirs=['/scratch/cs/faceedit/mengting/data/CelebV-HQ/videos'],
        render_dirs=['/scratch/cs/faceedit/mengting/data/CelebV-HQ/render_videos'],
        image_size=512,
        sample_frames=24,
        sample_rate=4,
        guids=["render"],
        bbox_crop=True,
        aug_type="Resize",
    )
    os.makedirs("check_video_samples", exist_ok=True)
    for i in range(5):
        item = train_dataset[i]
        tgt = item["tgt_vid"] * 0.5 + 0.5
        guid = item["tgt_guid_vid"][:, :3]
        save_image(torch.cat([tgt, guid], dim=0), os.path.join("check_video_samples", f"{i}.jpg"), nrow=tgt.shape[0])
        print(i, item["tgt_vid"].shape, item["tgt_guid_vid"].shape, item["ref_img"].shape)