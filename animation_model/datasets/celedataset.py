import os
import random
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from decord import VideoReader, cpu


class CeleDataset(Dataset):
    def __init__(
            self,
            video_dirs: list,             # original video directory，for example [".../videos"]
            render_dirs: list,            # corresponding render directory，for example [".../render_videos"], share the same video names
            image_size: int = 768,
            sample_margin: int = 30,
            guids: list = ["render", "depth", "normal"],
            bbox_crop=True,
            bbox_resize_ratio=(0.8, 1.2),
            aug_type: str = "Resize",  # "Resize" or "Padding"
            select_face=False,
            max_retry: int = 5,
    ):
        super().__init__()
        assert len(video_dirs) == len(render_dirs)
        self.video_dirs = video_dirs
        self.render_dirs = render_dirs
        self.image_size = image_size
        self.sample_margin = sample_margin
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
        print(f"CeleDataset: {len(data_lst)} videos")
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

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def set_tgt_idx(self, ref_img_idx, video_length):  
        margin = self.sample_margin
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin >= 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        return tgt_img_idx

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        video_path, render_dir, name = self.data_lst[idx]

        src_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        guid_vrs = [VideoReader(os.path.join(render_dir, guid, name), ctx=cpu(0), num_threads=1)
                    for guid in self.guids]
        video_length = min([len(src_vr)] + [len(vr) for vr in guid_vrs])

        for _ in range(self.max_retry):
            ref_img_idx = random.randint(0, video_length - 1)
            tgt_img_idx = self.set_tgt_idx(ref_img_idx, video_length)
            guid_frames = [vr[tgt_img_idx].asnumpy() for vr in guid_vrs]
            if all(f.max() > 0 for f in guid_frames):
                break

        src_frames = src_vr.get_batch([ref_img_idx, tgt_img_idx]).asnumpy()  # (2, H, W, 3) RGB
        h, w = src_frames.shape[1:3]
        ref_img_pil = Image.fromarray(src_frames[0])
        tgt_img_pil = Image.fromarray(src_frames[1])
        tgt_guid_pil_lst = [Image.fromarray(np.ascontiguousarray(f[:h, :w])) for f in guid_frames]

        # augmentation
        state = torch.get_rng_state()

        tgt_img = self.augmentation(tgt_img_pil, self.pixel_transform, state)  
        tgt_guid = self.augmentation(tgt_guid_pil_lst, self.guid_transform, state)  
        ref_img_vae = self.augmentation(ref_img_pil, self.pixel_transform, state)  
        prompt = 'A close up of a person.'

        sample = dict(
            tgt_img=tgt_img,
            tgt_guid=tgt_guid,
            ref_img=ref_img_vae,
            prompt=prompt,
        )

        return sample


if __name__ == '__main__':
    from torchvision.utils import save_image

    train_dataset = CeleDataset(
        video_dirs=['/scratch/cs/faceedit/mengting/data/CelebV-HQ/videos'],
        render_dirs=['/scratch/cs/faceedit/mengting/data/CelebV-HQ/render_videos'],
        image_size=512,
        sample_margin=30,
        guids=["render"],
        bbox_crop=True,
        aug_type="Resize",
    )
    out_dir = "check_samples"
    os.makedirs(out_dir, exist_ok=True)
    bad_lst = []

    for i in range(len(train_dataset)):
        video_path, render_dir, name = train_dataset.data_lst[i]
        src_vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        h, w = src_vr[0].shape[:2]
        reasons = []
        if (h, w) != (512, 512):
            reasons.append(f"src {w}x{h}")

        for guid in train_dataset.guids:
            vr = VideoReader(os.path.join(render_dir, guid, name), ctx=cpu(0), num_threads=1)
            gh, gw = vr[0].shape[:2]
            if gh < h or gw < w:
                reasons.append(f"{guid} {gw}x{gh} smaller than src")
            if len(vr) != len(src_vr):
                reasons.append(f"{guid} len={len(vr)} vs src len={len(src_vr)}")
            idxs = list(range(0, len(vr), max(1, len(vr) // 10)))
            n_black = sum(int(vr[j].asnumpy().max() == 0) for j in idxs)
            if n_black > 0:
                reasons.append(f"{guid} black {n_black}/{len(idxs)}")

        item = train_dataset[i]
        guid_imgs = item["tgt_guid"].split(3, dim=0)
        row = [item["ref_img"] * 0.5 + 0.5, item["tgt_img"] * 0.5 + 0.5] + list(guid_imgs)
        tag = "BAD" if reasons else "ok"
        # save_image(torch.stack(row), os.path.join(out_dir, f"{i:03d}_{tag}_{name[:-4]}.jpg"), nrow=len(row))

        print(f"[{i:03d}] {tag:3s} {name}  src {w}x{h} len={len(src_vr)}  {'; '.join(reasons)}")
        if reasons:
            bad_lst.append((name, reasons))

    print(f"\n{len(bad_lst)} / 50 have problems:")
    for name, reasons in bad_lst:
        print(f"  {name}: {'; '.join(reasons)}")