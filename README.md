# AUAnimation: Semantically-controllable Facial Animation from a Static Image

Official implementation of **AUAnimation**, accepted to *IEEE Transactions on Affective Computing* (TAC).

**Mengting Wei, Yante Li, Licai Sun, Yan Jiang, Ali Salar, Guoying Zhao**

[[Paper]](https://ieeexplore.ieee.org/abstract/document/11702529)

<!-- TODO: add teaser figure, e.g. ![teaser](assets/teaser.png) -->

## Overview

AUAnimation animates a single face image under semantic control from facial Action Units (AUs). The framework has two components:

- **Sem2Geo Mapper**: maps AU intensities to FLAME expression and jaw-pose parameters.
- **Animation Model** (`animation_model/`): a reference-guided video diffusion model that animates the source image from rendered FLAME geometry. It is built on Stable Diffusion v1.5, with AnimateDiff motion modules for temporal consistency.

This repository currently contains the Animation Model. <!-- TODO: Sem2Geo Mapper release -->

## Repository Structure

```
animation_model/
├── configs/            
├── models/             
├── pipelines/          
├── utils/
├── train_s1.py         # stage 1: image-level training
├── train_s2.py         # stage 2: motion module training
└── inference.py
```

## Installation

```bash
git clone https://github.com/weimengting/AUAnimation.git
cd AUAnimation/animation_model
conda create -n auanimation python=3.10 -y
conda activate auanimation
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
 
**Render-video extraction uses [DECA](https://github.com/yfeng95/DECA) in a separate environment. Follow DECA's installation instructions; `ffmpeg` must also be available on the system.**

## Pretrained Models

Download the following models and set their paths in the config files:

| Model | Source | Config key |
|---|---|---|
| Stable Diffusion v1.5 | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `base_model_path` |
| SD VAE (ft-MSE) | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | `vae_model_path` |
| AnimateDiff motion module | [guoyww/animatediff](https://huggingface.co/guoyww/animatediff) | `mm_path` (stage 2 init) |

### AUAnimation checkpoints

Our trained weights are available on [Hugging Face](https://huggingface.co/mengtingwei/AUAnimation):

| File | Description |
|---|---|
| `stage1/denoising_unet.pth` | Denoising UNet (stage 1) |
| `stage1/reference_unet.pth` | Reference UNet (stage 1) |
| `stage1/guidance_encoder_render.pth` | Guidance encoder for rendered FLAME maps (stage 1) |
| `stage2/motion_module.pth` | Motion module (stage 2) |

Download them with:

```bash
hf download mengtingwei/AUAnimation --local-dir checkpoints/AUAnimation
```

Then set the following in `configs/inference/inference.yaml`:

```yaml
ckpt_dir: 'checkpoints/AUAnimation/stage1'
motion_module_path: 'checkpoints/AUAnimation/stage2/motion_module.pth'
```

## Data Preparation

We train on [CelebV-HQ](https://github.com/CelebV-HQ/CelebV-HQ). To keep the file count manageable, guidance maps are stored as **one video per clip**, not as per-frame images. Each render video is frame-aligned with its source video.

### 1. Source videos

Place the cropped face clips in a single folder:

```
CelebV-HQ/
└── videos/
    ├── -2KGPYEFnsU_9.mp4
    ├── ...
```

### 2. Render videos

For every frame, we fit FLAME with [DECA](https://github.com/yfeng95/DECA) and render the face mesh. Results are written as losslessly encoded mp4 files (`libx264rgb -crf 0`), one per source video and guidance type. The file name is the same as the source video's:

```
CelebV-HQ/
├── videos/
│   └── -2KGPYEFnsU_9.mp4
└── render_videos/
    ├── render/                 # rendered shaded mesh (used by default)
    │   └── -2KGPYEFnsU_9.mp4
    ├── depth/                  # optional
    │   └── -2KGPYEFnsU_9.mp4
    └── normal/                 # optional
        └── -2KGPYEFnsU_9.mp4
```

Notes:

- **Subfolder names must match the guidance type names** in the config (`guids` for training, `guidance_types` for inference). For example, `guids: ['render']` reads from `render_videos/render/`, and the corresponding checkpoint is named `guidance_encoder_render.pth`.
- **Odd frame dimensions are padded to even sizes** by ffmpeg. The data loaders crop render frames back to the source resolution.

### 3. How samples are built

- **Stage 1 (image level)**: from each clip, a *reference* frame and a *target* frame are sampled. The model receives the reference image and the guidance map of the **target** frame, and learns to reconstruct the target frame.
- **Stage 2 (video level)**: a clip of `sample_frames` consecutive frames is sampled with stride `sample_rate`, together with the aligned guidance clip and one reference frame from the same video. Only the motion modules are trained; the other weights are initialized from stage 1.

All frames are resized to 512×512.

## Training

Set the data root, render root and pretrained model paths in the configs first. `train_bs` is the per-GPU batch size.

**Stage 1**

```bash
accelerate launch train_s1.py --config configs/train/stage1.yaml
```

**Stage 2**: set `stage1_ckpt_dir` to the stage 1 output.

```bash
accelerate launch train_s2.py --config configs/train/stage2.yaml
```

Keep `sample_frames` ≤ `temporal_position_encoding_max_len` (32).

## Inference

Edit `configs/inference/inference.yaml`:

```yaml
data:
  video_path: '/path/to/CelebV-HQ/videos/-2KGPYEFnsU_9.mp4'   # driving video
  render_dir: '/path/to/CelebV-HQ/render_videos'
  ref_frame_idx: 0        # reference frame taken from the driving video (self-reenactment)
  ref_image_path: null    # set to an external face image for cross-identity animation
  frame_range: [0, 120]   # null = whole video

ckpt_dir: '/path/to/stage1_checkpoint'
motion_module_path: '/path/to/motion_module.pth'
guidance_types: ['render']
context_frames: 24        # should match stage 2 sample_frames
context_overlap: 4
```

Then run:

```bash
python inference.py --config configs/inference/inference.yaml
```

## Citation

```bibtex
@article{wei2026auanimation,
  title   = {AUAnimation: Semantically-controllable Facial Animation from a Static Image},
  author  = {Wei, Mengting and Li, Yante and Sun, Licai and Jiang, Yan and Salar, Ali and Zhao, Guoying},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2026}
}
```

## Acknowledgements

This code builds on [Champ](https://github.com/fudan-generative-vision/champ), [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [MagicAnimate](https://github.com/magic-research/magic-animate), [AnimateDiff](https://github.com/guoyww/AnimateDiff) and [DECA](https://github.com/yfeng95/DECA). We thank the authors for releasing their code.