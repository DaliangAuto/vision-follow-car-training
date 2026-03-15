# Autofollow — End-to-End Imitation Learning

English | [中文](README_zh.md)

Autonomous following vehicle project: given the last 5 frames and corresponding speeds, predict steering, throttle, brake, and target validity to implement a safe "follow when target present, stop when absent" control policy.

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Training](#training)
- [Inference & Post-processing](#inference--post-processing)
- [Model Architecture](#model-architecture)
- [Tool Scripts](#tool-scripts)
- [FAQ](#faq)

---

## Overview

| Item | Description |
|------|--------------|
| **Task** | End-to-end imitation learning: predict control from images + speed and detect valid follow target |
| **Input** | 5 consecutive frames + 5 speeds (time window [t-4, t-3, t-2, t-1, t]) |
| **Output** | 4-D: `steer`, `throttle`, `brake_logit`, `target_valid_logit` |
| **Control Logic** | When target_valid: follow; when invalid (no target): safe stop |
| **Frame Rate** | ~10 Hz, 5 frames ≈ 0.5 s |

### Data Types

| Type | Path | Description |
|------|------|-------------|
| **main1** | `dataset/0312/main1/` | Primary driving data, target-present frames, may include short target-absent tail |
| **main2** | `dataset/0312/main2/` | Same as main1 |
| **no_target** | `dataset/0312/no_target/` | Target-absent data, labels: steer=0, throttle=0, brake=1, target_valid=0 |

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
torch>=2.0
torchvision>=0.15
pandas>=1.0
Pillow>=9.0
numpy>=1.20
tqdm>=4.60
PyYAML>=6.0
```

### Recommended

- Python 3.8+
- CUDA 11.x (for training; Jetson supported for inference)

---

## Project Structure

```
autofollow/
├── dataset/
│   ├── 0312/                    # training data (from config.yaml)
│   │   ├── main1/
│   │   │   ├── frames/          # images 000000.jpg, 000001.jpg, ...
│   │   │   └── controls.csv
│   │   ├── main2/
│   │   │   ├── frames/
│   │   │   └── controls.csv
│   │   └── no_target/
│   │       ├── frames/
│   │       └── controls.csv
│   └── raw/                     # raw data for tool sync
├── checkpoints/                 # weights
│   └── 20260313_1223_55/        # per-run dir (YYYYMMDD_HHMM_SS)
│       ├── best_model.pth       # best model
│       ├── epoch_05.pth         # saved from epoch 5 onward
│       └── ...
├── tool/                        # utilities
│   └── verify_dataset.py        # verify photos vs CSV, sort CSV by frame_idx
├── config.yaml                  # training config
├── dataset.py                   # dataset builder
├── model.py                     # model definition
├── train.py                     # training script
├── infer.py                     # inference script
├── utils.py
├── requirements.txt
└── README.md
```

---

## Data Format

### Rules

- `config.yaml` must set `data.data_root` to `dataset/0312` with subdirs `main1`, `main2`, `no_target`
- Each subdir needs `frames/` and `controls.csv` (no_target can fall back to image chunks if no sequential info)

### controls.csv Fields (main1 / main2)

| Field | Description |
|-------|--------------|
| frame_idx | Frame index, must match image filename (e.g. 000302.jpg → 302) |
| image_path | Relative path, e.g. `frames/000123.jpg` |
| steer | Steering [-1, 1] |
| throttle | Throttle [0, 1] |
| brake | Brake [0, 1] |
| speed | Current speed |
| **target_valid** | **Required**. 1=valid follow target present, 0=absent or should not follow |
| gear, ts, seq, ts_ms, raw | Optional (gear, timestamp, etc.) |

### Sample Construction

**main1 / main2:**

- Use 5-frame windows with consecutive `frame_idx` (difference 1)
- Labels from **last frame** `target_valid`:
  - `target_valid > 0.5`: use original steer/throttle, binarize brake, target_valid=1
  - `target_valid ≤ 0.5`: steer=0, throttle=0, brake=1, target_valid=0

**no_target:**

- Prefer controls.csv with consecutive 5-frame windows when available
- Else: non-overlapping image chunks (5 images per chunk)
- Labels fixed: steer=0, throttle=0, brake=1, target_valid=0

---

## Training

### config.yaml

```yaml
data:
  data_root: dataset/0312   # must contain main1, main2, no_target

train:
  batch_size: 32
  epochs: 30
  lr: 0.0001
  weight_decay: 0.0001
  use_type_sampler: true   # main:no_target = 8:1 sampling
```

Data path, batch size, lr, etc. come from config; CLI args override.

### Quick Start

```bash
python train.py
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Config path |
| `--batch_size` | from config | Batch size |
| `--epochs` | from config | Epochs |
| `--lr` | from config | Learning rate |
| `--weight_decay` | from config | AdamW weight decay |
| `--use_type_sampler` | from config | main:no_target=8:1 sampling |

### Examples

```bash
# default (uses config.yaml)
python train.py

# custom config and epochs
python train.py --config config.yaml --epochs 50
```

### Loss & Weights

| Component | Loss | Weight |
|-----------|------|--------|
| steer | SmoothL1 | 2.0 |
| throttle | SmoothL1 | 1.0 |
| brake | BCEWithLogits | 2.0 |
| target_valid | BCEWithLogits | 3.0 |

### Metrics

- **steer MAE**, **throttle MAE**: mean absolute error
- **brake acc**, **target_valid acc**: classification accuracy (sigmoid > 0.5)
- Per-type stats for main / no_target

### Checkpoints

- One dir per run: `checkpoints/YYYYMMDD_HHMM_SS/`
- `best_model.pth`: overwritten when val loss improves
- `epoch_XX.pth`: saved every epoch from epoch 5 onward

---

## Inference & Post-processing

### Model Output (4-D)

| Index | Meaning | Type |
|-------|---------|------|
| 0 | steer | regression |
| 1 | throttle | regression |
| 2 | brake_logit | binary logit (sigmoid → brake prob) |
| 3 | target_valid_logit | binary logit (sigmoid → target valid) |

### Post-processing (for deployment)

```python
target_valid_prob = sigmoid(pred_target_valid_logit)
brake_prob = sigmoid(pred_brake_logit)

if target_valid_prob < 0.5:
    throttle = 0.0
    brake = 1.0
else:
    brake = 1.0 if brake_prob > 0.5 else 0.0
    if brake > 0.5:
        throttle = 0.0
    else:
        throttle = pred_throttle
```

I.e.: force stop when target invalid; when valid, use brake_prob to decide braking.

### CLI Inference

```bash
# random input test
python infer.py --demo

# with real images and speeds
python infer.py \
  --images frame_001.jpg frame_002.jpg frame_003.jpg frame_004.jpg frame_005.jpg \
  --speeds 4.3 4.3 4.3 4.7 4.7

# custom checkpoint
python infer.py --ckpt checkpoints/20260313_1223_55/best_model.pth --device cuda
```

> **Note**: `infer.py` still uses 3 outputs by default; for 4-output models, apply the above post-processing to map 4-D to steer/throttle/brake.

---

## Model Architecture

- **Backbone**: ResNet18 (without final FC), 512-D features per frame
- **Speed Encoder**: MLP (1→16→16), 16-D per speed
- **Temporal**: Concat image + speed features → GRU (hidden=256), one layer
- **Head**: MLP → 4-D `[steer, throttle, brake_logit, target_valid_logit]`

```
Input: images [B, 5, 3, 224, 224], speeds [B, 5, 1]
     ↓ ResNet18 (per frame) + Speed MLP
     ↓ concat → [B, 5, 528]
     ↓ GRU(hidden=256)
     ↓ last-step hidden
     ↓ MLP
Output: [B, 4]
```

---

## Tool Scripts

| Script | Description |
|--------|-------------|
| `tool/verify_dataset.py` | Verify photo IDs match CSV (1:1, photos as reference) and sort CSV by frame_idx |

**Usage:**

```bash
python tool/verify_dataset.py                    # Check dataset/0312
python tool/verify_dataset.py --path dataset/0312
python tool/verify_dataset.py --fix             # Sort CSV, remove rows without matching photos
```

---

## FAQ

**Q: "target_valid column missing" error?**  
A: main1/main2 controls.csv must have `target_valid`. Add the column manually to controls.csv.

**Q: no_target_stop or no_target?**  
A: Current version uses `no_target`. Rename or update config/dataset paths if you use `no_target_stop`.

**Q: frame_idx mismatches image number?**  
A: frame_idx should match image filename (e.g. 000302.jpg → frame_idx=302). Use `tool/verify_dataset.py --fix` to sort and clean CSV.

**Q: Training stuck on first epoch?**  
A: First epoch is slow (data load + CUDA init). Wait 1–2 min or adjust `num_workers`.

**Q: Inference hangs on Jetson?**  
A: Use `pretrained_backbone=False` to avoid downloading ImageNet weights. Check PyTorch/CUDA versions.

**Q: How to use a different data directory?**  
A: Change `data.data_root` in `config.yaml` or pass `--config` with a new config file.
