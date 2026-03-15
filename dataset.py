"""
Autofollow Dataset: 仅支持 main1、main2、no_target 三类数据
- main: 基于 controls.csv 连续 5 帧，按 target_valid 决定标签
- no_target: 无人状态，标签统一 steer=0, throttle=0, brake=1, target_valid=0
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGENET_MEAN, IMAGENET_STD, get_data_root

SEQ_LEN = 5  # 每个样本使用的连续帧数

SAMPLE_TYPE_MAIN = "main"       # 有人跟随状态
SAMPLE_TYPE_NO_TARGET = "no_target"  # 无人停车状态


# ========== main1/main2：基于 controls.csv 连续时序 ==========


def build_main_samples(session_path):
    """
    从 main1/main2 构建样本：frame_idx 连续相差 1，5 帧窗口。
    target_valid 必须存在；根据最后一帧 target_valid 决定标签。
    """
    csv_path = session_path / "controls.csv"
    frames_dir = session_path / "frames"
    if not csv_path.exists():
        raise FileNotFoundError(f"[dataset] controls.csv 不存在: {csv_path}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"[dataset] frames 目录不存在: {frames_dir}")

    df = pd.read_csv(csv_path)
    if "target_valid" not in df.columns:
        raise ValueError(f"[dataset] controls.csv 必须包含 target_valid 列: {csv_path}")
    df = df.sort_values("frame_idx").reset_index(drop=True)  # 按帧号排序，保证时序


    # 必需列：frame_idx, image_path, steer, throttle, brake, speed, target_valid
    cols = ["frame_idx", "image_path", "steer", "throttle", "brake", "speed", "target_valid"]
    for c in cols:
        if c not in df.columns:
            return []

    samples = []
    rows = df.to_dict("records")
    # 滑动窗口：仅当 frame_idx 连续相差 1 时构造样本
    for i in range(len(rows) - SEQ_LEN + 1):
        window = rows[i : i + SEQ_LEN]
        frame_indices = [r["frame_idx"] for r in window]
        # 检查帧号是否连续
        if [frame_indices[j + 1] - frame_indices[j] for j in range(SEQ_LEN - 1)] != [1] * (SEQ_LEN - 1):
            continue

        image_paths = []
        valid = True
        for r in window:
            img_path = session_path / r["image_path"]
            if not img_path.exists():
                valid = False
                break
            image_paths.append(img_path)

        if not valid:
            continue

        # 标签由最后一帧的 target_valid 决定
        last = window[-1]
        tv_raw = last["target_valid"]
        tv = 1.0 if (pd.notna(tv_raw) and float(tv_raw) > 0.5) else 0.0

        speeds = [float(r["speed"]) for r in window]
        if any(pd.isna(s) for s in speeds):
            continue

        # target_valid >= 0.5: 使用原始 steer/throttle，brake 二值化
        if tv >= 0.5:
            steer = float(last["steer"])
            throttle = float(last["throttle"])
            brake_raw = float(last["brake"])
            brake = 1.0 if brake_raw > 0.5 else 0.0
            if pd.isna(steer) or pd.isna(throttle) or pd.isna(brake_raw):
                continue
        # target_valid < 0.5: 无人，强制停车
        else:
            steer = 0.0
            throttle = 0.0
            brake = 1.0

        samples.append({
            "image_paths": image_paths,
            "speeds": speeds,
            "target": [steer, throttle, brake, float(tv)],
            "sample_type": SAMPLE_TYPE_MAIN,
        })
    return samples


# ========== no_target：无人状态数据 ==========


def _parse_frame_num(image_path):
    """从图像路径中解析帧号，如 000302.jpg -> 302"""
    s = str(image_path)
    m = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", s, re.I)
    return int(m.group(1)) if m else None


def _collect_image_paths(base_path):
    """从 frames/ 或场景子目录收集图像路径，按帧号排序。"""
    base = Path(base_path)
    frames_dir = base / "frames"
    if not frames_dir.is_dir():
        return []
    imgs = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: _parse_frame_num(p.name) or 0,
    )
    return imgs


def _chunk_non_overlap(image_paths, seq_len=5):
    """非重叠分块，每 seq_len 张为一组；不足 seq_len 时用最后一张重复补齐。"""
    chunks = []
    for i in range(0, len(image_paths), seq_len):
        block = image_paths[i : i + seq_len]
        if len(block) < seq_len:
            block = block + [block[-1]] * (seq_len - len(block))
        chunks.append(block)
    return chunks


def build_no_target_samples(no_target_path):
    """
    构建 no_target 样本：
    - 若有 controls.csv 且 frame_idx 连续可用，按 5 帧窗口构造
    - 否则退化为按图像顺序分块
    标签统一：steer=0, throttle=0, brake=1, target_valid=0
    """
    no_target_path = Path(no_target_path)
    if not no_target_path.is_dir():
        raise FileNotFoundError(f"[dataset] no_target 目录不存在: {no_target_path}")

    target = [0.0, 0.0, 1.0, 0.0]
    samples = []

    # 尝试从 controls.csv 构建连续窗口
    csv_path = no_target_path / "controls.csv"
    frames_dir = no_target_path / "frames"
    if csv_path.exists() and frames_dir.is_dir():
        df = pd.read_csv(csv_path)
        # no_target 标签固定，不依赖 target_valid；仅需 frame_idx/image_path 构建连续窗口
        need_cols = ["frame_idx", "image_path"]
        if all(c in df.columns for c in need_cols):
            df = df.sort_values("frame_idx").reset_index(drop=True)
            rows = df.to_dict("records")
            for i in range(len(rows) - SEQ_LEN + 1):
                window = rows[i : i + SEQ_LEN]
                fidxs = [int(r["frame_idx"]) for r in window]
                if [fidxs[j + 1] - fidxs[j] for j in range(SEQ_LEN - 1)] != [1] * (SEQ_LEN - 1):
                    continue
                image_paths = []
                speeds = []
                ok = True
                for r in window:
                    p = no_target_path / r["image_path"]
                    if not p.exists():
                        ok = False
                        break
                    image_paths.append(p)
                    s = r.get("speed", 0)
                    speeds.append(float(s) if pd.notna(s) else 0.0)
                if not ok:
                    continue
                samples.append({
                    "image_paths": image_paths,
                    "speeds": speeds,
                    "target": target.copy(),
                    "sample_type": SAMPLE_TYPE_NO_TARGET,
                })

    # 若无连续窗口样本，退化为图像分块
    if len(samples) == 0:
        imgs = _collect_image_paths(no_target_path)
        if not imgs:
            raise FileNotFoundError(f"[dataset] 在 {no_target_path} 下未找到图像")
        for chunk in _chunk_non_overlap(imgs, SEQ_LEN):
            samples.append({
                "image_paths": chunk,
                "speeds": [0.0] * SEQ_LEN,
                "target": target.copy(),
                "sample_type": SAMPLE_TYPE_NO_TARGET,
            })

    return samples


# ========== train/val 划分 ==========


def split_train_val(samples_list, train_ratio=0.7):
    """对每组样本按 train_ratio 前后划分，保证时间顺序不泄漏。"""
    train_samples = []
    val_samples = []
    for samples in samples_list:
        n = len(samples)
        if n == 0:
            continue
        split_idx = max(1, min(int(n * train_ratio), n - 1))
        train_samples.extend(samples[:split_idx])
        val_samples.extend(samples[split_idx:])
    return train_samples, val_samples


# ========== Dataset ==========


class AutofollowDataset(Dataset):
    """统一 Dataset：main + no_target。每个样本为 5 帧图像 + 5 个 speed + 4 维标签。"""

    def __init__(self, samples, data_root, train=True):
        self.samples = samples
        self.data_root = Path(data_root)
        self.train = train
        # Resize 到 224x224，ImageNet 归一化
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        # 训练时做轻微亮度/对比度增强
        self.augment = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0) if train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        images = []
        for p in s["image_paths"]:
            img = Image.open(p).convert("RGB")
            if self.augment is not None:
                img = self.augment(img)
            images.append(self.transform(img))
        return {
            "images": torch.stack(images),
            "speeds": torch.tensor(s["speeds"], dtype=torch.float32).unsqueeze(-1),
            "target": torch.tensor(s["target"], dtype=torch.float32),
            "sample_type": s["sample_type"],
        }


# ========== 构建数据集 ==========


def build_datasets(data_root=None, use_type_sampler=False):
    """
    从 data_root 读取 main1, main2, no_target 三类目录。
    返回: (train_ds, val_ds, sampler_weights)
    """
    data_root = Path(get_data_root(data_root))
    if not data_root.exists():
        raise FileNotFoundError(f"[dataset] 数据根目录不存在: {data_root}")

    main1_path = data_root / "main1"
    main2_path = data_root / "main2"
    no_target_path = data_root / "no_target"

    for name, p in [("main1", main1_path), ("main2", main2_path), ("no_target", no_target_path)]:
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"[dataset] 必须存在目录: {p}")

    main1_samples = build_main_samples(main1_path)
    main2_samples = build_main_samples(main2_path)
    no_target_samples = build_no_target_samples(no_target_path)

    train_main1, val_main1 = split_train_val([main1_samples])
    train_main2, val_main2 = split_train_val([main2_samples])
    train_no, val_no = split_train_val([no_target_samples])

    train_samples = train_main1 + train_main2 + train_no
    val_samples = val_main1 + val_main2 + val_no

    train_ds = AutofollowDataset(train_samples, data_root, train=True)
    val_ds = AutofollowDataset(val_samples, data_root, train=False)

    # 按 main:no_target=8:1 计算采样权重
    sampler_weights = None
    if use_type_sampler and len(train_samples) > 0:
        type_target_mass = {SAMPLE_TYPE_MAIN: 8.0, SAMPLE_TYPE_NO_TARGET: 1.0}
        type_count = {SAMPLE_TYPE_MAIN: 0, SAMPLE_TYPE_NO_TARGET: 0}
        for s in train_samples:
            type_count[s["sample_type"]] += 1
        # 每样本权重 = 目标质量 / 该类型样本数
        weights = np.array([
            type_target_mass[s["sample_type"]] / max(type_count[s["sample_type"]], 1)
            for s in train_samples
        ], dtype=np.float64)
        weights /= weights.sum()
        sampler_weights = weights

    return train_ds, val_ds, sampler_weights
