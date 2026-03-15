"""
工具函数：ImageNet 归一化常量、数据路径解析、目录创建。
"""
import os
from pathlib import Path

# ImageNet 预训练模型使用的归一化均值/标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_data_root(override=None):
    """
    解析数据根目录路径。
    override: 配置中的路径；若为相对路径，则相对于项目根目录解析。
    未指定时返回 dataset/raw。
    """
    root = Path(__file__).resolve().parent
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        return p
    return root / "dataset" / "raw"


def ensure_dir(path):
    """创建目录（含父目录），已存在不报错，返回 Path 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
