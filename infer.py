"""
Jetson / 本地推理脚本。
输入: 5 帧图像 + 5 个 speed
输出: [steer, throttle, brake]
注意: 模型输出 4 维（含 brake_logit, target_valid_logit），需用 train.postprocess_prediction 后处理。
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import AutofollowModel
from utils import IMAGENET_MEAN, IMAGENET_STD

_proj = Path(__file__).resolve().parent

# 与 dataset.py 一致的预处理：Resize 224x224 + ImageNet 归一化
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(ckpt_path=None, device=None):
    """加载 checkpoint。默认 checkpoints/best_model.pth。推理时 pretrained_backbone=False 避免下载。"""
    if ckpt_path is None:
        ckpt_path = _proj / "checkpoints" / "best_model.pth"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 推理不需下载预训练 backbone，避免 Jetson 离线卡住
    model = AutofollowModel(pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


def preprocess_images(images):
    """
    将 5 张图像转为模型输入格式。
    images: list of 5 PIL.Image (RGB) 或 numpy (H,W,3) uint8
    返回: Tensor [1, 5, 3, 224, 224]
    """
    tensors = []
    for im in images:
        if not isinstance(im, Image.Image):
            im = Image.fromarray(im)
        im = im.convert("RGB")
        t = _preprocess(im)
        tensors.append(t)
    return torch.stack(tensors).unsqueeze(0)


def infer(model, images, speeds, device=None):
    """
    前向推理。若模型输出 4 维，需在调用方做 postprocess_prediction 转 steer/throttle/brake。
    images: Tensor [1, 5, 3, 224, 224] (已归一化)
    speeds: Tensor [1, 5, 1] 或 list of 5 float
    返回: numpy [steer, throttle, brake]（当前直接取前 3 维，4 维模型需后处理）
    """
    if device is None:
        device = next(model.parameters()).device
    if not isinstance(images, torch.Tensor):
        images = preprocess_images(images)
    images = images.to(device)
    if not isinstance(speeds, torch.Tensor):
        speeds = torch.tensor([speeds], dtype=torch.float32).view(1, 5, 1).to(device)
    else:
        speeds = speeds.to(device)
    with torch.no_grad():
        pred = model(images, speeds)
    return pred.cpu().numpy()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--demo", action="store_true", help="随机输入测试")
    parser.add_argument("--images", type=str, nargs=5, help="5 张图片路径 (按 t-4..t 顺序)")
    parser.add_argument("--speeds", type=float, nargs=5, help="5 个 speed 值")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.ckpt, device)

    if args.images and args.speeds:
        images = [Image.open(p).convert("RGB") for p in args.images]
        out = infer(model, images, args.speeds, device)
    elif args.demo or (not args.images and not args.speeds):
        images = torch.randn(1, 5, 3, 224, 224)
        speeds = torch.rand(1, 5, 1) * 10
        out = infer(model, images, speeds, device)
    else:
        print("需要 --images 和 --speeds，或 --demo")
        exit(1)

    steer, throttle, brake = out
    print(f"steer={steer:.4f}, throttle={throttle:.4f}, brake={brake:.4f}")
