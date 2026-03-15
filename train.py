"""
Autofollow 训练脚本：4 输出（steer, throttle, brake_logit, target_valid_logit）
支持 main / no_target 两类样本，按 8:1 比例采样
配置从 config.yaml 读取，数据集目录在配置文件中指定。
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import (
    SAMPLE_TYPE_MAIN,
    SAMPLE_TYPE_NO_TARGET,
    build_datasets,
)
from model import AutofollowModel
from utils import ensure_dir, get_data_root

# 默认超参数（可被 config.yaml 和命令行覆盖）
BATCH_SIZE = 32          # 批大小
EPOCHS = 30              # 训练轮数
LR = 1e-4                # 学习率
WEIGHT_DECAY = 1e-4      # AdamW 权重衰减
STEER_WEIGHT = 2.0       # steer loss 权重
THROTTLE_WEIGHT = 1.0    # throttle loss 权重
BRAKE_WEIGHT = 2.0       # brake BCE loss 权重
TARGET_VALID_WEIGHT = 3.0  # target_valid BCE loss 权重


def fmt_float(x):
    """浮点数格式化：小值用 6 位小数，大值用科学计数法。"""
    return f"{x:.6f}" if abs(x) < 1e4 else f"{x:.4e}"


# ========== 推理后处理辅助函数（供后续推理复用） ==========


def postprocess_prediction(pred_steer, pred_throttle, pred_brake_logit, pred_target_valid_logit):
    """
    将模型 4 维输出转为可执行控制量。
    pred_steer, pred_throttle: 连续回归值（支持 tensor 或 float）
    pred_brake_logit, pred_target_valid_logit: logit（支持 tensor 或 float），需 sigmoid 转为概率

    返回: (steer, throttle, brake)
    - target_valid < 0.5: throttle=0, brake=1
    - target_valid >= 0.5: brake 由 brake_prob 决定；若 brake>0.5 则 throttle=0
    """
    # 区分 tensor 与普通 float，统一转为 Python float
    if hasattr(pred_target_valid_logit, "item"):
        target_valid = torch.sigmoid(pred_target_valid_logit).item()
        brake_prob = torch.sigmoid(pred_brake_logit).item()
        steer_out = pred_steer.item() if hasattr(pred_steer, "item") else float(pred_steer)
        throttle_raw = pred_throttle.item() if hasattr(pred_throttle, "item") else float(pred_throttle)
    else:
        target_valid = 1.0 / (1.0 + math.exp(-float(pred_target_valid_logit)))
        brake_prob = 1.0 / (1.0 + math.exp(-float(pred_brake_logit)))
        steer_out = float(pred_steer)
        throttle_raw = float(pred_throttle)

    # 目标无效时强制停车
    if target_valid < 0.5:
        return steer_out, 0.0, 1.0
    # 目标有效时，由 brake_prob 决定是否刹车；若刹车则油门归零
    brake = 1.0 if brake_prob > 0.5 else 0.0
    if brake > 0.5:
        throttle_out = 0.0
    else:
        throttle_out = throttle_raw
    return steer_out, throttle_out, brake


def load_config(config_path):
    """从 YAML 配置文件加载配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # ---------- 1. 解析命令行与配置 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径，数据目录等在配置中指定",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument(
        "--use_type_sampler",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="按 main:no_target=8:1 比例采样",
    )
    args = parser.parse_args()

    # 配置文件路径校验
    project_root = Path(__file__).resolve().parent
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    train_cfg = cfg.get("train", {})

    # 数据根目录（必须配置）
    data_root_val = cfg.get("data", {}).get("data_root")
    if not data_root_val:
        print("[ERROR] config.yaml 中缺少 data.data_root 配置")
        sys.exit(1)

    # 命令行参数覆盖配置，未指定时使用 config
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", BATCH_SIZE)
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", EPOCHS)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", LR)
    weight_decay = args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", WEIGHT_DECAY)
    use_type_sampler = args.use_type_sampler if args.use_type_sampler is not None else train_cfg.get("use_type_sampler", True)

    try:
        data_root = get_data_root(data_root_val)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not data_root.exists():
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  Autofollow Training (4 outputs)")
    print("=" * 70)
    print()

    # ---------- 2. 加载数据集 ----------
    print("[1/4] Loading datasets...")
    try:
        # 构建 train/val，并返回采样权重（用于 main:no_target=8:1）
        train_ds, val_ds, sampler_weights = build_datasets(
            data_root, use_type_sampler=use_type_sampler
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if len(train_ds) == 0:
        print("[ERROR] No training samples.")
        sys.exit(1)
    print(f"       Train: {len(train_ds):,} samples  |  Val: {len(val_ds):,} samples")
    if use_type_sampler:
        print("       Sampler: main:no_target = 8:1")
    print()

    # DataLoader：多进程加载 + 显存锁页
    num_workers = 4
    loader_kw = {"num_workers": num_workers, "pin_memory": True}
    if use_type_sampler and sampler_weights is not None:
        # 按 main:no_target=8:1 加权随机采样
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, **loader_kw
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **loader_kw
        )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)

    # ---------- 3. 模型与优化器 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[2/4] Building model...")
    model = AutofollowModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Device: {device}  |  Parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # steer/throttle 用 SmoothL1；brake/target_valid 用 BCE
    criterion_smooth = nn.SmoothL1Loss(reduction="mean")
    criterion_bce = nn.BCEWithLogitsLoss(reduction="mean")

    project_root = Path(__file__).resolve().parent
    best_val_loss = float("inf")
    best_epoch = 0

    # 本次训练独立目录，格式：YYYYMMDD_HHMM_SS
    run_time = datetime.now().strftime("%Y%m%d_%H%M_%S")
    run_dir = ensure_dir(project_root / "checkpoints" / run_time)
    print(f"       Checkpoint dir: {run_dir}")
    print()

    print("[3/4] Training configuration:")
    print(
        f"       Epochs: {epochs}  |  Batch size: {batch_size}  |  LR: {lr}"
    )
    print(f"       Loss weights: steer={STEER_WEIGHT}, throttle={THROTTLE_WEIGHT}, "
          f"brake={BRAKE_WEIGHT}, target_valid={TARGET_VALID_WEIGHT}")
    print()

    print("[4/4] Starting training")
    print()
    print("  指标: L=Loss  mae=MAE  acc=误差<0.1%%  br/tv=brake/target_valid正确率%%")
    print("  每轮两行: 第1行=总览 | 第2行=steer/throttle详情 + [main][no_target]")
    print()
    sep = "─" * 95
    print(sep)
    print(f"  {'Epoch':>5} │ {'Train':>8} │ {'Val':>8} │ {'brake':>6} │ {'tv':>6} │ {'Time':>6} │ Best")
    print(sep)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()

        train_loss = 0.0
        train_steer_loss = 0.0
        train_throttle_loss = 0.0
        train_steer_mae = 0.0
        train_throttle_mae = 0.0
        train_steer_acc = 0   # |pred-target|<0.1 的样本数
        train_throttle_acc = 0
        train_n = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:2d}/{epochs}",
            ncols=100,
            leave=False,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for batch in pbar:
            images = batch["images"].to(device)
            speeds = batch["speeds"].to(device)
            target = batch["target"].to(device)  # [B, 4]: steer, throttle, brake, target_valid

            optimizer.zero_grad()
            pred = model(images, speeds)  # [B, 4]: steer, throttle, brake_logit, target_valid_logit

            # 分维度计算 Loss
            loss_steer = criterion_smooth(pred[:, 0], target[:, 0])
            loss_throttle = criterion_smooth(pred[:, 1], target[:, 1])
            loss_brake = criterion_bce(pred[:, 2], target[:, 2])
            loss_target_valid = criterion_bce(pred[:, 3], target[:, 3])

            # 加权求和作为总 Loss
            loss = (
                STEER_WEIGHT * loss_steer
                + THROTTLE_WEIGHT * loss_throttle
                + BRAKE_WEIGHT * loss_brake
                + TARGET_VALID_WEIGHT * loss_target_valid
            )
            loss.backward()
            optimizer.step()

            # 统计本 batch 的各类指标
            n = images.size(0)
            train_loss += loss.item() * n
            train_steer_loss += loss_steer.item() * n
            train_throttle_loss += loss_throttle.item() * n
            train_steer_mae += (pred[:, 0] - target[:, 0]).abs().sum().item()
            train_throttle_mae += (pred[:, 1] - target[:, 1]).abs().sum().item()
            # 容差 acc：|pred-target|<0.1 的样本数
            train_steer_acc += ((pred[:, 0] - target[:, 0]).abs() < 0.1).sum().item()
            train_throttle_acc += ((pred[:, 1] - target[:, 1]).abs() < 0.1).sum().item()
            train_n += n
            pbar.set_postfix(
                loss=fmt_float(loss.item()),
                s_L=fmt_float(loss_steer.item()),
                t_L=fmt_float(loss_throttle.item()),
            )

        train_loss /= max(train_n, 1)
        train_steer_loss /= max(train_n, 1)
        train_throttle_loss /= max(train_n, 1)
        train_steer_mae /= max(train_n, 1)
        train_throttle_mae /= max(train_n, 1)
        train_steer_acc_pct = 100.0 * train_steer_acc / max(train_n, 1)
        train_throttle_acc_pct = 100.0 * train_throttle_acc / max(train_n, 1)

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        val_steer_loss = 0.0
        val_throttle_loss = 0.0
        val_mae_steer = 0.0
        val_mae_throttle = 0.0
        val_steer_acc = 0
        val_throttle_acc = 0
        val_brake_correct = 0
        val_tv_correct = 0
        val_n = 0
        # 按类型统计 brake / target_valid acc
        type_stats = {
            SAMPLE_TYPE_MAIN: {"n": 0, "brake_ok": 0, "tv_ok": 0},
            SAMPLE_TYPE_NO_TARGET: {"n": 0, "brake_ok": 0, "tv_ok": 0},
        }

        with torch.no_grad():
            pbar_val = tqdm(
                val_loader,
                desc=f"  Val [{epoch}/{epochs}]",
                ncols=90,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            )
            for batch in pbar_val:
                images = batch["images"].to(device)
                speeds = batch["speeds"].to(device)
                target = batch["target"].to(device)
                sample_types = batch["sample_type"]
                pred = model(images, speeds)

                loss_steer = criterion_smooth(pred[:, 0], target[:, 0])
                loss_throttle = criterion_smooth(pred[:, 1], target[:, 1])
                loss_brake = criterion_bce(pred[:, 2], target[:, 2])
                loss_target_valid = criterion_bce(pred[:, 3], target[:, 3])
                loss = (
                    STEER_WEIGHT * loss_steer
                    + THROTTLE_WEIGHT * loss_throttle
                    + BRAKE_WEIGHT * loss_brake
                    + TARGET_VALID_WEIGHT * loss_target_valid
                )
                n = images.size(0)
                val_loss += loss.item() * n
                val_steer_loss += loss_steer.item() * n
                val_throttle_loss += loss_throttle.item() * n
                val_mae_steer += (pred[:, 0] - target[:, 0]).abs().sum().item()
                val_mae_throttle += (pred[:, 1] - target[:, 1]).abs().sum().item()
                val_steer_acc += ((pred[:, 0] - target[:, 0]).abs() < 0.1).sum().item()
                val_throttle_acc += ((pred[:, 1] - target[:, 1]).abs() < 0.1).sum().item()

                # brake / target_valid：logit 转概率后 >0.5 判为正类
                brake_pred = (torch.sigmoid(pred[:, 2]) > 0.5).float()
                brake_gt = (target[:, 2] > 0.5).float()
                tv_pred = (torch.sigmoid(pred[:, 3]) > 0.5).float()
                tv_gt = (target[:, 3] > 0.5).float()
                val_brake_correct += (brake_pred == brake_gt).sum().item()
                val_tv_correct += (tv_pred == tv_gt).sum().item()

                # 按 main / no_target 分别统计 brake 和 target_valid 正确率
                for i in range(n):
                    st = sample_types[i] if isinstance(sample_types[i], str) else str(sample_types[i])
                    if st not in type_stats:
                        continue
                    type_stats[st]["n"] += 1
                    if brake_pred[i].item() == brake_gt[i].item():
                        type_stats[st]["brake_ok"] += 1
                    if tv_pred[i].item() == tv_gt[i].item():
                        type_stats[st]["tv_ok"] += 1

                val_n += n

        val_loss /= max(val_n, 1)
        val_steer_loss /= max(val_n, 1)
        val_throttle_loss /= max(val_n, 1)
        val_mae_steer /= max(val_n, 1)
        val_mae_throttle /= max(val_n, 1)
        val_steer_acc_pct = 100.0 * val_steer_acc / max(val_n, 1)
        val_throttle_acc_pct = 100.0 * val_throttle_acc / max(val_n, 1)
        val_brake_acc = 100.0 * val_brake_correct / max(val_n, 1)
        val_tv_acc = 100.0 * val_tv_correct / max(val_n, 1)

        elapsed = time.perf_counter() - t0
        is_best = val_loss < best_val_loss
        # 保存最佳模型
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                run_dir / "best_model.pth",
            )
        # 第 5 个 epoch 起，每轮都保存 checkpoint
        if epoch >= 5:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                run_dir / f"epoch_{epoch:02d}.pth",
            )
        best_mark = " *" if is_best else "  "

        # 第1行：总览 Loss + brake/tv + Time
        print(
            f"  {epoch:>2}/{epochs:<2} │ {train_loss:>8.4f} │ {val_loss:>8.4f} │ "
            f"{val_brake_acc:>5.1f}% │ {val_tv_acc:>5.1f}% │ {elapsed:>5.0f}s │{best_mark}"
        )

        # 第2行：Train + Val 的 steer/throttle 完整详情
        ts = f"L={train_steer_loss:.4f} mae={train_steer_mae:.3f} acc={train_steer_acc_pct:.0f}%"
        tt = f"L={train_throttle_loss:.4f} mae={train_throttle_mae:.3f} acc={train_throttle_acc_pct:.0f}%"
        vs = f"L={val_steer_loss:.4f} mae={val_mae_steer:.3f} acc={val_steer_acc_pct:.0f}%"
        vt = f"L={val_throttle_loss:.4f} mae={val_mae_throttle:.3f} acc={val_throttle_acc_pct:.0f}%"
        print(f"       Train steer({ts}) thr({tt})")
        print(f"       Val   steer({vs}) thr({vt})")

        # 第3行：[main] [no_target]
        type_parts = []
        for st, stats in type_stats.items():
            n = stats["n"]
            if n == 0:
                continue
            ba = 100.0 * stats["brake_ok"] / n
            ta = 100.0 * stats["tv_ok"] / n
            type_parts.append(f"[{st}] n={n} br={ba:.0f}% tv={ta:.0f}%")
        if type_parts:
            print(f"       {'  '.join(type_parts)}")
        print()  # 每 epoch 后空一行

    print(sep)
    print()
    print("  " + "=" * 60)
    print("  Training completed.")
    print(f"  Best val_loss: {fmt_float(best_val_loss)}  (epoch {best_epoch})")
    print(f"  Checkpoint dir: {run_dir}")
    print(f"  Best model:    {run_dir / 'best_model.pth'}")
    if epochs >= 5:
        print(f"  Epoch ckpts:   epoch_05.pth ~ epoch_{epochs:02d}.pth (from epoch 5)")
    print("  " + "=" * 60)
    print()


if __name__ == "__main__":
    main()
