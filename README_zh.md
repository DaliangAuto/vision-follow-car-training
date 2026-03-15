# Autofollow — 端到端模仿学习

[English](README.md) | 中文

自动跟随小车项目：输入最近 5 帧图像及对应车速，输出转向、油门、刹车及目标有效性，实现「有人跟随、无人停车」的安全控制策略。

---

## 目录

- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [目录结构](#目录结构)
- [数据格式](#数据格式)
- [训练](#训练)
- [推理与后处理](#推理与后处理)
- [模型结构](#模型结构)
- [工具脚本](#工具脚本)
- [常见问题](#常见问题)

---

## 项目概述

| 项目 | 说明 |
|------|------|
| **任务** | 端到端模仿学习：从图像 + 速度预测控制动作，并判断画面中是否有有效跟随目标 |
| **输入** | 连续 5 帧图像 + 5 个 speed（时间窗口 [t-4, t-3, t-2, t-1, t]） |
| **输出** | 4 维：`steer`, `throttle`, `brake_logit`, `target_valid_logit` |
| **控制逻辑** | target_valid 为有效时学习正常跟随；无效时（无人）进入安全停车 |
| **图像频率** | ~10 Hz，5 帧约 0.5 秒 |

### 三类数据

| 类型 | 目录 | 说明 |
|------|------|------|
| **main1** | `dataset/0312/main1/` | 主驾驶数据，以有人目标为主，可能含少量无人尾段 |
| **main2** | `dataset/0312/main2/` | 主驾驶数据，同上 |
| **no_target** | `dataset/0312/no_target/` | 无人状态数据，标签统一为 steer=0, throttle=0, brake=1, target_valid=0 |

---

## 环境配置

### 依赖

```bash
pip install -r requirements.txt
```

**requirements.txt 内容：**

```
torch>=2.0
torchvision>=0.15
pandas>=1.0
Pillow>=9.0
numpy>=1.20
tqdm>=4.60
PyYAML>=6.0
```

### 推荐环境

- Python 3.8+
- CUDA 11.x（训练用 GPU，推理 Jetson 可用）

---

## 目录结构

```
autofollow/
├── dataset/
│   ├── 0312/                    # 训练用数据（由 config.yaml 指定）
│   │   ├── main1/
│   │   │   ├── frames/          # 图像 000000.jpg, 000001.jpg, ...
│   │   │   └── controls.csv
│   │   ├── main2/
│   │   │   ├── frames/
│   │   │   └── controls.csv
│   │   └── no_target/
│   │       ├── frames/
│   │       └── controls.csv
│   └── raw/                     # 原始数据，供工具同步用
├── checkpoints/                 # 权重目录
│   └── 20260313_1223_55/        # 每次训练独立目录（YYYYMMDD_HHMM_SS）
│       ├── best_model.pth       # 最佳模型
│       ├── epoch_05.pth         # 第 5 个 epoch 起每轮保存
│       └── ...
├── tool/                        # 辅助工具
│   └── verify_dataset.py        # 校验照片与 CSV 一一对应，按 frame_idx 排序
├── config.yaml                  # 训练配置（数据路径、超参等）
├── dataset.py                   # 数据集构建
├── model.py                     # 模型定义
├── train.py                     # 训练脚本
├── infer.py                     # 推理脚本
├── utils.py
├── requirements.txt
└── README.md
```

---

## 数据格式

### 数据目录规则

- `config.yaml` 中 `data.data_root` 指向 `dataset/0312`，其下必须包含 `main1`、`main2`、`no_target` 三个子目录
- 每个子目录需有 `frames/` 和 `controls.csv`（no_target 若无连续时序信息，可退化为纯图像分块）

### controls.csv 字段（main1 / main2）

| 字段 | 说明 |
|------|------|
| frame_idx | 帧编号，应与照片名称序号一致 |
| image_path | 图像相对路径，如 `frames/000123.jpg` |
| steer | 转向 [-1, 1] |
| throttle | 油门 [0, 1] |
| brake | 刹车 [0, 1] |
| speed | 当前车速 |
| **target_valid** | **必须存在**。1=画面中有有效跟随目标，0=无人或不应跟随 |
| gear, ts, seq, ts_ms, raw | 档位、时间戳等，可辅助调试 |

### 样本构建规则

**main1 / main2：**

- 仅对 `frame_idx` 连续且相差 1 的 5 帧窗口构建样本
- 标签由**最后一帧**的 `target_valid` 决定：
  - `target_valid > 0.5`：steer/throttle 用原始值，brake 按 >0.5 二值化，target_valid=1
  - `target_valid ≤ 0.5`：steer=0, throttle=0, brake=1, target_valid=0

**no_target：**

- 若有 controls.csv 且存在连续 5 帧窗口，优先按连续时序构建
- 否则按图像顺序非重叠分块（每 5 张一组）
- 标签固定：steer=0, throttle=0, brake=1, target_valid=0

---

## 训练

### 配置文件 config.yaml

```yaml
data:
  data_root: dataset/0312   # 需包含 main1、main2、no_target

train:
  batch_size: 32
  epochs: 30
  lr: 0.0001
  weight_decay: 0.0001
  use_type_sampler: true   # 按 main:no_target=8:1 比例采样
```

数据路径、批大小、学习率等均在配置中，命令行参数可覆盖。

### 快速开始

```bash
python train.py
```

### 命令行参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | `config.yaml` | 配置文件路径 |
| `--batch_size` | 来自 config | 批大小 |
| `--epochs` | 来自 config | 训练轮数 |
| `--lr` | 来自 config | 学习率 |
| `--weight_decay` | 来自 config | AdamW 权重衰减 |
| `--use_type_sampler` | 来自 config | 是否按 main:no_target=8:1 采样 |

### 示例

```bash
# 默认训练（使用 config.yaml）
python train.py

# 指定配置与 epoch
python train.py --config config.yaml --epochs 50
```

### Loss 与权重

| 分量 | Loss 类型 | 权重 |
|------|-----------|------|
| steer | SmoothL1 | 2.0 |
| throttle | SmoothL1 | 1.0 |
| brake | BCEWithLogits | 2.0 |
| target_valid | BCEWithLogits | 3.0 |

### 验证指标

- **steer MAE**、**throttle MAE**：平均绝对误差
- **brake acc**、**target_valid acc**：sigmoid 后 >0.5 判正，与标签对比正确率
- 分别统计 main / no_target 上的 brake_acc 和 target_valid_acc

### Checkpoint 保存

- 每次训练使用独立目录：`checkpoints/YYYYMMDD_HHMM_SS/`
- `best_model.pth`：验证 loss 最优时覆盖保存
- 第 5 个 epoch 起，每轮额外保存 `epoch_XX.pth`

---

## 推理与后处理

### 模型输出（4 维）

| 索引 | 含义 | 类型 |
|------|------|------|
| 0 | steer | 连续回归 |
| 1 | throttle | 连续回归 |
| 2 | brake_logit | 二分类 logit（sigmoid 后为 brake 概率） |
| 3 | target_valid_logit | 二分类 logit（sigmoid 后为目标是否有效） |

### 后处理逻辑（供部署使用）

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

即：无人或目标无效时强制停车；有人时由 brake_prob 决定是否刹停。

### 命令行推理

```bash
# 随机输入测试
python infer.py --demo

# 使用真实图像与 speed
python infer.py \
  --images frame_001.jpg frame_002.jpg frame_003.jpg frame_004.jpg frame_005.jpg \
  --speeds 4.3 4.3 4.3 4.7 4.7

# 指定 checkpoint
python infer.py --ckpt checkpoints/20260313_1223_55/best_model.pth --device cuda
```

> **注意**：当前 `infer.py` 仍按 3 输出接口使用；若模型为 4 输出，需在推理侧加入上述后处理逻辑，将 4 维转为 steer/throttle/brake。

---

## 模型结构

- **Backbone**：ResNet18（去掉最后一层），每帧提取 512 维特征
- **Speed Encoder**：MLP (1→16→16)，每帧 speed 编码为 16 维
- **时序**：图像特征 + speed 特征 concat 后送入单层 GRU，hidden=256
- **输出头**：MLP 输出 4 维 `[steer, throttle, brake_logit, target_valid_logit]`

```
输入: images [B, 5, 3, 224, 224], speeds [B, 5, 1]
     ↓ ResNet18 (per frame) + Speed MLP
     ↓ concat → [B, 5, 528]
     ↓ GRU(hidden=256)
     ↓ 取最后时刻 hidden
     ↓ MLP
输出: [B, 4]
```

---

## 工具脚本

| 脚本 | 说明 |
|------|------|
| `tool/verify_dataset.py` | 校验照片 ID 与 CSV 一一对应（以照片为准），并按 frame_idx 排序 CSV |

**用法：**

```bash
python tool/verify_dataset.py                    # 校验 dataset/0312
python tool/verify_dataset.py --path dataset/0312
python tool/verify_dataset.py --fix             # 排序 CSV，删除无对应照片的行
```

---

## 常见问题

**Q: 训练报错「target_valid 列缺失」？**  
A: main1/main2 的 controls.csv 必须包含 `target_valid` 列，请手动添加该列。

**Q: 数据目录是 no_target_stop 还是 no_target？**  
A: 当前版本使用 `no_target`。若实际目录为 `no_target_stop`，需重命名或修改 config 及 dataset 中的路径。

**Q: frame_idx 与照片编号不一致？**  
A: dataset 要求 frame_idx 与照片名称序号一致（如 000302.jpg → frame_idx=302）。可用 `tool/verify_dataset.py --fix` 排序并清理 CSV。

**Q: 训练时卡在第一个 epoch？**  
A: 首轮加载数据与 CUDA 初始化较慢，等待 1–2 分钟；可适当调整 `num_workers`。

**Q: Jetson 上推理卡住？**  
A: 使用 `pretrained_backbone=False` 避免下载 ImageNet 权重；若仍异常，检查 PyTorch 与 CUDA 版本。

**Q: 如何换用其他数据目录？**  
A: 修改 `config.yaml` 中的 `data.data_root`，或通过 `--config` 指定新配置。
