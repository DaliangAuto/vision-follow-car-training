"""
Autofollow 模型：ResNet18 图像编码 + Speed MLP + GRU 时序融合，输出 4 维控制量。
输入: [B, 5, 3, 224, 224] 图像 + [B, 5, 1] 速度
输出: [B, 4] (steer, throttle, brake_logit, target_valid_logit)
"""
import torch
import torch.nn as nn
from torchvision import models

# 模型结构常量
RESNET_FEAT_DIM = 512   # ResNet18 去掉 fc 后的特征维度
SPEED_DIM = 16          # 速度编码后的维度
GRU_HIDDEN = 256        # GRU 隐藏层大小
SEQ_LEN = 5              # 时序长度


class SpeedEncoder(nn.Module):
    """速度编码器：1 维速度 -> 16 维特征，供与时序特征拼接。"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


class AutofollowModel(nn.Module):
    """
    端到端跟随模型：
    - Backbone: ResNet18 提取每帧 512 维特征
    - SpeedEncoder: 速度 -> 16 维
    - GRU: 拼接特征 [B,T,528] 做时序融合
    - Head: 输出 4 维 [steer, throttle, brake_logit, target_valid_logit]
    """

    def __init__(self, pretrained_backbone=True):
        super().__init__()
        # ResNet18 去掉最后一层 fc，保留特征提取部分
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.speed_encoder = SpeedEncoder()
        # 输入: 512(图像) + 16(速度) = 528
        self.gru = nn.GRU(
            input_size=RESNET_FEAT_DIM + SPEED_DIM,
            hidden_size=GRU_HIDDEN,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # [steer, throttle, brake_logit, target_valid_logit]
        )

    def forward(self, images, speeds):
        B, T, C, H, W = images.shape
        # 每帧独立过 backbone，再恢复 [B, T, 512]
        x = images.view(B * T, C, H, W)
        feat = self.backbone(x)
        feat = feat.view(B, T, -1)

        # 每帧速度编码 [B*T, 1] -> [B, T, 16]
        speeds_flat = speeds.view(B * T, 1)
        speed_feat = self.speed_encoder(speeds_flat)
        speed_feat = speed_feat.view(B, T, -1)

        # 拼接后送 GRU，取最后时刻 hidden 过 head
        combined = torch.cat([feat, speed_feat], dim=-1)
        _, h = self.gru(combined)
        h = h.squeeze(0)
        out = self.head(h)
        return out
