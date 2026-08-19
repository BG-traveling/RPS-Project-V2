# -*- coding: utf-8 -*-
"""공용 모델 정의 — 학습 스크립트와 웹캠 앱이 모두 이 모듈을 사용한다.

설계 원칙(§ARCHITECTURE.md 원칙 2): 전처리 규격은 모델에 내장한다.
입력 계약은 "0~1 float, (N, 3, 224, 224)" 하나뿐이며, ImageNet 정규화는
forward 안에서 수행되므로 학습/추론 어느 쪽도 정규화 상수를 알 필요가 없다.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision import models

CLASSES = ["rock", "paper", "scissors"]  # label index = 이 순서
IMG_SIZE = 224
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rps_mobilenetv2.pt"


class RPSModel(nn.Module):
    """MobileNetV2 백본 + 분류 head. 입력: 0~1 float (N,3,224,224)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # ImageNet 정규화를 버퍼로 내장 (state_dict에 포함되어 파일만으로 규격 복원)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # GlobalAveragePooling
        return self.head(x)  # logits (loss 는 CrossEntropyLoss 로)

    # --- 동결 제어 -----------------------------------------------------
    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_top_blocks(self, n_blocks: int = 3):
        """상위 n개 블록만 학습 허용. BatchNorm 은 전부 동결(파라미터+통계).

        BN 을 풀면 작은 배치 통계에 하위층 입력 분포가 흔들려 발산했던 전적이
        있어(§ARCHITECTURE.md 사건 5) 파라미터는 물론 running stats 도 고정한다.
        """
        self.freeze_backbone()
        for block in self.features[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(False)

    def train(self, mode: bool = True):
        """BN running stats 오염 방지 — 학습 모드에서도 BN 은 eval 유지."""
        super().train(mode)
        if mode:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


def load_trained(path=MODEL_PATH, device="cpu") -> RPSModel:
    model = RPSModel(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model
