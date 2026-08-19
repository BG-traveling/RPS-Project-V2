# -*- coding: utf-8 -*-
"""데이터셋 공용 모듈 — 파일 목록 구성, 도메인별 전처리, Dataset/DataLoader.

두 도메인의 전처리 차이(§ARCHITECTURE.md 3.5)를 샘플 단위 플래그로 처리한다:
  - 합성 이미지(300×200): 왜곡 리사이즈 (기존 학습 규격 유지)
  - 웹캠 프레임(360×640): 중앙 정사각 크롭 후 리사이즈 (비율 보존)
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rps_model import CLASSES, IMG_SIZE

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data_synthetic"
WEBCAM_DIR = ROOT / "Data" / "webcam"
TRAIN_FRACTION = 0.7  # 웹캠 시간순 분할 비율

_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # scale 상한 1.3: 실전에서 손이 ROI를 넘칠 때(확대)의 손가락 잘림 대응
    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.75, 1.3)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # 모션 블러/웹캠 초점 흐림 대응 — 블러 시 paper 쏠림이 진단으로 확인됨
    transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.5, 2.5))], p=0.35),
])
_to_tensor = transforms.ToTensor()  # 0~1 float, (C,H,W)


def list_syn(split_name):
    """data_synthetic/<split>/ 의 (경로, 라벨, crop_square=False) 목록."""
    items = []
    for i, c in enumerate(CLASSES):
        for f in sorted((SYN / split_name / c).glob("*.png")):
            items.append((str(f), i, False))
    return items


def webcam_split():
    """세션(영상)별 시간순 70/30 분할. crop_square=True.

    파일명 규칙 `<클래스>_<세션>__f0000.jpg` 의 세션 프리픽스로 그룹핑한 뒤
    각 세션 안에서 앞 70% → train, 뒤 30% → val 로 나눈다. 연속 프레임이
    train/val 에 섞이는 리크를 막으면서, 촬영 세션이 늘어나도 모든 세션이
    양쪽에 기여하게 한다.
    """
    train, val = [], []
    for i, c in enumerate(CLASSES):
        fs = sorted((WEBCAM_DIR / c).glob("*.jpg")) + sorted((WEBCAM_DIR / c).glob("*.png"))
        sessions = {}
        for f in fs:
            key = f.name.split("__")[0]  # 예: scissors_video, scissors_cap20260819_121500
            sessions.setdefault(key, []).append(f)
        for key in sorted(sessions):
            group = sessions[key]
            cut = int(len(group) * TRAIN_FRACTION)
            train += [(str(f), i, True) for f in group[:cut]]
            val += [(str(f), i, True) for f in group[cut:]]
    return train, val


class RPSDataset(Dataset):
    """items: (path, label, crop_square) 목록. train=True 면 증강 적용."""

    def __init__(self, items, train=False):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, crop_square = self.items[idx]
        img = Image.open(path).convert("RGB")
        if crop_square:
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        if self.train:
            img = _augment(img)
        return _to_tensor(img), label


def make_loader(items, train=False, batch_size=32, num_workers=2):
    return torch.utils.data.DataLoader(
        RPSDataset(items, train=train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
