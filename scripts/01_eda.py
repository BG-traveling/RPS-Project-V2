# -*- coding: utf-8 -*-
"""01. EDA — 클래스 분포, 이미지 규격/손상 점검, 밝기 분포, RGB 채널 분석.

작업계획서 §4-1 에 해당. 결과 그래프는 outputs/eda/ 에 저장.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "outputs" / "eda"
CLASSES = ["rock", "paper", "scissors"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 클래스별 개수
    counts = {}
    files = {}
    for c in CLASSES:
        fs = sorted((DATA_DIR / c).glob("*.png"))
        files[c] = fs
        counts[c] = len(fs)
    print("클래스별 이미지 수:", counts, "| 총", sum(counts.values()))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.keys(), counts.values(), color=["#888", "#6b9", "#69b"])
    ax.set_title("Class distribution")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 5, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)

    # 2) 규격/손상 점검
    sizes, modes, corrupt = set(), set(), []
    for c in CLASSES:
        for f in files[c]:
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    sizes.add(im.size)
                    modes.add(im.mode)
            except Exception:
                corrupt.append(f)
    print("이미지 크기 종류:", sizes, "| 모드:", modes, "| 손상 파일:", len(corrupt))

    # 3) 샘플 시각화 (클래스별 5장)
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    rng = np.random.default_rng(42)
    for r, c in enumerate(CLASSES):
        picks = rng.choice(len(files[c]), 5, replace=False)
        for col, idx in enumerate(picks):
            axes[r, col].imshow(Image.open(files[c][idx]))
            axes[r, col].axis("off")
            if col == 0:
                axes[r, col].set_title(c, loc="left")
    fig.suptitle("Class samples")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "class_samples.png", dpi=120)
    plt.close(fig)

    # 4) 밝기 분포 + RGB 채널 평균 (클래스당 100장 샘플)
    bright = {c: [] for c in CLASSES}
    chans = {c: np.zeros(3) for c in CLASSES}
    n_sample = 100
    for c in CLASSES:
        picks = rng.choice(len(files[c]), n_sample, replace=False)
        for idx in picks:
            arr = np.asarray(Image.open(files[c][idx]).convert("RGB"), dtype=np.float32)
            bright[c].append(arr.mean())
            chans[c] += arr.reshape(-1, 3).mean(axis=0)
        chans[c] /= n_sample

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for c in CLASSES:
        ax1.hist(bright[c], bins=30, alpha=0.5, label=c)
    ax1.set_title("Mean brightness distribution")
    ax1.legend()

    x = np.arange(3)
    w = 0.25
    for i, c in enumerate(CLASSES):
        ax2.bar(x + i * w, chans[c], w, label=c)
    ax2.set_xticks(x + w, ["R", "G", "B"])
    ax2.set_title("Mean RGB channel value (background bias check)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "brightness_rgb.png", dpi=120)
    plt.close(fig)

    for c in CLASSES:
        r, g, b = chans[c]
        print(f"[{c}] R={r:.1f} G={g:.1f} B={b:.1f}  (G 우세 → 초록 배경 편향 리스크)")

    if corrupt:
        print("손상 파일 목록:", corrupt, file=sys.stderr)
        sys.exit(1)
    print("EDA 완료 → outputs/eda/")


if __name__ == "__main__":
    main()
