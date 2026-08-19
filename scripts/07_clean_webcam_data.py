# -*- coding: utf-8 -*-
"""07. 웹캠 데이터 라벨 노이즈 정리.

연속 촬영(R) 중 손이 프레임을 떠난 뒤에도 저장된 "배경만 있는" 프레임을
피부색 픽셀 비율로 검출해 Data/webcam_excluded/ 로 격리한다 (삭제하지 않음).

- 판정: 중앙 정사각(모델 입력 영역)의 YCrCb 피부 마스크 비율 < SKIN_MIN
- 격리된 프레임은 outputs/excluded_montage.png 로 시각 확인 가능
- 복구: 격리 폴더에서 원래 클래스 폴더로 파일을 되돌리면 됨
"""
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEBCAM_DIR = ROOT / "Data" / "webcam"
EXCLUDED_DIR = ROOT / "Data" / "webcam_excluded"
CLASSES = ["rock", "paper", "scissors"]
SKIN_MIN = 0.03  # 중앙 정사각에서 피부 픽셀 3% 미만이면 "손 없음"으로 판정


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def skin_ratio(img_bgr):
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    crop = img_bgr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    # 표준적인 YCrCb 피부색 범위 (Cr 133~173, Cb 77~127)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return mask.mean() / 255.0


def main():
    moved = []
    for c in CLASSES:
        (EXCLUDED_DIR / c).mkdir(parents=True, exist_ok=True)
        for f in sorted((WEBCAM_DIR / c).glob("*.jpg")):
            img = imread_u(f)
            if img is None:
                continue
            r = skin_ratio(img)
            if r < SKIN_MIN:
                shutil.move(str(f), str(EXCLUDED_DIR / c / f.name))
                moved.append((EXCLUDED_DIR / c / f.name, c, r))
        n_left = len(list((WEBCAM_DIR / c).glob("*.jpg")))
        n_out = len([m for m in moved if m[1] == c])
        print(f"{c}: 격리 {n_out}장, 잔여 {n_left}장")

    if moved:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(moved)
        cols = 8
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2.2 * rows), squeeze=False)
        for k, (path, c, r) in enumerate(moved):
            ax = axes[k // cols][k % cols]
            ax.imshow(cv2.cvtColor(imread_u(path), cv2.COLOR_BGR2RGB))
            ax.set_title(f"{c} skin={r:.3f}", fontsize=7)
            ax.axis("off")
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(f"excluded frames ({n}) — 잘못 격리된 게 있으면 폴더에서 되돌리세요")
        fig.tight_layout()
        fig.savefig(ROOT / "outputs" / "excluded_montage.png", dpi=110)
        print(f"격리 목록 시각화 → outputs/excluded_montage.png")


if __name__ == "__main__":
    main()
