# -*- coding: utf-8 -*-
"""03. 크로마키 배경 합성 (작업계획서 §5).

splits/*.csv 의 분리를 그대로 유지한 채, 각 세트 안에서만 합성한다 (§5-2).
  - train: 이미지당 서로 다른 배경 2개 → data_synthetic/train/
  - val  : 이미지당 배경 1개        → data_synthetic/val/
  - test : ① 원본 복사 → data_synthetic/test_original/
           ② 배경 1개 합성 → data_synthetic/test_synthetic/

절차(§5-4): HSV 변환 → inRange 초록 마스크 → 모폴로지(열림/닫힘) 노이즈 제거
→ 마스크 반전으로 손 추출 → 배경(300×200)과 합성. 마스크 경계는 가우시안
블러로 소프트 블렌딩해 절단면을 완화한다.
샘플 그리드는 outputs/synthesis_samples.png 로 저장해 육안 검증한다.
"""
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BG_DIR = ROOT / "backgrounds"
OUT_ROOT = ROOT / "data_synthetic"
CLASSES = ["rock", "paper", "scissors"]
SEED = 42

# HSV 초록 임계값 — 샘플 시각화로 튜닝한 값 (H 35~85, 저채도/저명도 제외)
LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])


def imread_u(path):
    """한글 등 비ASCII 경로에서도 동작하는 imread 대체 (Windows cv2 제약 우회)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise RuntimeError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def load_split(name):
    rows = []
    with open(ROOT / "splits" / f"{name}.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((ROOT / row["path"], row["label"]))
    return rows


def green_mask(img_bgr):
    """초록 배경 마스크(255=배경). 모폴로지로 노이즈 제거."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 손 내부 오검출 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 배경 구멍 메움
    return mask


def composite(img_bgr, bg_bgr):
    """손 영역만 남기고 배경 교체. 경계는 소프트 블렌딩."""
    h, w = img_bgr.shape[:2]
    bg = cv2.resize(bg_bgr, (w, h))
    hand_mask = cv2.bitwise_not(green_mask(img_bgr))  # 255=손
    alpha = cv2.GaussianBlur(hand_mask, (5, 5), 0).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    out = img_bgr.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)
    return out.astype(np.uint8)


def synthesize_set(rows, out_dir, n_bg_per_img, bgs, rng):
    for c in CLASSES:
        (out_dir / c).mkdir(parents=True, exist_ok=True)
    n = 0
    for path, label in rows:
        img = imread_u(path)
        if img is None:
            raise RuntimeError(f"읽기 실패: {path}")
        bg_idx = rng.choice(len(bgs), size=n_bg_per_img, replace=False)
        for bi in bg_idx:
            out = composite(img, bgs[bi])
            name = f"{path.stem}_bg{bi}.png"
            imwrite_u(out_dir / label / name, out)
            n += 1
    return n


def main():
    rng = np.random.default_rng(SEED)
    bgs = []
    for f in sorted(BG_DIR.glob("*")):
        bg = imread_u(f)
        if bg is not None:
            bgs.append(bg)
    print(f"배경 {len(bgs)}장 로드")

    train, val, test = load_split("train"), load_split("val"), load_split("test")

    n = synthesize_set(train, OUT_ROOT / "train", 2, bgs, rng)
    print(f"train 합성: {n}장")
    n = synthesize_set(val, OUT_ROOT / "val", 1, bgs, rng)
    print(f"val 합성: {n}장")
    n = synthesize_set(test, OUT_ROOT / "test_synthetic", 1, bgs, rng)
    print(f"test_synthetic 합성: {n}장")

    # test 원본(초록 배경) 복사본 보존 — 배경 증강 효과 정량 비교용 (§5-3)
    for c in CLASSES:
        (OUT_ROOT / "test_original" / c).mkdir(parents=True, exist_ok=True)
    for path, label in test:
        shutil.copy2(path, OUT_ROOT / "test_original" / label / path.name)
    print(f"test_original 복사: {len(test)}장")

    # 샘플 3×5 그리드 시각 검증 (§5-4-6)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(13, 8))
    for r, c in enumerate(CLASSES):
        samples = sorted((OUT_ROOT / "train" / c).glob("*.png"))[:5]
        for col, f in enumerate(samples):
            axes[r, col].imshow(cv2.cvtColor(imread_u(f), cv2.COLOR_BGR2RGB))
            axes[r, col].axis("off")
            if col == 0:
                axes[r, col].set_title(c, loc="left")
    fig.suptitle("Background synthesis samples (visual check)")
    fig.tight_layout()
    out_png = ROOT / "outputs" / "synthesis_samples.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"샘플 그리드 저장 → {out_png}")


if __name__ == "__main__":
    main()

