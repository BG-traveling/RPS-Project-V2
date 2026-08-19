# -*- coding: utf-8 -*-
"""08. 웹캠 프레임에서 손 영역 크롭 생성 (B안 학습 데이터 정렬).

Data/webcam/<cls>/*.jpg 각 프레임에서 가장 큰 손을 검출해 정사각(+30% 마진)
크롭을 Data/webcam_cropped/<cls>/<같은 파일명> 으로 저장한다.

- 분류기 입력 분포를 앱(검출 크롭)과 일치시키는 것이 목적
- 검출 실패 프레임은 기존 중앙 정사각 크롭으로 폴백 (앱도 동일 정책)
- 이미 크롭이 있는 파일은 건너뜀 (새 촬영분만 증분 처리)
- 끝나면 검출률과 샘플 몽타주(outputs/crop_samples.png) 출력
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hand_detector import HandDetector, expand_square  # noqa: E402

WEBCAM_DIR = ROOT / "Data" / "webcam"
CROPPED_DIR = ROOT / "Data" / "webcam_cropped"
CLASSES = ["rock", "paper", "scissors"]


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise RuntimeError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]


def main():
    det = HandDetector(num_hands=2)
    stats = {}
    samples = []
    for c in CLASSES:
        (CROPPED_DIR / c).mkdir(parents=True, exist_ok=True)
        n_det, n_fb, n_skip = 0, 0, 0
        for f in sorted((WEBCAM_DIR / c).glob("*.jpg")):
            out = CROPPED_DIR / c / f.name
            if out.exists():
                n_skip += 1
                continue
            img = imread_u(f)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hands = det.detect(rgb)
            if hands:
                h, w = img.shape[:2]
                target = max(hands, key=lambda hd: (hd.box[2] - hd.box[0]) * (hd.box[3] - hd.box[1]))
                x1, y1, x2, y2 = expand_square(target.box, w, h)
                crop = img[y1:y2, x1:x2]
                n_det += 1
            else:
                crop = center_square(img)  # 폴백 — 앱과 동일 정책
                n_fb += 1
            imwrite_u(out, crop)
            if len(samples) < 24 and (n_det + n_fb) % 97 == 1:
                samples.append((out, c, bool(hands)))
        stats[c] = (n_det, n_fb, n_skip)
        print(f"{c}: 검출 크롭 {n_det} / 폴백 {n_fb} / 건너뜀(기존) {n_skip}")

    total_new = sum(d + f for d, f, _ in stats.values())
    if total_new:
        rate = sum(d for d, _, _ in stats.values()) / total_new
        print(f"신규 처리 {total_new}장, 손 검출률 {rate:.1%}")

    if samples:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = 8
        rows = (len(samples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 2.2 * rows), squeeze=False)
        for k, (path, c, detected) in enumerate(samples):
            ax = axes[k // cols][k % cols]
            ax.imshow(cv2.cvtColor(imread_u(path), cv2.COLOR_BGR2RGB))
            ax.set_title(f"{c} {'det' if detected else 'fallback'}", fontsize=8,
                         color="green" if detected else "red")
            ax.axis("off")
        for k in range(len(samples), rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle("hand crop samples")
        fig.tight_layout()
        fig.savefig(ROOT / "outputs" / "crop_samples.png", dpi=110)
        print("샘플 → outputs/crop_samples.png")
    det.close()


if __name__ == "__main__":
    main()
