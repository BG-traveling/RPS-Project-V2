# -*- coding: utf-8 -*-
"""실전(PC 웹캠) paper 쏠림 진단.

가설별 실험:
  H1. 손이 없는 입력(배경만)에 대해 모델이 특정 클래스로 확신 → threshold 무력화
  H2. 좌우 반전(카메라 미러링 차이)에 취약한 클래스가 있다
  H3. 손 크기(줌) 변화에 취약하다 — ROI 를 손이 덜 채우면 붕괴
  H4. 밝기/색감(웹캠 화질) 변화에 취약하다
각 실험의 클래스별 recall / 예측 분포를 출력한다.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rps_data import RPSDataset, make_loader, webcam_split  # noqa: E402
from rps_model import CLASSES, IMG_SIZE, load_trained  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_batch(model, imgs):
    x = torch.stack(imgs).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)
    return probs.cpu().numpy()


def load_val_images(items):
    """webcam_val 프레임을 정사각 크롭 + 224 리사이즈 텐서로 로드."""
    ds = RPSDataset(items, train=False)
    return [ds[i] for i in range(len(ds))]


def report(title, probs, labels=None):
    pred = probs.argmax(1)
    conf = probs.max(1)
    dist = {c: int((pred == i).sum()) for i, c in enumerate(CLASSES)}
    line = f"{title:<38} 예측분포 {dist}  평균확신 {conf.mean():.2f}  conf≥0.75 비율 {(conf >= 0.75).mean():.2f}"
    if labels is not None:
        labels = np.asarray(labels)
        rec = {c: round(float((pred[labels == i] == i).mean()), 2)
               for i, c in enumerate(CLASSES)}
        line += f"  recall {rec}"
    print(line)
    return pred, conf


def main():
    model = load_trained(device=DEVICE)
    _, wc_val = webcam_split()
    labels = [l for _, l, _ in wc_val]

    # ---- H1: 손 없는 입력 ------------------------------------------------
    print("== H1. 손이 없는 입력 (이상적: 낮은 확신) ==")
    bg_imgs = []
    for f in sorted((ROOT / "backgrounds").glob("*.jpg"))[:50]:
        img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        bg_imgs.append(TF.to_tensor(img))
    report("Unsplash 배경 50장", predict_batch(model, bg_imgs))

    noise = [torch.rand(3, IMG_SIZE, IMG_SIZE) for _ in range(20)]
    report("랜덤 노이즈 20장", predict_batch(model, noise))
    gray = [torch.full((3, IMG_SIZE, IMG_SIZE), v) for v in np.linspace(0.1, 0.9, 20)]
    report("단색 회색 20장", predict_batch(model, gray))

    # ---- 기준선 -----------------------------------------------------------
    print("\n== 기준선: webcam_val 원본 ==")
    base = [t for t, _ in load_val_images(wc_val)]
    report("원본", predict_batch(model, base), labels)

    # ---- H2: 좌우 반전 -----------------------------------------------------
    print("\n== H2. 좌우 반전 ==")
    report("hflip", predict_batch(model, [TF.hflip(t) for t in base]), labels)

    # ---- H3: 손 크기(줌아웃) ----------------------------------------------
    print("\n== H3. 줌아웃 (손이 ROI를 덜 채울 때) ==")
    for scale in [0.8, 0.6, 0.45]:
        zoomed = [TF.affine(t, angle=0, translate=[0, 0], scale=scale, shear=[0.0],
                            fill=[0.5]) for t in base]
        report(f"scale={scale}", predict_batch(model, zoomed), labels)

    # ---- H4: 밝기/색감 -----------------------------------------------------
    print("\n== H4. 밝기/대비/채도 변화 (웹캠 화질 시뮬레이션) ==")
    for name, fn in [
        ("어둡게 ×0.6", lambda t: TF.adjust_brightness(t, 0.6)),
        ("밝게 ×1.4", lambda t: TF.adjust_brightness(t, 1.4)),
        ("저대비 ×0.6", lambda t: TF.adjust_contrast(t, 0.6)),
        ("저채도 ×0.5", lambda t: TF.adjust_saturation(t, 0.5)),
        ("블러 σ=2", lambda t: TF.gaussian_blur(t, 9, 2.0)),
    ]:
        report(name, predict_batch(model, [fn(t) for t in base]), labels)


if __name__ == "__main__":
    main()
