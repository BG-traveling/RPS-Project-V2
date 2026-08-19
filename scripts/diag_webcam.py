# -*- coding: utf-8 -*-
"""웹캠 fine-tuning 진단 (PyTorch).

1) wc_train vs wc_val 정확도 비교 → 암기(과적합) vs 학습 실패 구분
2) 클래스별 시간순 정답률 타임라인 → 영상 구간별 라벨 노이즈/포즈 변화 확인
3) val 프레임 몽타주(예측 라벨 표기) → 육안 확인
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rps_data import TRAIN_FRACTION, make_loader, webcam_split  # noqa: E402
from rps_model import CLASSES, load_trained  # noqa: E402
from rps_train import predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def acc_by_class(pred, true):
    out = {}
    for i, c in enumerate(CLASSES):
        m = true == i
        out[c] = float((pred[m] == i).mean()) if m.any() else float("nan")
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(device=device)
    wc_train, wc_val = webcam_split()

    p_tr, y_tr = predict(model, make_loader(wc_train), device)
    p_va, y_va = predict(model, make_loader(wc_val), device)
    print(f"wc_train acc: {(p_tr == y_tr).mean():.4f}  {acc_by_class(p_tr, y_tr)}")
    print(f"wc_val   acc: {(p_va == y_va).mean():.4f}  {acc_by_class(p_va, y_va)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # 시간순 타임라인
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    all_items = {}
    for i, c in enumerate(CLASSES):
        items = [t for t in wc_train + wc_val if t[1] == i]
        items.sort(key=lambda t: t[0])
        all_items[c] = items
        preds, _ = predict(model, make_loader(items), device)
        correct = (preds == i).astype(int)
        axes[i].scatter(range(len(items)), correct, s=8,
                        c=["tab:green" if v else "tab:red" for v in correct])
        axes[i].axvline(int(len(items) * TRAIN_FRACTION), color="k", ls="--", lw=1)
        axes[i].set_ylabel(c)
        axes[i].set_yticks([0, 1], ["wrong", "ok"])
    axes[-1].set_xlabel("frame index (점선 오른쪽 = val)")
    fig.suptitle("Per-frame correctness over video timeline")
    fig.tight_layout()
    fig.savefig(OUT / "diag_timeline.png", dpi=120)

    # val 몽타주
    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    for r, c in enumerate(CLASSES):
        items = all_items[c]
        val_items = items[int(len(items) * TRAIN_FRACTION):]
        picks = val_items[::max(1, len(val_items) // 8)][:8]
        preds, _ = predict(model, make_loader(picks), device)
        for col, ((path, _, _), pr) in enumerate(zip(picks, preds)):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            axes[r, col].imshow(img)
            ok = CLASSES[pr] == c
            axes[r, col].set_title(f"pred: {CLASSES[pr]}", fontsize=9,
                                   color="green" if ok else "red")
            axes[r, col].axis("off")
    fig.suptitle("val frames (center square crop = model input view)")
    fig.tight_layout()
    fig.savefig(OUT / "diag_val_montage.png", dpi=120)
    print(f"저장: {OUT / 'diag_timeline.png'}, {OUT / 'diag_val_montage.png'}")


if __name__ == "__main__":
    main()
