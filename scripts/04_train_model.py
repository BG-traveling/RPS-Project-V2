# -*- coding: utf-8 -*-
"""04. MobileNetV2 전이학습 (PyTorch) — 합성 배경 데이터로 1차 학습.

- 백본 동결, head 만 lr=1e-3 로 10 epoch (조기종료 patience 4)
- 매 epoch 지표를 wandb 로깅 (project: rps-project)
  - 온라인 로깅: 먼저 `wandb login` 실행. 오프라인: WANDB_MODE=offline
- 평가: test_original vs test_synthetic (배경 증강 효과 정량 비교)
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rps_data import list_syn, make_loader  # noqa: E402
from rps_model import CLASSES, IMG_SIZE, MODEL_PATH, RPSModel  # noqa: E402
from rps_train import evaluate_set, fit  # noqa: E402

import wandb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 42

CONFIG = {
    "stage": "04_base_training",
    "backbone": "mobilenet_v2 (ImageNet, frozen)",
    "img_size": IMG_SIZE,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 20,
    "patience": 5,
    "augment": "hflip, affine(rot90, trans0.1, scale0.15), jitter(b0.2, c0.2)",
}


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    run = wandb.init(project="rps-project", job_type="train",
                     config=CONFIG, mode=os.environ.get("WANDB_MODE", "online"))

    train_loader = make_loader(list_syn("train"), train=True)
    val_loader = make_loader(list_syn("val"))

    model = RPSModel(pretrained=True).to(device)
    model.freeze_backbone()

    fit(model, train_loader, val_loader, device,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], model_path=MODEL_PATH,
        wandb_run=run, patience=CONFIG["patience"])

    # 평가 — 디스크에 저장된 best 모델 기준
    from rps_model import load_trained
    model = load_trained(device=device)
    report_lines = []
    acc_orig = evaluate_set(model, "test_original", make_loader(list_syn("test_original")),
                            device, CLASSES, report_lines, run)
    acc_syn = evaluate_set(model, "test_synthetic", make_loader(list_syn("test_synthetic")),
                           device, CLASSES, report_lines, run)
    report_lines += ["\n===== 배경 증강 효과 비교 =====",
                     f"test_original(초록 배경): {acc_orig:.4f}",
                     f"test_synthetic(실사 배경): {acc_syn:.4f}"]
    (OUT / "evaluation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    json.dump({"classes": CLASSES, "img_size": IMG_SIZE,
               "test_original_acc": acc_orig, "test_synthetic_acc": acc_syn},
              open(OUT / "metrics.json", "w", encoding="utf-8"), indent=2)
    run.finish()
    print(f"모델 저장 → {MODEL_PATH}")


if __name__ == "__main__":
    main()
