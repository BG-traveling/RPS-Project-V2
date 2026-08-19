# -*- coding: utf-8 -*-
"""05. 자체 웹캠(실전) 데이터 fine-tuning (PyTorch).

- 합성 train + 웹캠 train×5 (실전 데이터 비중 약 32%)
- 웹캠 시간순 70/30 분할, 뒤 30%(webcam_val)가 조기종료·모델 선택 기준
- 상위 3블록 unfreeze + BatchNorm 전체 동결, lr=5e-5
- 학습 전 webcam_val loss 를 initial_best 로 → 나빠지면 저장 안 함(퇴행 방지)
- 매 epoch wandb 로깅 (project: rps-project)
"""
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rps_data import list_syn, make_loader, webcam_split  # noqa: E402
from rps_model import CLASSES, MODEL_PATH, load_trained  # noqa: E402
from rps_train import evaluate_set, fit, run_epoch  # noqa: E402

import wandb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BACKUP_PATH = MODEL_PATH.with_name("rps_mobilenetv2_prev.pt")
SEED = 42

CONFIG = {
    "stage": "05_webcam_finetune",
    "webcam_repeat": 5,
    "unfreeze_blocks": 3,
    "bn": "frozen (params + running stats)",
    "lr": 5e-5,
    "epochs": 15,
    "patience": 4,
    "webcam_split": "temporal 70/30 (leak-safe)",
}


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = wandb.init(project="rps-project", job_type="finetune",
                     config=CONFIG, mode=os.environ.get("WANDB_MODE", "online"))

    wc_train, wc_val = webcam_split()
    syn_items = list_syn("train")
    train_items = syn_items + wc_train * CONFIG["webcam_repeat"]
    print(f"train 총 {len(train_items)}장 (합성 {len(syn_items)} + 웹캠 {len(wc_train)}×{CONFIG['webcam_repeat']})")

    train_loader = make_loader(train_items, train=True)
    val_loader = make_loader(wc_val)  # 실전 지표로 조기종료 판단

    shutil.copy2(MODEL_PATH, BACKUP_PATH)  # 이전 모델 백업
    model = load_trained(device=device)
    model.unfreeze_top_blocks(CONFIG["unfreeze_blocks"])

    # 학습 전 성능을 저장 기준선으로 — 나빠지면 모델 파일을 덮어쓰지 않음.
    # 기준은 정확도: 실전(분포 밖) 검증에서 loss 기준은 개선을 걸러버린다 (fit 참고)
    init_loss, init_acc = run_epoch(model, val_loader, device)
    print(f"학습 전 webcam_val: loss {init_loss:.4f} / acc {init_acc:.4f}")

    fit(model, train_loader, val_loader, device,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], model_path=MODEL_PATH,
        wandb_run=run, patience=CONFIG["patience"],
        monitor="acc", initial_best=init_acc)

    # 평가 — 디스크의 best 모델 기준
    model = load_trained(device=device)
    report_lines = []
    acc_wc = evaluate_set(model, "webcam_val", val_loader, device, CLASSES, report_lines, run)
    acc_orig = evaluate_set(model, "test_original", make_loader(list_syn("test_original")),
                            device, CLASSES, report_lines, run)
    acc_syn = evaluate_set(model, "test_synthetic", make_loader(list_syn("test_synthetic")),
                           device, CLASSES, report_lines, run)
    report_lines += ["\n===== 요약 =====",
                     f"webcam_val: {acc_wc:.4f} / test_original: {acc_orig:.4f} / test_synthetic: {acc_syn:.4f}"]
    (OUT / "evaluation_webcam_ft.txt").write_text("\n".join(report_lines), encoding="utf-8")
    run.finish()
    print(f"리포트 → {OUT / 'evaluation_webcam_ft.txt'}")
    print(f"이전 모델 백업 → {BACKUP_PATH}")


if __name__ == "__main__":
    main()
