# -*- coding: utf-8 -*-
"""학습 루프 공용 모듈 — 04/05 스크립트가 공유하는 train/evaluate 유틸.

- 매 epoch 지표를 wandb 에 로깅 (train/val loss·accuracy, lr)
- EarlyStopping(최적 가중치 복원) + best-only 체크포인트
- initial_best 로 "이전 학습보다 나쁘면 저장 안 함" 퇴행 방지 지원
"""
import copy

import numpy as np
import torch
from torch import nn


def run_epoch(model, loader, device, optimizer=None):
    """optimizer 가 있으면 학습, 없으면 평가. (loss, acc) 반환."""
    training = optimizer is not None
    model.train() if training else model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, device, *, epochs, lr, model_path,
        wandb_run=None, patience=4, monitor="loss", initial_best=None):
    """조기종료 + best-only 저장 학습 루프. best 지표값을 반환.

    monitor:
      - "loss": val_loss 최소화 — 검증이 학습과 같은 분포일 때 (04단계)
      - "acc" : val_accuracy 최대화 — 분포 밖(실전) 검증에서는 정확도가 올라도
        확신에 찬 오답 때문에 loss 가 높게 유지될 수 있어 loss 기준 저장이
        개선을 전부 걸러버리는 문제가 있었음 (05단계는 반드시 acc 기준)
    initial_best: 학습 전 모델의 지표 — 이보다 나빠지면 저장하지 않는 퇴행 방지.
    """
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    maximize = monitor == "acc"
    best = initial_best if initial_best is not None else (-float("inf") if maximize else float("inf"))
    best_state = None
    wait = 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device)
        print(f"epoch {epoch:2d}/{epochs}  train {tr_loss:.4f}/{tr_acc:.4f}"
              f"  val {va_loss:.4f}/{va_acc:.4f}")
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": tr_loss, "train/accuracy": tr_acc,
                "val/loss": va_loss, "val/accuracy": va_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })
        metric = va_acc if maximize else va_loss
        improved = metric > best if maximize else metric < best
        if improved:
            best = metric
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_path)  # best-only 체크포인트
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"조기종료 (patience={patience})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)  # 최적 가중치 복원
    return best


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def evaluate_set(model, name, loader, device, classes, report_lines, wandb_run=None):
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred, y_true = predict(model, loader, device)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=classes, digits=4)
    report_lines += [f"\n===== {name} =====", f"accuracy: {acc:.4f}",
                     f"confusion matrix (rows=true {classes}):\n{cm}", rep]
    print(f"{name} accuracy: {acc:.4f}")
    if wandb_run is not None:
        wandb_run.summary[f"{name}/accuracy"] = acc
        import wandb as _wandb
        wandb_run.log({f"{name}/confusion_matrix": _wandb.plot.confusion_matrix(
            y_true=y_true.tolist(), preds=y_pred.tolist(), class_names=classes
        )})
    return acc
