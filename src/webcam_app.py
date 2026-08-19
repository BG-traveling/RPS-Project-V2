# -*- coding: utf-8 -*-
"""실시간 웹캠 가위바위보 인식 앱 (PyTorch + MediaPipe 손 검출, B안 2단계 구조).

실행:  python src/webcam_app.py  (종료: q)

파이프라인:
  프레임 → 손 검출(여러 개 가능) → 대상 손 선택(최초: 최대 크기, 이후: IoU 추적)
  → bbox 정사각+30% 마진 크롭 → BGR→RGB → 224 리사이즈 → /255.0
  → CNN 분류(logits→softmax) → 최근 N=7 프레임 확률 평균 스무딩
  → 최대 확률 < 0.75 → "WAITING" / 손 미검출 → "NO HAND"
  → 관절 각도 기반 게이트가 5프레임 연속 "학습된 3제스처 중 하나가 아님"으로
    판단하면 → "UNKNOWN GESTURE" (CNN 강제 분류 대신 재시도 유도)
  → 예측 안정화 후 쿨다운(2초) → 제스처-액션 매핑

화면: 대상 손 = 초록 박스 + 라벨, 그 외 손 = 회색 박스, 클래스별 확률 바.

실시간 보정 학습 (키 입력):
  1/2/3 : 현재 화면이 rock/paper/scissors 로 잘못 인식됐을 때, 정답을 눌러
          그 프레임을 학습 데이터로 저장 (Data/webcam/<class>/, 즉시
          Data/webcam_cropped/<class>/ 에도 크롭 반영)
  T     : 지금까지 모은 보정 데이터로 즉석 미세조정 (카메라 일시정지,
          완료 후 자동으로 새 가중치를 이어서 사용 — 재시작 불필요)

  주의: 이 즉석 학습은 웹캠 데이터만으로 짧게(수 epoch) 이어서 학습하는
  "빠른 보정"이며, 05_finetune_webcam.py 의 정식 재학습(합성 데이터 포함,
  더 많은 epoch)을 대체하지 않는다. 정식 학습과 동일하게 정확도가
  나빠지면 저장하지 않는 퇴행 방지가 적용되고, 직전 가중치는 자동 백업된다.
"""
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_detector import (  # noqa: E402
    HandDetector, TargetTracker, classify_geometric, expand_square,
)
from rps_model import CLASSES, IMG_SIZE, MODEL_PATH, load_trained  # noqa: E402

SMOOTH_N = 7               # 스무딩 창 크기
CONF_THRESHOLD = 0.75      # 이 미만이면 대기 상태
COOLDOWN_SEC = 2.0         # 액션 트리거 후 재트리거 금지 시간
GATE_FAIL_PATIENCE = 5     # 관절 게이트가 이 프레임 수만큼 연속 실패해야 "미학습 모양"
CORRECTION_MIN = 6         # 즉석 미세조정에 필요한 최소 보정 샘플 수
LIVE_SESSION = time.strftime("live_cap%Y%m%d_%H%M%S")

KEY_TO_CLASS = {ord("1"): "rock", ord("2"): "paper", ord("3"): "scissors",
                ord("r"): "rock", ord("p"): "paper", ord("s"): "scissors"}


def on_action(gesture: str):
    """제스처-액션 매핑. 필요에 맞게 교체 (키 입력 전송, 게임 로직 등)."""
    actions = {
        "rock": lambda: print(">>> ACTION: ROCK — 공격!"),
        "paper": lambda: print(">>> ACTION: PAPER — 방어!"),
        "scissors": lambda: print(">>> ACTION: SCISSORS — 스킬!"),
    }
    actions[gesture]()


def imwrite_u(path, img):
    """한글 경로에서도 동작하는 imwrite (cv2 Windows 제약 우회)."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


def save_correction(frame, crop_box, cls, idx_counter):
    """현재 프레임을 정답 라벨로 저장 — 원본은 webcam/, 크롭은 webcam_cropped/."""
    from rps_data import CROPPED_DIR, WEBCAM_DIR

    idx = idx_counter[cls]
    idx_counter[cls] += 1
    name = f"{cls}_{LIVE_SESSION}__f{idx:04d}.jpg"
    (WEBCAM_DIR / cls).mkdir(parents=True, exist_ok=True)
    (CROPPED_DIR / cls).mkdir(parents=True, exist_ok=True)
    imwrite_u(WEBCAM_DIR / cls / name, frame)
    x1, y1, x2, y2 = crop_box
    imwrite_u(CROPPED_DIR / cls / name, frame[y1:y2, x1:x2])


def quick_finetune(model, device, epochs=3):
    """방금 모은 보정 데이터를 포함해 웹캠 데이터로 짧게 이어서 학습.

    05_finetune_webcam.py 와 같은 안전장치(정확도 기준 체크포인트, 이전
    가중치 백업)를 쓰되, 합성 데이터는 빼고 웹캠 데이터만 돌려 수 분 내에
    끝나도록 한다. 반환: (직전 webcam 정확도, 갱신 후 webcam 정확도).
    """
    from rps_data import make_loader, webcam_split
    from rps_train import fit, run_epoch

    backup = MODEL_PATH.with_name(MODEL_PATH.stem + "_prev" + MODEL_PATH.suffix)
    shutil.copy2(MODEL_PATH, backup)

    # num_workers=0: 이미 열려 있는 카메라·CUDA·cv2 창이 있는 인터랙티브
    # 프로세스에서 멀티프로세스 워커를 새로 스폰하면 Windows 에서 자칫
    # 불안정해질 수 있어, 짧은 즉석 학습은 단일 프로세스로 안전하게 돌린다.
    wc_train, wc_val = webcam_split()
    train_loader = make_loader(wc_train, train=True, num_workers=0)
    val_loader = make_loader(wc_val, num_workers=0)

    model.unfreeze_top_blocks(3)
    init_loss, init_acc = run_epoch(model, val_loader, device)
    best_acc = fit(model, train_loader, val_loader, device, epochs=epochs, lr=5e-5,
                    model_path=MODEL_PATH, patience=2, monitor="acc", initial_best=init_acc)
    return init_acc, best_acc


def draw_center_message(frame, lines, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 40), (w, h // 2 + 40 * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (24, h // 2 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained(device=device)
    with torch.no_grad():  # 워밍업
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))
    detector = HandDetector(num_hands=2)
    tracker = TargetTracker()
    print(f"모델 로드 완료: {MODEL_PATH} (device={device})")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    probs_hist = deque(maxlen=SMOOTH_N)
    last_trigger = 0.0
    gate_fail_streak = 0
    correction_counts = {c: 0 for c in CLASSES}
    correction_idx = {c: 0 for c in CLASSES}
    last_crop_box = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        now = time.time()

        t0 = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = detector.detect(rgb)
        target, switched = tracker.select(hands)
        if switched:
            probs_hist.clear()  # 대상 손이 바뀌면 이전 손의 확률 섞임 방지
            gate_fail_streak = 0

        label, color = "NO HAND", (120, 120, 120)
        avg, idx, unknown_gesture = None, 0, False
        if target is not None:
            x1, y1, x2, y2 = expand_square(target.box, w, h)
            last_crop_box = (x1, y1, x2, y2)
            inp = cv2.resize(rgb[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
            inp = inp.astype(np.float32) / 255.0
            with torch.no_grad():
                x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            probs_hist.append(probs)

            geo = classify_geometric(target.landmarks)
            gate_fail_streak = gate_fail_streak + 1 if geo is None else 0
            unknown_gesture = gate_fail_streak >= GATE_FAIL_PATIENCE

            avg = np.mean(probs_hist, axis=0)
            idx = int(avg.argmax())
            conf = float(avg[idx])
            if unknown_gesture:
                probs_hist.clear()
                label, color = "UNKNOWN GESTURE", (0, 0, 220)
            elif conf >= CONF_THRESHOLD and len(probs_hist) == SMOOTH_N:
                label, color = f"{CLASSES[idx]} {conf:.2f}", (0, 200, 0)
                if now - last_trigger >= COOLDOWN_SEC:
                    on_action(CLASSES[idx])
                    last_trigger = now
            else:
                label, color = f"WAITING ({conf:.2f})", (0, 180, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if unknown_gesture:
                cv2.putText(frame, "show a clear rock / paper / scissors shape",
                            (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            for hd in hands:
                if hd is not target:
                    bx1, by1, bx2, by2 = expand_square(hd.box, w, h)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (150, 150, 150), 1)
        else:
            probs_hist.clear()
            gate_fail_streak = 0
            cv2.putText(frame, label, (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        infer_ms = (time.time() - t0) * 1000
        cv2.putText(frame, f"{infer_ms:.0f} ms/frame", (w - 160, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 클래스별 확률 바 (좌하단) — WAITING 일 때도 내부 상태가 보이게
        if avg is not None:
            base_y = h - 76
            for i, (cname, p) in enumerate(zip(CLASSES, avg)):
                by = base_y + i * 22
                cv2.putText(frame, f"{cname:<8}{p:.2f}", (12, by + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (122, by), (122 + int(120 * p), by + 12),
                              (0, 200, 0) if i == idx else (160, 160, 160), -1)
            thr_x = 122 + int(120 * CONF_THRESHOLD)
            cv2.line(frame, (thr_x, base_y - 4), (thr_x, base_y + 3 * 22 - 6), (0, 180, 255), 1)

        cooldown_left = max(0.0, COOLDOWN_SEC - (now - last_trigger))
        if cooldown_left > 0:
            cv2.putText(frame, f"cooldown {cooldown_left:.1f}s", (w - 190, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        n_corr = sum(correction_counts.values())
        help_line = (f"1/2/3: save correction (rock/paper/scissors)  |  "
                     f"T: quick-tune ({n_corr}/{CORRECTION_MIN})  |  Q: quit")
        cv2.putText(frame, help_line, (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1)

        cv2.imshow("RPS — q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in KEY_TO_CLASS:
            if target is None or last_crop_box is None:
                print("보정 저장 실패: 화면에 손이 보여야 합니다.")
            else:
                cls = KEY_TO_CLASS[key]
                save_correction(frame, last_crop_box, cls, correction_idx)
                correction_counts[cls] += 1
                print(f"보정 저장: {cls} (누적 {sum(correction_counts.values())}장)")
        elif key == ord("t"):
            if sum(correction_counts.values()) < CORRECTION_MIN:
                print(f"보정 데이터 부족: {sum(correction_counts.values())}/{CORRECTION_MIN}")
                continue
            draw_center_message(
                frame, ["Updating model with your corrections...",
                        "(camera paused, ~1-2 min)"], (0, 200, 255))
            cv2.imshow("RPS — q to quit", frame)
            cv2.waitKey(1)
            before, after = quick_finetune(model, device)
            print(f"즉석 미세조정 완료: webcam_val {before:.3f} -> {after:.3f}")
            probs_hist.clear()
            gate_fail_streak = 0
            correction_counts = {c: 0 for c in CLASSES}
            ok2, frame2 = cap.read()
            msg = f"Done! val acc {before:.1%} -> {after:.1%}"
            if ok2:
                draw_center_message(frame2, [msg], (0, 200, 0))
                cv2.imshow("RPS — q to quit", frame2)
                cv2.waitKey(800)

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
