# -*- coding: utf-8 -*-
"""실시간 웹캠 가위바위보 인식 앱 (PyTorch + MediaPipe 손 검출, B안 2단계 구조).

실행:  python src/webcam_app.py  (종료: q)

파이프라인:
  프레임 → 손 검출(여러 개 가능) → 대상 손 선택(최초: 최대 크기, 이후: IoU 추적)
  → bbox 정사각+30% 마진 크롭 → BGR→RGB → 224 리사이즈 → /255.0
  → CNN 분류(logits→softmax) → 최근 N=7 프레임 확률 평균 스무딩
  → 최대 확률 < 0.75 → "WAITING" / 손 미검출 → "NO HAND"
  → 예측 안정화 후 쿨다운(2초) → 제스처-액션 매핑

화면: 대상 손 = 초록 박스 + 라벨, 그 외 손 = 회색 박스, 클래스별 확률 바.
"""
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_detector import HandDetector, TargetTracker, expand_square  # noqa: E402
from rps_model import CLASSES, IMG_SIZE, MODEL_PATH, load_trained  # noqa: E402

SMOOTH_N = 7           # 스무딩 창 크기
CONF_THRESHOLD = 0.75  # 이 미만이면 대기 상태
COOLDOWN_SEC = 2.0     # 액션 트리거 후 재트리거 금지 시간


def on_action(gesture: str):
    """제스처-액션 매핑. 필요에 맞게 교체 (키 입력 전송, 게임 로직 등)."""
    actions = {
        "rock": lambda: print(">>> ACTION: ROCK — 공격!"),
        "paper": lambda: print(">>> ACTION: PAPER — 방어!"),
        "scissors": lambda: print(">>> ACTION: SCISSORS — 스킬!"),
    }
    actions[gesture]()


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

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        now = time.time()

        t0 = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detector.detect(rgb)
        target, switched = tracker.select(boxes)
        if switched:
            probs_hist.clear()  # 대상 손이 바뀌면 이전 손의 확률 섞임 방지

        label, color = "NO HAND", (120, 120, 120)
        avg, idx = None, 0
        if target is not None:
            x1, y1, x2, y2 = expand_square(target, w, h)
            inp = cv2.resize(rgb[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
            inp = inp.astype(np.float32) / 255.0
            with torch.no_grad():
                x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            probs_hist.append(probs)

            avg = np.mean(probs_hist, axis=0)
            idx = int(avg.argmax())
            conf = float(avg[idx])
            if conf >= CONF_THRESHOLD and len(probs_hist) == SMOOTH_N:
                label, color = f"{CLASSES[idx]} {conf:.2f}", (0, 200, 0)
                if now - last_trigger >= COOLDOWN_SEC:
                    on_action(CLASSES[idx])
                    last_trigger = now
            else:
                label, color = f"WAITING ({conf:.2f})", (0, 180, 255)

            # 대상 손 박스 + 나머지 손 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for b in boxes:
                if b is not target:
                    bx1, by1, bx2, by2 = expand_square(b, w, h)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (150, 150, 150), 1)
        else:
            probs_hist.clear()
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

        cv2.imshow("RPS — q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
