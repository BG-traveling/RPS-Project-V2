# -*- coding: utf-8 -*-
"""실시간 웹캠 가위바위보 인식 앱 (PyTorch).

실행:  python src/webcam_app.py  (종료: q)

파이프라인:
  웹캠 프레임 → 정사각 ROI 크롭 → BGR→RGB → 224 리사이즈 → /255.0
  → 모델 추론(logits→softmax) → 최근 N=7 프레임 확률 평균 스무딩
  → 최대 확률 < 0.75 이면 "WAITING" 표시
  → 예측 안정화 후 쿨다운(2초) 적용 → 제스처-액션 매핑 트리거
"""
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rps_model import CLASSES, IMG_SIZE, MODEL_PATH, load_trained  # noqa: E402

SMOOTH_N = 7           # 스무딩 창 크기
CONF_THRESHOLD = 0.75  # 이 미만이면 대기 상태
COOLDOWN_SEC = 2.0     # 액션 트리거 후 재트리거 금지 시간
# ROI = 프레임의 "최대 중앙 정사각" — 학습 전처리(중앙 정사각 크롭) 및
# 촬영 도구(06)의 가이드와 동일한 영역. 임의 크기 ROI 는 학습 대비
# 확대/축소가 생겨 인식률을 떨어뜨린다 (고정 좌표 대신 프레임에서 계산).


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
    model = load_trained(device=device)  # eval 모드로 로드됨
    # 워밍업 — 첫 추론의 커널 초기화 비용을 미리 치름
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))
    print(f"모델 로드 완료: {MODEL_PATH} (device={device})")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    probs_hist = deque(maxlen=SMOOTH_N)
    last_trigger = 0.0
    x1 = y1 = x2 = y2 = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        s = min(h, w)  # 최대 중앙 정사각 (학습 크롭과 동일)
        x1, y1 = (w - s) // 2, (h - s) // 2
        x2, y2 = x1 + s, y1 + s

        t0 = time.time()
        roi = frame[y1:y2, x1:x2]
        inp = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        with torch.no_grad():
            x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        probs_hist.append(probs)
        infer_ms = (time.time() - t0) * 1000

        avg = np.mean(probs_hist, axis=0)
        idx = int(avg.argmax())
        conf = float(avg[idx])

        now = time.time()
        if conf >= CONF_THRESHOLD and len(probs_hist) == SMOOTH_N:
            label = f"{CLASSES[idx]} {conf:.2f}"
            color = (0, 200, 0)
            if now - last_trigger >= COOLDOWN_SEC:
                on_action(CLASSES[idx])
                last_trigger = now
        else:
            label = f"WAITING ({conf:.2f})"
            color = (0, 180, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 8, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{infer_ms:.0f} ms/frame", (x2 - 155, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 클래스별 확률 바 — 임계값 미달로 WAITING 일 때도 내부 상태가 보이게
        for i, (cname, p) in enumerate(zip(CLASSES, avg)):
            by = y1 + 52 + i * 22
            cv2.putText(frame, f"{cname:<8}{p:.2f}", (x1 + 8, by + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (x1 + 118, by), (x1 + 118 + int(120 * p), by + 12),
                          (0, 200, 0) if i == idx else (160, 160, 160), -1)
        thr_x = x1 + 118 + int(120 * CONF_THRESHOLD)  # 임계값 위치 표시
        cv2.line(frame, (thr_x, y1 + 50), (thr_x, y1 + 52 + 3 * 22 - 8), (0, 180, 255), 1)

        cooldown_left = max(0.0, COOLDOWN_SEC - (now - last_trigger))
        if cooldown_left > 0:
            cv2.putText(frame, f"cooldown {cooldown_left:.1f}s", (x1 + 8, y2 - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("RPS — q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
