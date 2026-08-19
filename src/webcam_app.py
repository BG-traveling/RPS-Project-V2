# -*- coding: utf-8 -*-
"""실시간 웹캠 가위바위보 인식 앱 (작업계획서 §7).

실행:  python src/webcam_app.py  (종료: q)

파이프라인:
  웹캠 프레임 → 고정 ROI 크롭 → BGR→RGB → 224 리사이즈 → /255.0
  → 모델 추론 → 최근 N=7 프레임 확률 평균 스무딩
  → 최대 확률 < 0.75 이면 "WAITING" 표시
  → 예측 안정화 후 쿨다운(2초) 적용 → 제스처-액션 매핑 트리거
"""
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rps_mobilenetv2.keras"
CLASSES = ["rock", "paper", "scissors"]  # 04_train_model.py 의 라벨 순서와 동일

IMG_SIZE = 224
SMOOTH_N = 7          # 스무딩 창 크기
CONF_THRESHOLD = 0.75  # 이 미만이면 대기 상태
COOLDOWN_SEC = 2.0     # 액션 트리거 후 재트리거 금지 시간
# 학습 데이터가 300×200(3:2)을 224×224로 왜곡 리사이즈하므로,
# ROI도 3:2로 잡아 동일한 왜곡이 걸리게 한다 (정사각 ROI는 비율 불일치).
ROI = (95, 120, 545, 420)  # (x1, y1, x2, y2) — 450×300, 3:2


def on_action(gesture: str):
    """제스처-액션 매핑. 필요에 맞게 교체 (키 입력 전송, 게임 로직 등)."""
    actions = {
        "rock": lambda: print(">>> ACTION: ROCK — 공격!"),
        "paper": lambda: print(">>> ACTION: PAPER — 방어!"),
        "scissors": lambda: print(">>> ACTION: SCISSORS — 스킬!"),
    }
    actions[gesture]()


def main():
    import tensorflow as tf  # 로딩이 느려 함수 안에서 임포트

    model = tf.keras.models.load_model(MODEL_PATH)

    # model.predict()는 호출당 오버헤드가 커서 실시간 루프에 부적합.
    # tf.function 직접 호출로 컴파일해 프레임당 지연을 최소화한다.
    infer = tf.function(
        lambda x: model(x, training=False),
        input_signature=[tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32)],
    )
    infer(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]))  # 워밍업(트레이싱)
    print(f"모델 로드 완료: {MODEL_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    probs_hist = deque(maxlen=SMOOTH_N)
    last_trigger = 0.0
    x1, y1, x2, y2 = ROI

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        roi = frame[y1:y2, x1:x2]
        inp = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        probs = infer(inp[None, ...]).numpy()[0]
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
        cv2.putText(frame, label, (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{infer_ms:.0f} ms/frame", (x2 - 150, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "fill the box with your hand", (x1, y2 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cooldown_left = max(0.0, COOLDOWN_SEC - (now - last_trigger))
        if cooldown_left > 0:
            cv2.putText(frame, f"cooldown {cooldown_left:.1f}s", (x1, y2 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("RPS — q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
