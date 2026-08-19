# -*- coding: utf-8 -*-
"""06. 웹캠으로 학습 데이터 직접 촬영.

사용법:
  python scripts/06_capture_webcam.py scissors            # 클래스 지정 (배치 300장)
  python scripts/06_capture_webcam.py rock --count 100    # 배치 크기 변경
  python scripts/06_capture_webcam.py paper --count 0     # 무제한 (수동 정지)
  python scripts/06_capture_webcam.py rock --interval 5   # 연속 촬영 간격(프레임)

조작:
  SPACE : 1장 촬영
  R     : 배치 촬영 시작/정지 — count 장을 채우면 자동 정지, 다시 R 로 다음 배치
  Q     : 종료

저장: Data/webcam/<클래스>/<클래스>_cap<세션시각>__f0000.jpg
  - 파일명의 세션 프리픽스(cap<시각>) 덕분에 rps_data.webcam_split 이
    세션 단위로 70/30 시간순 분할을 수행한다 (연속 프레임 리크 방지).
  - 화면의 정사각 가이드는 학습 전처리(중앙 정사각 크롭)와 같은 영역 —
    손이 가이드 안에 오도록 다양한 각도/거리/위치로 촬영하는 것이 좋다.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["rock", "paper", "scissors"]


def imwrite_u(path, img):
    """한글 경로에서도 동작하는 imwrite (cv2 Windows 제약 우회)."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise RuntimeError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cls", choices=CLASSES, help="촬영할 클래스")
    ap.add_argument("--interval", type=int, default=3,
                    help="연속 촬영 시 저장 간격(프레임), 기본 3")
    ap.add_argument("--count", type=int, default=300,
                    help="배치 크기 — R 연속 촬영이 이 장수를 채우면 자동 정지 (0 = 무제한)")
    args = ap.parse_args()

    out_dir = ROOT / "Data" / "webcam" / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("cap%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    n_saved = 0
    frame_i = 0
    recording = False
    batch_saved = 0  # 현재 배치에서 저장한 장수
    flash_until = 0.0

    batch_label = "무제한" if args.count == 0 else f"{args.count}장"
    print(f"[{args.cls}] 촬영 시작 — SPACE: 1장 / R: 배치({batch_label}) 토글 / Q: 종료")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        h, w = frame.shape[:2]
        s = min(h, w)
        x0, y0 = (w - s) // 2, (h - s) // 2

        do_save = False
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            do_save = True
        elif key == ord("r"):
            recording = not recording
            if recording:
                batch_saved = 0  # 새 배치 시작
        if recording and frame_i % args.interval == 0:
            do_save = True

        if do_save:
            name = f"{args.cls}_{session}__f{n_saved:04d}.jpg"
            imwrite_u(out_dir / name, frame)  # 원본 프레임 그대로 저장 (크롭은 학습 파이프라인 담당)
            n_saved += 1
            flash_until = time.time() + 0.15
            if recording:
                batch_saved += 1
                if args.count and batch_saved >= args.count:
                    recording = False  # 배치 완료 → 자동 정지
                    print(f"배치 완료: {batch_saved}장 (누적 {n_saved}장)")

        # --- 오버레이 ---
        view = frame.copy()
        guide_color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), guide_color, 2)
        status = f"[{args.cls}] saved: {n_saved}"
        if recording:
            status += f"  ● REC {batch_saved}/{args.count if args.count else '-'}"
        cv2.putText(view, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if recording else (255, 255, 255), 2)
        if recording and args.count:  # 배치 진행률 바
            bar_w = int((w - 24) * batch_saved / args.count)
            cv2.rectangle(view, (12, 44), (12 + bar_w, 52), (0, 0, 255), -1)
            cv2.rectangle(view, (12, 44), (w - 12, 52), (255, 255, 255), 1)
        cv2.putText(view, "SPACE: shot | R: record | Q: quit", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        if time.time() < flash_until:  # 저장 피드백 플래시
            view = cv2.addWeighted(view, 0.6, np.full_like(view, 255), 0.4, 0)
        cv2.imshow(f"capture: {args.cls} — q to quit", view)

    cap.release()
    cv2.destroyAllWindows()
    print(f"총 {n_saved}장 저장 → {out_dir} (세션 {session})")
    if n_saved:
        print("다음 단계: python scripts/05_finetune_webcam.py 재실행으로 반영")


if __name__ == "__main__":
    main()
