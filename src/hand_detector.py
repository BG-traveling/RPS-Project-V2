# -*- coding: utf-8 -*-
"""손 검출 공용 모듈 (MediaPipe Tasks HandLandmarker).

역할: 프레임에서 손들의 bbox 를 찾고, 분류기 입력용 정사각 크롭 좌표를 만든다.
분류는 기존 CNN(rps_model)이 담당 — 검출과 분류의 2단계 구조 (B안).

대상 손 선택 정책:
  - 최초: 가장 큰 손 (카메라에 가장 가까운 손)
  - 이후: 직전 대상과 IoU 가 가장 큰 손을 같은 손으로 간주해 추적
  - MISS_LIMIT 프레임 연속 미검출 시 추적 리셋
"""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_TASK_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
BOX_MARGIN = 0.3   # 손 bbox 확장 비율 — 학습 크롭과 유사하게 주변 배경 약간 포함
MISS_LIMIT = 10    # 이 프레임 수만큼 연속 미검출이면 추적 리셋


class HandDetector:
    def __init__(self, num_hands: int = 2, min_conf: float = 0.5):
        # model_asset_path 는 C 라이브러리가 열기 때문에 한글 경로에서 실패 →
        # 파이썬이 읽은 바이트 버퍼로 전달 (cv2 imread 우회와 같은 이유)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_buffer=MODEL_TASK_PATH.read_bytes()),
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray):
        """RGB 프레임 → 손 bbox 목록 [(x1, y1, x2, y2), ...] (픽셀 좌표)."""
        h, w = rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(mp_img)
        boxes = []
        for lms in result.hand_landmarks:
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            boxes.append((
                int(min(xs) * w), int(min(ys) * h),
                int(max(xs) * w), int(max(ys) * h),
            ))
        return boxes

    def close(self):
        self._detector.close()


def expand_square(box, img_w, img_h, margin: float = BOX_MARGIN):
    """bbox 를 margin 만큼 키운 뒤 정사각으로 만들어 프레임 안으로 클램프."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    side = min(side, img_w, img_h)
    half = side / 2
    cx = min(max(cx, half), img_w - half)
    cy = min(max(cy, half), img_h - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


class TargetTracker:
    """여러 손 중 대상 손 1개를 고르고 프레임 간 유지한다."""

    def __init__(self):
        self.prev_box = None
        self.misses = 0

    def select(self, boxes):
        """boxes 중 대상 손 bbox 를 반환 (없으면 None). switched 여부도 반환."""
        if not boxes:
            self.misses += 1
            if self.misses >= MISS_LIMIT:
                self.prev_box = None
            return None, False

        self.misses = 0
        if self.prev_box is not None:
            best = max(boxes, key=lambda b: iou(b, self.prev_box))
            if iou(best, self.prev_box) > 0.1:  # 같은 손으로 판단
                self.prev_box = best
                return best, False
        # 최초 선택 또는 추적 실패 → 가장 큰 손
        best = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        switched = self.prev_box is not None
        self.prev_box = best
        return best, switched or True
