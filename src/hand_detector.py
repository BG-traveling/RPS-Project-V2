# -*- coding: utf-8 -*-
"""손 검출 공용 모듈 (MediaPipe Tasks HandLandmarker).

역할: 프레임에서 손들의 bbox+관절좌표를 찾고, 분류기 입력용 정사각 크롭
좌표와 "학습된 3제스처 중 하나인가"를 판정한다. 실제 rock/paper/scissors
구분은 기존 CNN(rps_model)이 담당 — 검출·형태게이트·분류의 구조 (B안).

대상 손 선택 정책:
  - 최초: 가장 큰 손 (카메라에 가장 가까운 손)
  - 이후: 직전 대상과 IoU 가 가장 큰 손을 같은 손으로 간주해 추적
  - MISS_LIMIT 프레임 연속 미검출 시 추적 리셋
"""
import math
from collections import namedtuple
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_TASK_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
BOX_MARGIN = 0.3   # 손 bbox 확장 비율 — 학습 크롭과 유사하게 주변 배경 약간 포함
MISS_LIMIT = 10    # 이 프레임 수만큼 연속 미검출이면 추적 리셋

# landmarks 는 정규화 좌표(x, y, z in ~[-1,1], z=상대 깊이). box 는 픽셀 좌표.
Hand = namedtuple("Hand", ["box", "landmarks"])


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
        """RGB 프레임 → Hand(box, landmarks) 목록."""
        h, w = rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(mp_img)
        hands = []
        for lms in result.hand_landmarks:
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            box = (int(min(xs) * w), int(min(ys) * h), int(max(xs) * w), int(max(ys) * h))
            landmarks = [(p.x, p.y, p.z) for p in lms]
            hands.append(Hand(box, landmarks))
        return hands

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

    def select(self, hands):
        """hands 중 대상 Hand 를 반환 (없으면 None). switched 여부도 반환."""
        if not hands:
            self.misses += 1
            if self.misses >= MISS_LIMIT:
                self.prev_box = None
            return None, False

        self.misses = 0
        if self.prev_box is not None:
            best = max(hands, key=lambda h: iou(h.box, self.prev_box))
            if iou(best.box, self.prev_box) > 0.1:  # 같은 손으로 판단
                self.prev_box = best.box
                return best, False
        # 최초 선택 또는 추적 실패 → 가장 큰 손
        best = max(hands, key=lambda h: (h.box[2] - h.box[0]) * (h.box[3] - h.box[1]))
        self.prev_box = best.box
        return best, True


# --- 기하학적 제스처 게이트 (미학습 손모양 거부) ---------------------------
#
# MediaPipe 관절 좌표로 "손가락이 펴졌는지"를 직접 계산해, CNN 을 거치기 전에
# rock/paper/scissors 세 패턴 중 하나에 해당하는지 먼저 확인한다.
# 각도 기반이라 손이 어느 방향으로 돌아가 있어도(옆면, 비스듬 등) 픽셀 CNN보다
# 훨씬 안정적으로 "펴짐/굽음"을 판별한다 — 이 프로젝트에서 겪은 각도 관련
# 오인식들과 같은 뿌리의 문제를 기하학적으로 우회하는 방법이다.
#
# 각 손가락(엄지 제외)의 PIP 관절에서 MCP↔TIP 사이 각도를 재서, 거의
# 일직선(FINGER_STRAIGHT_DEG 이상)이면 "펴짐"으로 판정한다. 엄지는 완전히
# 펴진 상태를 취하기 어렵고(자세마다 각도 편차가 큼) 판정에서 제외한다.
# 임계값·규칙은 실제 촬영 데이터(scripts/diag_geometric_gate.py)로 검증한
# 값이다 — 중지(middle)는 카메라를 향한 각도에서 유독 오검출이 잦아
# (실측 시 정답 프레임의 10%가량이 굽음으로 오판됨) scissors 판정에서
# 제외했다. 검지 펴짐 + 약지·소지 굽음만으로 판정 시 정확도가
# 뚜렷하게 개선됨(가위 recall 64%→87%).
FINGER_JOINTS = {  # (MCP, PIP, TIP) landmark 인덱스
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
FINGER_STRAIGHT_DEG = 150.0


def _joint_angle(landmarks, mcp_i, pip_i, tip_i):
    """MCP-PIP-TIP 사이 각도(3D). 2D(x,y)만 쓰면 손가락이 카메라를 향할 때
    투영 원근으로 실제론 편 손가락도 굽어 보이는 문제가 있어, MediaPipe 의
    상대 깊이(z)까지 포함해 계산한다 — 카메라 방향(회전) 오판 완화."""
    mcp, pip, tip = landmarks[mcp_i], landmarks[pip_i], landmarks[tip_i]
    v1 = tuple(mcp[i] - pip[i] for i in range(3))
    v2 = tuple(tip[i] - pip[i] for i in range(3))
    n1 = math.sqrt(sum(c * c for c in v1))
    n2 = math.sqrt(sum(c * c for c in v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    dot = sum(a * b for a, b in zip(v1, v2))
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def extended_fingers(landmarks):
    """{"index": bool, "middle": bool, "ring": bool, "pinky": bool} — 엄지 제외."""
    return {
        name: _joint_angle(landmarks, *joints) > FINGER_STRAIGHT_DEG
        for name, joints in FINGER_JOINTS.items()
    }


def classify_geometric(landmarks):
    """관절 각도만으로 rock/paper/scissors 판정. 셋 다 아니면 None (미학습 모양)."""
    ext = extended_fingers(landmarks)
    n = sum(ext.values())
    if n == 0:
        return "rock"
    if n == 4:
        return "paper"
    if ext["index"] and not ext["ring"] and not ext["pinky"]:  # middle 은 노이즈가 커 미사용
        return "scissors"
    return None
