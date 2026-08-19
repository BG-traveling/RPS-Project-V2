# -*- coding: utf-8 -*-
"""기하학적 제스처 게이트(hand_detector.classify_geometric) 정확도 검증.

실제 촬영 데이터에서 "정상 제스처인데 None(거부)으로 오판"하는 비율을 재서
FINGER_STRAIGHT_DEG 임계값이 적절한지 확인한다. 이 비율이 높으면 실사용 중
정상적으로 손을 냈는데도 "정확한 모양을 보여주세요"가 계속 뜨는 문제가 생긴다.
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hand_detector import HandDetector, classify_geometric  # noqa: E402

WEBCAM_DIR = ROOT / "Data" / "webcam"
CLASSES = ["rock", "paper", "scissors"]
N_PER_CLASS = 200


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    det = HandDetector(num_hands=1)
    rng = np.random.default_rng(42)
    for c in CLASSES:
        files = sorted((WEBCAM_DIR / c).glob("*.jpg"))
        picks = rng.choice(len(files), size=min(N_PER_CLASS, len(files)), replace=False)
        results = Counter()
        no_hand = 0
        for i in picks:
            img = imread_u(files[i])
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hands = det.detect(rgb)
            if not hands:
                no_hand += 1
                continue
            target = max(hands, key=lambda h: (h.box[2] - h.box[0]) * (h.box[3] - h.box[1]))
            geo = classify_geometric(target.landmarks)
            results[geo] += 1
        total = sum(results.values())
        correct = results.get(c, 0)
        print(f"{c:<9} n={total:4d}  정답={correct:4d} ({correct/total:.1%})  "
              f"분포={dict(results)}  검출실패={no_hand}")
    det.close()


if __name__ == "__main__":
    main()
