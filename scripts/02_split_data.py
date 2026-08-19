# -*- coding: utf-8 -*-
"""02. train/val/test 분리 — stratified 70:15:15 (작업계획서 §4-2).

split 결과를 splits/{train,val,test}.csv 로 저장한다. 이후 단계(배경 합성,
학습)는 모두 이 파일 목록만 참조하므로 데이터 리크가 원천 차단된다 (§5-2).
"""
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
SPLIT_DIR = ROOT / "splits"
CLASSES = ["rock", "paper", "scissors"]
SEED = 42


def main():
    SPLIT_DIR.mkdir(exist_ok=True)
    paths, labels = [], []
    for c in CLASSES:
        for f in sorted((DATA_DIR / c).glob("*.png")):
            paths.append(f.relative_to(ROOT).as_posix())
            labels.append(c)

    # 1단계: 70% train / 30% temp, 2단계: temp를 반씩 val/test → 70:15:15
    tr_p, tmp_p, tr_y, tmp_y = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    va_p, te_p, va_y, te_y = train_test_split(
        tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=SEED
    )

    for name, ps, ys in [("train", tr_p, tr_y), ("val", va_p, va_y), ("test", te_p, te_y)]:
        with open(SPLIT_DIR / f"{name}.csv", "w", encoding="utf-8") as fh:
            fh.write("path,label\n")
            for p, y in zip(ps, ys):
                fh.write(f"{p},{y}\n")
        dist = {c: ys.count(c) for c in CLASSES}
        print(f"{name}: {len(ps)}장 {dist}")

    print("분리 완료 → splits/*.csv")


if __name__ == "__main__":
    main()
