# -*- coding: utf-8 -*-
"""05. 자체 웹캠(실전) 데이터 fine-tuning.

Data/webcam/{rock,paper,scissors}/ 의 영상 추출 프레임(360×640 세로, 실사 배경)을
합성 train 에 섞어 기존 모델을 이어서 학습한다.

- 프레임 리크 방지: 연속 프레임이므로 클래스별 시간순 앞 70% → train,
  뒤 30% → webcam_val (무작위 분할은 인접 프레임이 양쪽에 흩어져 수치가 부풀려짐).
- 웹캠 train 은 ×5 반복해 합성 3,062장 대비 비중(약 32%)을 확보.
- 저장된 모델(base 동결)을 로드해 head 만 lr=1e-4 로 이어서 학습.
- 평가: webcam_val(실전 지표) + test_original/test_synthetic(망각 여부 확인).
"""
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
WEBCAM_DIR = ROOT / "Data" / "webcam"
SYN = ROOT / "data_synthetic"
OUT = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "rps_mobilenetv2.keras"
BACKUP_PATH = ROOT / "models" / "rps_mobilenetv2_prev.keras"
CLASSES = ["rock", "paper", "scissors"]
IMG_SIZE = 224
BATCH = 32
SEED = 42
WEBCAM_REPEAT = 5
TRAIN_FRACTION = 0.7


def webcam_split():
    """클래스별 시간순 70/30 분할 (연속 프레임 리크 방지)."""
    train, val = [], []
    for i, c in enumerate(CLASSES):
        fs = sorted((WEBCAM_DIR / c).glob("*.jpg")) + sorted((WEBCAM_DIR / c).glob("*.png"))
        cut = int(len(fs) * TRAIN_FRACTION)
        train += [(str(f), i) for f in fs[:cut]]
        val += [(str(f), i) for f in fs[cut:]]
        print(f"webcam {c}: {len(fs)}장 → train {cut} / val {len(fs) - cut}")
    return train, val


def list_syn(split_dir):
    items = []
    for i, c in enumerate(CLASSES):
        items += [(str(f), i) for f in sorted((split_dir / c).glob("*.png"))]
    return items


def decode(path, label, crop_square):
    """crop_square=True 면 중앙 정사각 크롭(비율 보존) 후 리사이즈.

    세로 영상 프레임(360×640)을 그대로 224×224 로 뭉개면 세로가 1.8배 눌려
    손가락 형태가 붕괴됨 → 웹캠 프레임은 중앙 360×360 크롭으로 왜곡 제거 +
    손 비중 확대. 합성 이미지(300×200)는 기존 규격(왜곡 리사이즈) 유지.
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    if crop_square:
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        size = tf.minimum(h, w)
        img = tf.image.crop_to_bounding_box(
            img, (h - size) // 2, (w - size) // 2, size, size
        )
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label


def make_ds(items, shuffle=False, crop_square=False):
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: decode(p, l, crop_square),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds, np.array(labels)


def evaluate_set(model, name, items, report_lines, crop_square=False):
    from sklearn.metrics import classification_report, confusion_matrix

    ds, y_true = make_ds(items, crop_square=crop_square)
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    report_lines += [f"\n===== {name} =====", f"accuracy: {acc:.4f}",
                    f"confusion matrix (rows=true {CLASSES}):\n{cm}", rep]
    print(f"{name} accuracy: {acc:.4f}")
    return acc


def main():
    tf.keras.utils.set_random_seed(SEED)
    wc_train, wc_val = webcam_split()

    # 합성(왜곡 리사이즈)과 웹캠(정사각 크롭)은 전처리가 달라 각각 디코딩 후 결합
    syn_items = list_syn(SYN / "train")
    wc_items = wc_train * WEBCAM_REPEAT
    print(f"train 총 {len(syn_items) + len(wc_items)}장 (합성 {len(syn_items)} + 웹캠 {len(wc_train)}×{WEBCAM_REPEAT})")

    def raw_ds(items, crop_square):
        ds = tf.data.Dataset.from_tensor_slices(
            ([p for p, _ in items], [l for _, l in items])
        )
        return ds.map(lambda p, l: decode(p, l, crop_square),
                      num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        raw_ds(syn_items, False)
        .concatenate(raw_ds(wc_items, True))
        .shuffle(len(syn_items) + len(wc_items), seed=SEED,
                 reshuffle_each_iteration=True)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds, _ = make_ds(wc_val, crop_square=True)  # 실전 지표로 조기종료 판단

    shutil.copy2(MODEL_PATH, BACKUP_PATH)  # 이전 모델 백업
    model = keras.models.load_model(MODEL_PATH)
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    base.trainable = False  # head-only 유지 (2단계 unfreeze 는 이전에 발산)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # 평가 — 디스크의 최고 성능 모델 기준
    model = keras.models.load_model(MODEL_PATH)
    report_lines = []
    acc_wc = evaluate_set(model, "webcam_val (실전 지표)", wc_val, report_lines, crop_square=True)
    acc_orig = evaluate_set(model, "test_original", list_syn(SYN / "test_original"), report_lines)
    acc_syn = evaluate_set(model, "test_synthetic", list_syn(SYN / "test_synthetic"), report_lines)
    report_lines += ["\n===== 요약 =====",
                    f"webcam_val: {acc_wc:.4f} / test_original: {acc_orig:.4f} / test_synthetic: {acc_syn:.4f}"]
    (OUT / "evaluation_webcam_ft.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"리포트 → {OUT / 'evaluation_webcam_ft.txt'}")
    print(f"이전 모델 백업 → {BACKUP_PATH}")


if __name__ == "__main__":
    main()
