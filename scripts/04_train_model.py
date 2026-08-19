# -*- coding: utf-8 -*-
"""04. MobileNetV2 전이학습 및 평가 (작업계획서 §6).

- 파이프라인: read_file → decode_png → resize(224) → /255.0 → shuffle → batch(32) → prefetch
- 정규화 불일치(§6-2): 파이프라인은 0~1 유지, 모델 입력단 Rescaling(2.0, offset=-1.0)
  으로 −1~1 변환 → 학습·추론 규격 자동 일치.
- 증강(train 전용): RandomFlip / RandomRotation(0.1) / RandomZoom(0.1)
  / RandomBrightness(0.2, value_range=(0,1)) / RandomContrast(0.2) + clip
- 1단계: base 동결 10 epoch, EarlyStopping(val_loss, patience=5, restore_best)
- 평가: 학습 곡선, confusion matrix, 클래스별 리포트,
  test_original vs test_synthetic 정확도 비교 (§6-4)
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data_synthetic"
OUT = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "rps_mobilenetv2.keras"
CLASSES = ["rock", "paper", "scissors"]  # label index = 이 순서
IMG_SIZE = 224
BATCH = 32
SEED = 42


def list_files(split_dir):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        for f in sorted((split_dir / c).glob("*.png")):
            paths.append(str(f))
            labels.append(i)
    return paths, labels


def decode(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label


def make_ds(split_dir, shuffle=False):
    paths, labels = list_files(split_dir)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds, np.array(labels)


def build_model():
    # 원본 데이터는 손이 전부 가로 방향 → 웹캠(손가락 위)과 방향 갭이 큼.
    # RandomRotation(0.25)=±90° 로 방향 커버리지 확보, Translation 으로 위치 변화 대응.
    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.25),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(0.2, value_range=(0, 1)),
            layers.RandomContrast(0.2),
        ],
        name="augmentation",
    )
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = keras.ops.clip(x, 0.0, 1.0)  # 증강 후 0~1 범위 보장 (§4-2 트러블슈팅)
    x = layers.Rescaling(2.0, offset=-1.0)(x)  # 0~1 → −1~1 (§6-2)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(hist):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(hist["accuracy"], label="train")
    ax1.plot(hist["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax2.plot(hist["loss"], label="train")
    ax2.plot(hist["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=120)
    plt.close(fig)


def evaluate_set(model, name, split_dir, report_lines):
    from sklearn.metrics import classification_report, confusion_matrix

    ds, y_true = make_ds(split_dir)
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    report_lines.append(f"\n===== {name} =====")
    report_lines.append(f"accuracy: {acc:.4f}")
    report_lines.append(f"confusion matrix (rows=true {CLASSES}):\n{cm}")
    report_lines.append(rep)
    print(f"{name} accuracy: {acc:.4f}")
    return acc, cm


def plot_cm(cm, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name}.png", dpi=120)
    plt.close(fig)


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUT.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    train_ds, _ = make_ds(SYN / "train", shuffle=True)
    val_ds, _ = make_ds(SYN / "val")

    model = build_model()
    model.summary(line_length=100)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # 2단계 fine-tuning (§6-3): 상위 블록 unfreeze, lr=1e-5.
    # 가위/보 혼동 완화 목적 — 저수준 특징은 유지하고 상위 표현만 미세 조정.
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    base.trainable = True
    fine_tune_from = len(base.layers) - 40  # 상위 약 2개 블록
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ft_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True,
            # 1단계 최고 기록보다 나빠지면 저장하지 않음 (모델 파일 퇴행 방지)
            initial_value_threshold=min(history.history["val_loss"]),
        ),
    ]
    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=ft_callbacks)

    merged = {
        k: history.history[k] + history_ft.history[k] for k in history.history
    }
    plot_history(merged)

    # 평가 (§6-4) — 디스크에 저장된 전체 최고 성능 모델 기준
    model = keras.models.load_model(MODEL_PATH)
    report_lines = []
    acc_orig, cm_orig = evaluate_set(model, "test_original", SYN / "test_original", report_lines)
    acc_syn, cm_syn = evaluate_set(model, "test_synthetic", SYN / "test_synthetic", report_lines)
    plot_cm(cm_orig, "test_original")
    plot_cm(cm_syn, "test_synthetic")

    report_lines.append("\n===== 배경 증강 효과 비교 =====")
    report_lines.append(f"test_original(초록 배경): {acc_orig:.4f}")
    report_lines.append(f"test_synthetic(실사 배경): {acc_syn:.4f}")
    (OUT / "evaluation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    json.dump(
        {"classes": CLASSES, "img_size": IMG_SIZE,
         "test_original_acc": acc_orig, "test_synthetic_acc": acc_syn},
        open(OUT / "metrics.json", "w", encoding="utf-8"), indent=2,
    )
    print(f"모델 저장 → {MODEL_PATH}")
    print(f"리포트 → {OUT / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
