# RPS(가위바위보) 손 제스처 인식

웹캠으로 실시간 가위/바위/보 손 모양을 인식해 액션을 트리거하는 딥러닝 미니 프로젝트.
MobileNetV2(ImageNet) 전이학습 사용. 상세 배경은 [작업계획서.md](작업계획서.md) 참고.

## 프로젝트 구조

```
Data/{rock,paper,scissors}/     원본 데이터셋 (Kaggle rps-cv-images, 2,188장, 초록 배경)
backgrounds/                    Unsplash 배경 50장 (크로마키 합성용)
splits/                         stratified 70/15/15 분리 목록 (train 1,531 / val 328 / test 329)
data_synthetic/                 배경 합성 결과 (train 3,062 / val 328 / test 원본+합성 329×2)
scripts/
  01_eda.py                     EDA — 클래스 분포, 규격/손상, RGB 채널(배경 편향) 분석
  02_split_data.py              stratified 분리 → splits/*.csv (데이터 리크 방지 기준점)
  03_background_synthesis.py    HSV 크로마키 마스킹 + 모폴로지 + 소프트 블렌딩 합성
  04_train_model.py             MobileNetV2 전이학습 + 평가 (PyTorch + wandb)
  05_finetune_webcam.py         실전 웹캠 데이터 fine-tuning (PyTorch + wandb)
  06_capture_webcam.py          웹캠으로 학습 데이터 직접 촬영 (배치 촬영)
  07_clean_webcam_data.py       손 없는 프레임 자동 격리 (피부색 비율 기반)
  08_crop_hands.py              손 검출 크롭으로 학습 데이터 재생성 (B안)
  diag_webcam.py                진단 도구 (train/val 분리 평가, 타임라인, 몽타주)
  diag_geometric_gate.py        관절각도 게이트(classify_geometric) 정확도 검증
src/
  rps_model.py                  공용 모델 정의 (정규화 내장, 동결/unfreeze 제어)
  rps_data.py                   공용 데이터셋 (도메인별 전처리, 시간순 분할)
  rps_train.py                  공용 학습 루프 (조기종료, best 체크포인트, wandb 로깅)
  hand_detector.py              MediaPipe 손 검출 + 대상 손 추적 + 관절각도 게이트
  webcam_app.py                 실시간 웹캠 추론 (검출 기반 동적 박스, 미학습 모양 거부,
                                 즉석 보정학습)
models/rps_mobilenetv2.pt       학습된 모델 (학습↔추론은 이 파일로만 연결)
models/hand_landmarker.task     MediaPipe 손 검출 사전학습 모델
outputs/                        EDA/합성/학습 그래프, confusion matrix, 평가 리포트
old/                            더 이상 코드에서 참조하지 않는 파일 보관 (Keras 모델 등)
```

## 실행 방법

```bash
# 1) 환경
python -m venv .venv
.venv\Scripts\pip install torch torchvision wandb opencv-python scikit-learn matplotlib pillow

# 2) wandb (선택 — 미로그인 시 WANDB_MODE=offline 로 로컬 기록 후 wandb sync 로 업로드)
.venv\Scripts\wandb login

# 3) 파이프라인 (순서대로)
.venv\Scripts\python scripts\01_eda.py
.venv\Scripts\python scripts\02_split_data.py
.venv\Scripts\python scripts\03_background_synthesis.py
.venv\Scripts\python scripts\04_train_model.py        # 합성 데이터 1차 학습
.venv\Scripts\python scripts\05_finetune_webcam.py    # 실전 웹캠 데이터 fine-tuning

# 4) 실시간 웹캠 앱
.venv\Scripts\python src\webcam_app.py
```

**웹캠 앱 조작키**

| 키 | 동작 |
|---|---|
| `1`/`2`/`3` (또는 `r`/`p`/`s`) | 화면이 잘못 인식됐을 때 정답을 눌러 그 프레임을 학습 데이터로 저장 (즉석 보정) |
| `T` | 지금까지 모은 보정 데이터로 즉석 미세조정 (카메라 일시정지 1~2분, 완료 후 자동 반영·재시작 불필요). 최소 6장 필요 |
| `Q` | 종료 |

- **미학습 손모양 거부**: MediaPipe 관절 좌표로 "펴진 손가락 패턴"을 직접 계산해
  rock(0개)/scissors(검지만)/paper(4개) 중 어디에도 안 맞으면 5프레임 연속 확인 후
  CNN 결과 대신 `UNKNOWN GESTURE`를 표시한다 (`src/hand_detector.py:classify_geometric`).
  각도 기반이라 손 방향이 달라져도 픽셀 분류보다 안정적이다.
- **즉석 보정학습**: `T`는 `05_finetune_webcam.py`와 동일한 안전장치(정확도 기준
  체크포인트, 이전 가중치 자동 백업, 퇴행 시 미저장)로 웹캠 데이터만 짧게(기본 3 epoch)
  이어서 학습한다. 합성 데이터를 포함한 정식 재학습을 대체하지 않으며, 저장된 보정
  프레임은 다음에 `05_finetune_webcam.py`를 정식으로 돌릴 때도 자동으로 포함된다.

프레임워크: **PyTorch** (torchvision MobileNetV2). 학습 지표는 wandb `rps-project`
프로젝트에 epoch 단위로 로깅된다 (train/val loss·accuracy, 최종 confusion matrix).
모델 정의는 `src/rps_model.py` 하나에 있고, 학습·추론 모두 이 모듈을 import 한다 —
입력 계약(0~1 float)과 ImageNet 정규화가 모델 forward 에 내장되어 규격 불일치가
구조적으로 불가능하다.

## 핵심 설계 결정

- **배경 편향 대응**: 원본이 전부 초록 크로마키 배경이라(EDA에서 G채널 평균 140 vs R 80/B 65)
  배경색 과적합 위험 → 실사 배경 50종으로 크로마키 합성 증강.
- **데이터 리크 방지**: split을 먼저 확정(`splits/*.csv`)하고 각 세트 **안에서만** 합성.
  같은 손 인스턴스가 train/test 에 흩어지는 누수 원천 차단.
- **정규화 일치**: 파이프라인은 0~1(`/255.0`) 유지, 모델 입력단 `Rescaling(2.0, offset=-1.0)`
  레이어로 MobileNetV2가 기대하는 −1~1 로 변환 → 학습/추론 규격 자동 동일.
- **test 이중 평가**: test_original(초록 배경) vs test_synthetic(실사 배경) 정확도를 비교해
  배경 증강 효과를 정량 확인.

## 학습 결과 (2026-08-19, PyTorch + GPU, 웹캠 데이터 2,564장 반영 최종)

| 지표 | 값 |
|---|---|
| **webcam_val (실전 지표, 650장)** | **0.934** — rock 95% / paper 93% / scissors 92% |
| test_original | 0.918 |
| test_synthetic | 0.930 |
| 확대+블러 스트레스 (실전 조건 시뮬레이션) | recall 91~96%, 클래스 쏠림 없음 |

- 학습 곡선·confusion matrix: wandb `rps-project` 프로젝트 (04: base 학습, 05: fine-tuning)
- 05 체크포인트는 **webcam_val 정확도 기준**으로 저장 — 분포 밖 검증에서는
  정확도가 올라도 loss 가 높게 유지될 수 있어, loss 기준 저장이 개선을 전부
  걸러버리는 문제를 겪은 뒤의 설계 결정.
- 실전 "paper 쏠림" 사건: 모션 블러에서 paper 과대 예측 + 앱 ROI(360)와 학습
  크롭(480)의 확대 불일치 + 임계값이 저확신 rock/scissors 를 WAITING 으로 가림.
  → 블러/확대 증강 추가, 앱 ROI 를 학습 크롭과 동일한 최대 중앙 정사각으로 교체,
  앱에 클래스별 확률 바 오버레이 추가 (`scripts/diag_live_gap.py` 로 재현 가능).

<details>
<summary>이전 Keras 시절 기록</summary>

## 학습 결과 (2026-08-19, 2차 — 웹캠 실전 피드백 반영 재학습)

| 지표 | 1차 (±36° 증강) | 2차 (±90° 증강 + 이동) |
|---|---|---|
| test_original (초록 배경) | 0.839 | **0.924** |
| test_synthetic (실사 배경) | 0.936 | 0.918 |

- 1차 모델은 웹캠 실전에서 보자기→가위 오인식이 심했음. 원인: 학습 이미지는 손이
  전부 가로 방향인데 웹캠은 손가락이 위를 향함(방향 갭) + 정사각 ROI의 비율 불일치.
- 2차: 회전 증강 ±90° + RandomTranslation 추가로 방향/위치 강건성 확보.
  test 수치가 소폭 내려간 것은 증강이 세져 과제가 어려워진 영향이며, test 셋 자체가
  가로 손 이미지뿐이라 방향 강건성 개선은 test 수치에 반영되지 않음 — 실전 웹캠이 진짜 지표.
- 상위 블록 unfreeze fine-tuning(lr=1e-5)은 val_loss 발산으로 조기종료 —
  체크포인트 퇴행 방지 장치로 모델 파일은 1단계 최고 성능(val_loss 0.180) 버전 유지.
- 잔여 혼동은 여전히 paper→scissors 방향 (계획서 예상 패턴).
- 상세: `outputs/evaluation_report.txt`, `outputs/metrics.json`, `outputs/training_curves.png`

</details>

## 남은 일 (사람 필요)

- [ ] 자체 웹캠 검증셋 30~50장 촬영 및 실전 테스트 — 실전 일반화 성능의 유일한 신뢰 지표
- [ ] 실전 성능 미달 시 상위 블록 unfreeze fine-tuning (lr=1e-5)
- [ ] Unsplash 배경 나머지 50장 보강 (API 시간당 요청 제한)

## 데이터 출처

Kaggle `rps-cv-images` — Julien de la Bruère-Terreault, CC-BY-SA 4.0
(원본 저장소: https://github.com/DrGFreeman/rps-cv)
