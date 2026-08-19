# RPS(가위바위보) 손 제스처 인식

웹캠으로 실시간 가위/바위/보 손 모양을 인식해 액션을 트리거하는 딥러닝 미니 프로젝트.
MobileNetV2(ImageNet) 전이학습 사용. 상세 배경은 [작업계획서.md](작업계획서.md) 참고.

## 프로젝트 구조

> `Data/`, `backgrounds/`, `data_synthetic/`, `outputs/`, `wandb/`, `old/`는 대용량/생성물이라
> `.gitignore` 처리되어 저장소에는 포함되지 않는다. 원본 데이터는 아래 "데이터 출처" 참고,
> 나머지는 `scripts/` 파이프라인을 순서대로 돌리면 로컬에 재생성된다.

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

프레임워크: **PyTorch** (torchvision MobileNetV2, 원래는 Keras로 시작했다가 도중에
전환 — 이유와 과정은 §"문제 해결 및 성능 개선 과정" 참고). 학습 지표는 wandb
`rps-project` 프로젝트에 epoch 단위로 로깅된다 (train/val loss·accuracy, 최종
confusion matrix). 모델 정의는 `src/rps_model.py` 하나에 있고, 학습·추론 모두 이
모듈을 import 한다.

더 상세한 폴더/파일별 설명은 [docs/TECHNICAL_REFERENCE.md](docs/TECHNICAL_REFERENCE.md),
시행착오 전체 서사는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 참고.

## 핵심 설계 결정

- **배경 편향 대응**: 원본이 전부 초록 크로마키 배경이라(EDA에서 G채널 평균 140 vs R 80/B 65)
  배경색 과적합 위험 → 실사 배경 50종으로 크로마키 합성 증강, 이후 손 검출 크롭(B안)으로
  배경 자체를 입력에서 최소화.
- **데이터 리크 방지**: 원본 데이터셋은 split을 먼저 확정(`splits/*.csv`)하고 각 세트
  **안에서만** 배경 합성. 웹캠 데이터는 연속 촬영 세션 단위로 **시간순** 70/30 분할
  (`rps_data.webcam_split`). 같은 손 인스턴스·연속 프레임이 train/val에 함께 섞여
  검증 점수가 부풀려지는 것을 두 경우 모두 원천 차단.
- **정규화는 모델에 내장**: `RPSModel.forward()`가 `register_buffer`로 저장한
  ImageNet 평균/표준편차로 직접 정규화한다. 파이프라인은 0~1 float만 넘기면 되고,
  정규화 상수는 모델 파일(`state_dict`)에 함께 저장되므로 학습·추론 규격이 어긋날 수 없다.
- **test 이중 평가**: test_original(초록 배경) vs test_synthetic(실사 배경) 정확도를 비교해
  배경 증강 효과를 정량 확인. 실전 지표는 별도로 `webcam_val`(직접 촬영분)을 기준으로 삼는다.
- **손 검출 + 분류 2단계(B안)**: CNN에 프레임 전체를 넣으면 배경까지 특징으로 학습해버리고
  여러 손이 들어왔을 때 기준이 없었다. MediaPipe로 손을 먼저 찾아 크롭한 뒤 기존 CNN에
  넣는 구조로 바꿔 배경 의존과 다중 손 문제를 함께 해결.
- **관절각도 게이트**: 손가락이 펴졌는지는 관절 3점의 각도로 직접 계산 가능해, CNN을 거치기
  전에 "학습된 3제스처 중 하나인가"를 기하학적으로 먼저 판정한다(`classify_geometric`).
  회전에 픽셀 분류보다 강인하고, OK사인·엄지척 같은 낯선 모양을 CNN의 강제 3분류 대신
  거부할 수 있게 해준다.

## 문제 해결 및 성능 개선 과정

실전(웹캠) 정확도가 이 프로젝트의 진짜 성적표였다. 아래는 그 수치가 어떻게 움직였고,
각 구간에서 무엇이 문제였는지의 기록이다.

| 시점 | webcam_val(실전) | 무엇이 바뀌었나 |
|---|---|---|
| 배경 합성 직후(Keras, 1차) | — (test_synthetic 93.6%뿐, 실전 미측정) | 실전에서 보자기→가위 오인식 심함 |
| 방향 증강 추가(Keras, 2차) | — | test_original 83.9%→92.4%, 그러나 실전 개선 미미 |
| 웹캠 데이터 fine-tuning 1차 | **0.293** | 도입만으론 부족 — 학습 자체가 실패(암기 아님) |
| unfreeze + BatchNorm 동결 | **0.634** | Keras 마지막 버전 |
| PyTorch+GPU 전환 직후 | 0.455 → **0.675** | 체크포인트 기준 버그 발견·수정 |
| paper 쏠림 수정 + 데이터 보강 | **0.934** | 블러 증강, ROI 정합, 확률 바 UI |
| 각도 특화 촬영(옆면 주먹 등) | **0.945** | 취약 각도 타겟 촬영 900장 |
| B안(손 검출 크롭) 도입 | **0.977**† | 배경 제거 + 다중 손 대상 추적 |

† B안 도입 직후 측정값. 최종 모델 기준 재측정 수치는 아래 "최종 성능" 참고.

### 사건 1 — "test 93.6%인데 실전에서 형편없다" (Keras, 배경 합성 직후)

배경 합성까지 마친 1차 모델은 test_synthetic 93.6%였지만 실전 웹캠에서 보자기를
가위로 계속 잘못 읽었다. **원인**: test 데이터도 결국 원본 Kaggle 데이터셋의 손
모양(가로 방향, 손가락을 모은 보자기)이라, 실전 손 방향·모양에 대한 성능은
애초에 측정조차 되지 않고 있었다. → 지표가 실전과 다른 분포면 지표가 좋을수록
더 위험하다는 것을 이때 배움.

### 사건 2 — fine-tuning 1차: 29.3%, 찍기보다 못한 점수

실전 프레임을 처음 섞어 fine-tuning했을 때 webcam_val이 29.3%(3클래스 무작위=33%)까지
떨어졌다. train 정확도도 함께 낮아서(54%) **과적합이 아니라 학습 자체의 실패**임을
진단 스크립트(`diag_webcam.py`)로 확인했다. 원인: 원본 데이터셋의 "보자기"는 손가락을
모은 손인데 사용자의 실제 보자기는 손가락을 편 손 — "벌어진 손가락"이라는 특징이 합성
데이터 세계에서는 가위의 전유물이었다. 동결된 얕은 head로는 이 충돌을 풀 수 없었다.

### 사건 3 — 상위 블록 unfreeze + BatchNorm 동결 → 63.4%

처방은 표현 용량 확대. 다만 BatchNorm까지 함께 풀면 작은 배치 통계가 하위층 입력
분포를 흔들어 학습이 발산한 전적이 있어, **BatchNorm은 파라미터·이동평균 모두
동결**한 채 상위 3블록만 unfreeze했다. 이 조합이 이후 모든 fine-tuning의 표준이 됨.

### 사건 4 — PyTorch 전환 직후: 체크포인트가 개선을 계속 놓침 (45.5%)

PyTorch+wandb로 옮긴 뒤 GPU로 재학습했는데, 실전 정확도가 51%→61%로 오르는 동안에도
모델 파일이 한 번도 갱신되지 않는 버그가 있었다. 원인: 체크포인트 저장 기준이 **loss**였는데,
분포가 다른(분포 밖) 검증에서는 정확도가 올라도 "확신에 찬 오답" 때문에 loss가 계속
나빠질 수 있다. **저장 기준을 정확도로 바꾸자** 즉시 67.5%로 올라갔다 — 이후 모든
웹캠 fine-tuning은 `monitor="acc"`를 쓴다.

### 사건 5 — "paper만 계속 나온다" (실사용 피드백)

사용자가 rock/scissors를 보여줘도 화면엔 paper만 뜨는 문제가 보고됐다. 스트레스
테스트 스크립트(`diag_live_gap.py`)로 원인을 세 가지로 특정:
1. **모션 블러**에서 모델이 paper로 쏠림(학습 데이터는 대부분 정지 프레임이라 블러를 본 적이 없음)
2. 앱의 ROI(360px)가 학습 크롭(480px)보다 확대된 상태라 손가락이 잘림
3. 블러로 확신도가 떨어진 rock/scissors가 임계값(0.75)에 걸려 "WAITING"으로 가려지고,
   블러에 상대적으로 강했던 paper만 화면에 보임 — 즉 "안 나오는" 게 아니라 "가려진" 것

→ 증강에 GaussianBlur(35% 확률)와 확대 범위를 추가, 앱 ROI를 학습 크롭과 동일한
"최대 중앙 정사각"으로 교체, 화면에 클래스별 확률 바를 추가해 WAITING 상태에서도
내부 확신도가 보이게 함. webcam_val 90.2%→93.4%.

### 사건 6 — "옆면 주먹을 가위로 인식"

오분류 프레임을 직접 조사하니 옆면(너클이 카메라를 향한) 주먹이 학습 데이터에
십수 장뿐이었다. 증강(회전·블러)으로는 만들 수 없는 **3차원 시점 변화**라, 해당
각도를 정조준해 클래스당 300장씩 촬영(`06_capture_webcam.py`)해 보강 →
rock→scissors 오인식이 3.7%→1.4%로 감소, webcam_val 94.5%.

### 사건 7 — 배경·다중 손 문제 → B안(손 검출) 도입

CNN이 배경 픽셀까지 특징으로 학습하는 근본 문제(손 없는 배경만 줘도 특정 클래스를
확신)와, 여러 손이 들어왔을 때 기준이 없는 문제를 동시에 풀기 위해 MediaPipe 손
검출을 앞단에 추가. 학습 데이터도 동일한 방식으로 재크롭(`08_crop_hands.py`)해
학습-추론 분포를 맞췄다. webcam_val 94.5%→97.5%.

### 사건 8 — 관절각도 게이트의 첫 버전이 가위를 자주 오탐 (64%→87%)

미학습 손모양 거부 기능을 만들며 관절 각도로 "펴진 손가락"을 판정했는데, 처음엔
scissors 인정률이 64%에 그쳤다. 손가락별 각도 분포를 직접 뽑아보니 **중지(middle)
판정만 불안정**했고, 원인은 손가락이 카메라를 향해 기울 때 2D 각도만으로는 편
손가락도 굽어 보이는 원근 왜곡이었다. → 관절 각도 계산에 MediaPipe의 상대 깊이(z)를
포함하고, 그래도 불안정한 중지는 판정에서 아예 제외(검지+약지/소지 굽음만으로 판정) →
87.4%로 개선 (`scripts/diag_geometric_gate.py`로 검증).

### 사건 9 — 즉석 학습 기능이 Windows에서 멈춤

실시간 앱에서 키 입력으로 즉석 미세조정을 붙이며 실제 GPU 테스트를 했는데, 실행이
멈추고 고아 `python.exe` 프로세스가 여러 개 남았다. 원인은 이미 CUDA·웹캠·GUI 창이
열려 있는 인터랙티브 프로세스 안에서 `DataLoader`가 멀티프로세스 워커(`num_workers=2`)를
새로 스폰하며 Windows에서 spawn이 교착한 것 — 즉석 학습 경로만 `num_workers=0`으로
바꿔 해결.

## 최종 성능 (현재 모델 기준)

`outputs/evaluation_webcam_ft.txt` 재측정 기준, `models/rps_mobilenetv2.pt` 하나로 고정:

| 평가셋 | Accuracy |
|---|---|
| **webcam_val** (직접 촬영, 실전 지표) | **97.15%** |
| test_original (Kaggle 원본, 초록 배경) | 96.05% |
| test_synthetic (배경 합성) | 98.78% |

## 남은 일 / 다음 개선 방향

- [ ] scissors↔paper 잔여 혼동 보강 — 손끝이 카메라를 향한 각도 위주로 추가 촬영,
  또는 관절각도 게이트를 CNN과 앙상블
- [ ] Unsplash 배경 나머지 50장 보강(선택, API 시간당 요청 제한)

## 데이터 출처

Kaggle `rps-cv-images` — Julien de la Bruère-Terreault, CC-BY-SA 4.0
(원본 저장소: https://github.com/DrGFreeman/rps-cv)
