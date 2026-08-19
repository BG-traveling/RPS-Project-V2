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
  04_train_model.py             MobileNetV2 전이학습 + 평가 (confusion matrix 등)
src/webcam_app.py               실시간 웹캠 추론 (ROI, 스무딩 N=7, 임계값 0.75, 쿨다운 2s)
models/rps_mobilenetv2.keras    학습된 모델 (학습↔추론은 이 파일로만 연결)
outputs/                        EDA/합성/학습 그래프, confusion matrix, 평가 리포트
```

## 실행 방법

```bash
# 1) 환경
python -m venv .venv
.venv\Scripts\pip install tensorflow opencv-python scikit-learn matplotlib

# 2) 파이프라인 (순서대로)
.venv\Scripts\python scripts\01_eda.py
.venv\Scripts\python scripts\02_split_data.py
.venv\Scripts\python scripts\03_background_synthesis.py
.venv\Scripts\python scripts\04_train_model.py

# 3) 실시간 웹캠 앱 (q 로 종료)
.venv\Scripts\python src\webcam_app.py
```

## 핵심 설계 결정

- **배경 편향 대응**: 원본이 전부 초록 크로마키 배경이라(EDA에서 G채널 평균 140 vs R 80/B 65)
  배경색 과적합 위험 → 실사 배경 50종으로 크로마키 합성 증강.
- **데이터 리크 방지**: split을 먼저 확정(`splits/*.csv`)하고 각 세트 **안에서만** 합성.
  같은 손 인스턴스가 train/test 에 흩어지는 누수 원천 차단.
- **정규화 일치**: 파이프라인은 0~1(`/255.0`) 유지, 모델 입력단 `Rescaling(2.0, offset=-1.0)`
  레이어로 MobileNetV2가 기대하는 −1~1 로 변환 → 학습/추론 규격 자동 동일.
- **test 이중 평가**: test_original(초록 배경) vs test_synthetic(실사 배경) 정확도를 비교해
  배경 증강 효과를 정량 확인.

## 학습 결과 (2026-08-19, head-only 10 epoch, CPU)

| 지표 | 값 |
|---|---|
| val accuracy | **0.951** |
| test_synthetic (실사 배경) | **0.936** |
| test_original (초록 배경) | 0.839 |

- 학습 곡선: 과적합 없음 (val ≥ train, val_loss ≈ 0.11) → `outputs/training_curves.png`
- test_synthetic 혼동: paper→scissors 10건이 최다 — 계획서에서 예상한 가위/보 혼동 패턴.
  → `outputs/confusion_test_synthetic.png`
- test_original(초록 배경)이 더 낮은 이유: train이 전부 합성 배경이라 초록 단색 배경이
  오히려 분포 밖(OOD)이 됨. rock/paper가 scissors로 쏠리는 오류(25+28건).
  실전 목표는 일반 배경이므로 test_synthetic 수치가 실전에 더 가까운 지표.
  초록 배경 성능도 필요하면 train에 원본 일부를 섞는 것으로 개선 가능.
- 상세 리포트: `outputs/evaluation_report.txt`, `outputs/metrics.json`

## 남은 일 (사람 필요)

- [ ] 자체 웹캠 검증셋 30~50장 촬영 및 실전 테스트 — 실전 일반화 성능의 유일한 신뢰 지표
- [ ] 실전 성능 미달 시 상위 블록 unfreeze fine-tuning (lr=1e-5)
- [ ] Unsplash 배경 나머지 50장 보강 (API 시간당 요청 제한)

## 데이터 출처

Kaggle `rps-cv-images` — Julien de la Bruère-Terreault, CC-BY-SA 4.0
(원본 저장소: https://github.com/DrGFreeman/rps-cv)
