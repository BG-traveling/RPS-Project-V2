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

## 남은 일 (사람 필요)

- [ ] 자체 웹캠 검증셋 30~50장 촬영 및 실전 테스트 — 실전 일반화 성능의 유일한 신뢰 지표
- [ ] 실전 성능 미달 시 상위 블록 unfreeze fine-tuning (lr=1e-5)
- [ ] Unsplash 배경 나머지 50장 보강 (API 시간당 요청 제한)

## 데이터 출처

Kaggle `rps-cv-images` — Julien de la Bruère-Terreault, CC-BY-SA 4.0
(원본 저장소: https://github.com/DrGFreeman/rps-cv)
