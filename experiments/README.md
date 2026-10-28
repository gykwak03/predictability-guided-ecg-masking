# 현재 작업 코드

새 연구 실행은 **[`run_phase1.py`](run_phase1.py)**와 **[`phase1/`](phase1/)**를 사용합니다.
50 Hz artifact 검증, P25/P50/P125·50% pilot, lead별 동일 개수 마스킹,
fit/score 환자 분리와 공통 bank의 Ridge/CNN/Transformer 복원을 구현했습니다.
자세한 설정, 결과 형식, checkpoint 재개, PowerShell·Seraph 명령은
[Phase 1 실행 안내](../docs/PHASE1.md), 검증 결과는 [로컬 검증 기록](../docs/PHASE1_VALIDATION.md)에 있습니다.

아래 표와 설명은 **보존한 기존 runner**의 인터페이스입니다.

| 파일 | 역할 |
| --- | --- |
| `predictability_estimators.py` | Lagged Ridge, SmallCNN, SmallTransformer, mask bank, 공통 오차·난이도 계산 |
| `run_predictability.py` | 동일 mask bank로 세 추정기를 학습·평가하는 기존 runner |
| `visualize_predictability.py` | 저장된 patch 오차를 실제 ECG 파형 위에 표시 |
| `io_utils.py` | JSON 저장 공통 함수 |
| `requirements.txt` | 새 환경 구성 시 참고할 기존 의존성 버전; CUDA Torch는 별도 설치 |
| `artifacts/` | 과거 결과, 원래 경로 유지 |

**아래 기존 runner는 과거 결과 재현용입니다. 새 Phase 1은 위 진입점을 사용합니다.**
기존 runner에는 40 Hz artifact 기본 경로, P50/P125/P250 선택지, global mask가 남아 있습니다.
`--help`로 인터페이스를 볼 수 있으며, GPU 실험은 [다음 구현 지침](../docs/PHASE1_HANDOFF.md)을 반영한 뒤 진행합니다.
기존 결과를 다시 시각화할 때는 그 결과와 manifest가 일치하는 전처리 artifact를 사용합니다.

기본 MAE 모델은 [과거 모델 구현](../archive/20261028_legacy/experiments/mae_cause_model.py)과
[노트북](../archive/20261028_legacy/experiments/Masking_pilot.ipynb)의 `factor_010` 조건을 참고합니다.
새 MAE runner는 `run_phase1.py pilot`이며, 원문 `CauseMAE` 클래스를 불러와 `factor_010` 설정으로 실행합니다.

Lagged Ridge는 다른 회귀 목적함수가 아니라, **여러 시점의 관측값을 입력에 포함한 Ridge**입니다.
현재 기본 offset은 `[-2P, -P, -1, 0, 1, P, 2P]` samples이며,
P50/250 Hz에서는 `[-0.4, -0.2, -0.004, 0, 0.004, 0.2, 0.4]`초입니다.
12개 lead의 visible 값과 관측 여부만 입력하고, 가린 값은 제거합니다.
양수 offset도 사용하는 오프라인 복원 모델이므로 과거 정보만 쓰는 인과적 예측 모델은 아닙니다.
