# Plain ECG-MAE 복원 부진의 원인 탐색

기존 `Masking_pilot.ipynb`와 과거 결과는 수정하지 않는다. 실제 노트북의
`PlainECGMAE`를 상속한 별도 구현에서 요소를 바꾼다. 모든 옵션을 끈 모델은
기존 모델과 동일한 가중치·출력을 내는지 테스트한다.

목표는 **어떤 변경이 정해진 데이터·학습량에서 복원 부진을 완화하는지** 밝히는 것이다.
단일 조건에서의 개선을 보편적·유일한 원인으로 단정하지 않는다.

## 통제하는 조건

- P50 = 250 Hz에서 0.2초, 12 lead × 50 time = 600 patches.
- 주 분석은 실제 mask 50% = 300 hidden patches. 모델별 반올림 차이 없음.
- 기본 모델 크기는 확대 Plain MAE(encoder 384/12/6, decoder 256/4/4).
  공식 ST-MEM Small과 폭·깊이를 맞추며 작은 Plain MAE도 용량 대조군으로 포함.
- train folds 1–8에서 고정 ECG 5,000개, validation fold 9. fold 10 파일은 열지 않음.
- 기본 final reconstruction은 과거 pilot과 같이 validation 128 ECG × 3 masks.
  확인 실험 계획을 만들 때 `--final-ecgs 0`으로 전체 validation을 평가할 수 있음.
  평가 통계는 batch별로 누적하여 전체 파형 예측을 메모리에 쌓지 않음.
- 동일 seed에서 학습 ECG 순서, 전체 effective-batch mask, optimizer/LR를 공유.
  mask를 먼저 batch 32개에 대해 생성한 뒤 microbatch 4 × accumulation 8로 나눔.
- AdamW 3e-4, betas=(0.9,0.95), weight decay=0.05, clip=1, FP32.
- 동일 update/ECG 노출 기준 비교. 구조마다 FLOPs·시간은 다르며 같다고 주장하지 않음.
- 매 조건을 **같은 global 및 per-lead validation mask bank**로 평가.
  공식 ST-MEM은 동일 visible count/lead가 필요하므로 per-lead bank만 지원.
  다른 bank의 점수를 섞지 않음.
- 각 조건의 frozen encoder를 같은 초기 가중치의 random encoder와 대조.

## 실험 구성

| 단계 | 비교 | 분리하는 질문 |
|---|---|---|
| controls | legacy_short → independent_short | 초기화만 바꾸면 좋아지는가? |
| controls | independent_short → independent_extend | 앞 1,000 updates의 가중치·LR·데이터·mask가 동일한 상태에서 추가 학습만 하면 좋아지는가? |
| controls | independent_extend → independent_long | 3,000 updates는 같게 두고 cosine horizon만 늘리면 좋아지는가? |
| controls | capacity_small_legacy → legacy_short | 모델 용량만 늘리면 좋아지는가? |
| architecture | LayerNorm × 고정 시간 embedding × lead-wise decoder, 2³=8조건 | 각 요소 및 조합의 효과가 있는가? |
| architecture | sep_only, per_lead_only | SEP와 train mask 정책의 단독 효과는? |
| architecture | all_on 및 5개 all_without_* | 결합 모델에서 어떤 요소를 빼면 개선이 사라지는가? |
| architecture | official_stmem | 남아 있는 공식 구현 전체의 차이가 큰가? 단일 요소 원인 비교가 아님 |
| normalization | per_record / train_lead / train_common / visible_record | 전처리 표현·hidden 통계 의존성이 영향을 주는가? |

`factor_100`은 patch LayerNorm만, `factor_010`은 시간 embedding만,
`factor_001`은 decoder만 바꾼 조건이다. `factor_111`은 이 세 요소를 모두 바꾼다.
`all_on`은 여기에 SEP와 per-lead train mask도 추가한다.

lead-wise decoder는 **동일한 가중치**를 lead마다 적용한다. Encoder는 계속 lead 간
정보를 함께 처리한다. 이 변경은 decoder parameter 수를 12배 늘리는 방식이 아니다.
같은 factor cube 안에서 바뀌지 않은 기존 모듈들의 초기값은 같은 seed로 유지된다.
고정 위치 embedding은 학습형과 같은 크기의 tensor이며 학습 여부와 값만 바뀐다.

### 학습량과 스케줄

- short: 1,000 updates, cosine horizon 1,000.
- extend: 3,000 updates, cosine horizon 1,000. 이후 LR는 3e-5로 고정한다.
- long: 3,000 updates, cosine horizon 3,000.

extend의 앞 1,000 updates가 short와 **모델 상태까지 동일한지** digest로 검사한다.
이 검사가 실패하면 학습량만의 효과로 분석하지 않는다.
LR 계산은 과거 LambdaLR의 update 인덱스와 일치시켰다.

### 과거 40 Hz / 75% 결과 재현

새 기본값은 요청한 0.67–50 Hz 데이터다. 과거 40 Hz 점수와 직접 합쳐 비교하지 않는다.
과거 조건은 별도 plan에서 `--artifact-dir .../ptbxl_250hz --ratio .75
--stages controls --size baseline`으로 지정한다.
P50/75%는 lead마다 정확히 같은 수를 가릴 수 없어 architecture 단계에서 거부한다.
architecture 비교에는 50% 또는 76%처럼 양쪽 정책에서 정확히 같은 비율을 사용한다.
과거 실험과 다른 CUDA/PyTorch 버전·microbatch 분할은 bitwise 재현을 보장하지 않는다.

## 정규화: mV 캐시를 먼저 만드는 이유

기존 Z-score 배열에서 원래 진폭을 추정하지 않는다. 원본 500 Hz WFDB에서 동일한
resample_poly(padtype=line) → Butterworth zero-phase filter를 다시 적용하고,
**정규화 전 mV 파형만** 별도 cache에 저장한다. 모든 선택 ECG의 per-record Z-score가
기존 source artifact와 최대 절대오차 1e-4 이내인지 검증한다.
필터 상한은 plan에 지정된 artifact의 설정을 따르므로 40/50 Hz가 섞이지 않는다.

통계는 선택된 SSL train ECG에서만 fit한다. validation/test 통계는 fit하지 않는다.

| 모드 | 입력/target 정규화 |
|---|---|
| per_record | ECG마다 lead별 평균·표준편차. 기존 방식, hidden 구간도 통계에 포함 |
| train_lead | train에서 고정한 lead별 평균·표준편차 |
| train_common | 중심화 없이 train의 공통 RMS 한 개로 나눔. 동차 lead 관계 보존 |
| visible_record | 각 ECG의 visible samples만으로 lead별 통계 계산. 완전히 hidden인 lead는 train 통계 사용 |

모델은 해당 표현의 masked waveform MSE로 학습한다. **모드 사이의 학습 MSE는 단위가
다르므로 직접 비교하지 않는다.** 예측을 mV로 역변환한 뒤, 모든 조건에 동일한 train
lead 표준편차를 사용한 squared error/NRMSE를 주 지표로 비교한다. mV MSE, RMS 비율,
상관계수, 선형 cross-lead 기준선과 정규화 단위 MSE도 함께 저장한다.
이는 입력과 target 표현을 바꾸는 전처리 전체의 효과이며, 입력 스케일만의 효과라고
해석하지 않는다. per_record 역변환에는 해당 전처리 통계가 사용된다는 것도 고려한다.

가린 값 변경 검사는 모델 입력 차단과 전처리 통계 의존성을 나누어 수행한다.
zero-phase filtering 이후 파형을 변경하는 검사이므로, 원시 hidden sample이 필터를
통해 visible sample에 영향을 주지 않는다고 주장하는 검사는 아니다.

## 실행 순서

GPU가 할당된 환경의 Python을 사용한다. 아래 경로는 기존 서버 기준이며 대여 서버에서는
데이터·저장소 경로를 바꾼다. 준비는 CPU에서도 가능하고, 실제 학습은 CUDA가 필요하다.
의존성은 기존 pilot 환경(torch/numpy/pandas/scipy/wfdb/sklearn/matplotlib/einops)을 재사용한다.
새 환경은 CUDA에 맞는 PyTorch를 별도로 설치하고 `requirements-mae-cause.txt`를 사용한다.
각 학습 프로세스는 GPU 한 장을 사용한다. 두 GPU에서는 서로 겹치지 않는 condition/seed를
나눠 실행할 수 있다. 실제 GPU 메모리·속도는 아직 측정하지 않았으므로 소규모 실행으로 확인한다.
전체 기본 계획은 24 conditions × 3 seeds = 198,000 updates다. 처음부터 전부 실행하는
명령을 권장하지 않으며 아래처럼 조건·seed를 명시해 단계적으로 진행한다.

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking

# 1. 계획만 저장. 이 명령은 학습하지 않는다.
python experiments/run_mae_cause_study.py plan \
  --artifact-dir preprocessing/artifacts/ptbxl_250hz_bp067_50 \
  --output experiments/artifacts/mae_cause/plan.json

# 2. 선택 ECG의 물리 단위 파형 재생·검증. test 파일은 열지 않는다.
python experiments/run_mae_cause_study.py prepare \
  --plan experiments/artifacts/mae_cause/plan.json \
  --raw-dir /data/gykwak03/capstone2/ptb-xl/ptb-xl \
  --cache experiments/artifacts/mae_cause/mv_cache

# 3. 우선 controls만 seed 42에서 screening. 본격적인 다중-seed 결론은 아직 금지.
python -u experiments/run_mae_cause_study.py run \
  --plan experiments/artifacts/mae_cause/plan.json \
  --cache experiments/artifacts/mae_cause/mv_cache \
  --output-root experiments/artifacts/mae_cause/study \
  --conditions capacity_small_legacy legacy_short independent_short independent_extend independent_long \
  --seeds 42 --no-probe

# 4. 구조/전처리 조건은 plan에 적힌 이름을 --conditions에 명시하여 이어서 실행.
#    --conditions와 --seeds를 생략하면 계획 전체를 실행하므로, 비용을 먼저 확인한다.
python -u experiments/run_mae_cause_study.py run \
  --plan experiments/artifacts/mae_cause/plan.json \
  --cache experiments/artifacts/mae_cause/mv_cache \
  --output-root experiments/artifacts/mae_cause/study \
  --conditions factor_100 factor_010 factor_001 factor_110 factor_101 factor_011 factor_111 \
  --seeds 42 --no-probe

# 5. 이미 학습한 encoder로 pooling·입력 조건·probe 수렴을 점검. 사전학습 재실행 없음.
python -u experiments/run_mae_cause_study.py probe \
  --plan experiments/artifacts/mae_cause/plan.json \
  --cache experiments/artifacts/mae_cause/mv_cache \
  --output-root experiments/artifacts/mae_cause/study \
  --conditions independent_long factor_111 --seeds 42 --probe-audit

# 6. 누락·단일 seed는 미확정으로 표시. 3 seeds 이상에서 재확인한다.
python experiments/analyze_mae_cause_study.py \
  --root experiments/artifacts/mae_cause/study \
  --output experiments/artifacts/mae_cause/analysis
```

공식 anchor에는 `--st-mem-repo /data/gykwak03/capstone2/ST-MEM`을 지정할 수 있다.
학습을 중단했다면 같은 명령에 `--resume`을 붙인다. source/plan/cache identity가 같은
자체 checkpoint의 optimizer·RNG·update에서 이어간다. 완료 결과는 덮어쓰지 않는다.
코드를 변경한 뒤에는 새 output root에서 실행해야 한다.

별도 tiny memorization 검사는 `run ... --conditions independent_long --seeds 42
--tiny-overfit --tiny-steps 1000 --no-probe`로 실행한다. ECG 2개·고정 mask·상수 LR이며
일반화 실험과 경로·kind가 다르다. 평가 mask가 training mask와 같은 tiny train 지표는
별도로 기록되는 학습 loss이며 일반 fixed validation 곡선과 구별한다.

## Frozen linear-probe 점검

사전학습 전·후 encoder 가중치를 고정하고 train feature 통계로만 표준화한 뒤
선형층 하나와 BCE를 학습한다. 300→1,500 updates는 같은 head 학습 궤적이다.
최고 validation epoch를 골라 점수를 부풀리지 않고 지정 checkpoint를 모두 기록한다.

`--probe-audit`는 동일 encoder에서 다음도 교차 비교한다.

- token별 norm 후 평균 / 평균 후 norm
- 전체 patch 입력 / 고정 visible patch만 입력

visible-only probe는 다른 평가 과제이므로 full-input의 공식 downstream 점수와 섞지 않는다.
공식 ST-MEM anchor는 official full-input pooling만 사용하며 지원하지 않는 조합을 흉내내지 않는다.
이 점검은 AUROC 평가의 문제를 찾는 것이고, 복원 부진의 원인 증거는 reconstruction에서 찾는다.

## 결과와 판단

각 run에 학습·고정 train/validation 곡선, 모듈별 gradient/실제 update, mV 파형,
환자/ECG/repeat별 복원 지표, classifier 학습 곡선, probe score, code/data/mask digest,
실제 GPU 이름·메모리·실행 시간을 저장한다. checkpoint는 optimizer/RNG까지 포함한다.

분석기는 한 요소만 다른 계획인지, 데이터/cache·code·평가 mask와 학습 stream이
맞는지 먼저 검사한다. 환자의 반복 ECG와 반복 mask를 독립 표본으로 세지 않는다.
같은 환자가 모든 seed에 반복되므로 **seed와 환자 축을 교차 bootstrap**한다.

탐색 기준은 계획 생성 때 고정한다: 계획 seed 완료(최소 3개), 모든 seed에서 개선,
환자 평균 standardized MSE의 상대 감소 ≥10%, 절대 개선의 95% bootstrap CI 하한 >0.02.
이는 작업용 실질 효과 기준이고 보편적 통계 규칙이 아니다. 다중 비교·validation 선택에
대한 확증은 별도의 seed/데이터에서 해야 한다. 보고서는 `consistent candidate`까지만
표시하며 자동으로 유일한 원인을 확정하지 않는다.

2³ factorial interaction과 결합 모델에서 요소를 제거하는 reversal을 함께 본다.
단독 효과와 결합 상태의 효과가 다르면 구조 간 상호작용으로 해석한다.
구조를 모두 바꿔도 공식 anchor와 차이가 크면 남은 attention/block·전체 초기화·정규화
세부 구현 차이가 있으며, 이 도구만으로 그 잔여 원인을 확정할 수 없다.

최종 산출물: `paired_effects.csv`, `paired_patient_effects.csv`, `interactions.json`,
`reconstruction_curves.png`, `REPORT.md`, `analysis_audit.json`.
완료된 probe가 있으면 `probe_seed_results.csv`, `probe_summary.csv`도 생성한다.
완료되지 않은 비교는 `missing scientific runs`, 단일 seed는 exploratory로 남긴다.

## 로컬 실행 검증

```bash
python experiments/run_mae_cause_study.py plan --smoke \
  --output /tmp/mae-smoke-plan.json --seeds 42
python experiments/run_mae_cause_study.py run \
  --plan /tmp/mae-smoke-plan.json --output-root /tmp/mae-smoke \
  --conditions independent_short independent_extend all_on official_stmem \
  --seeds 42 --device cpu --no-probe
python -m pytest tests/test_mae_cause_study.py -q
```

Smoke는 축소 모델과 합성 파형이다. 분석기는 실제 과학적 결과에서 제외한다.
로컬 CPU 검증 성공은 실제 GPU 학습 개선, 소요 시간 또는 원인 규명을 뜻하지 않는다.
