# 0.67–50 Hz 데이터와 A2000 12GB 모델 비교

2026-10-24에 추가한 실행 경로입니다. 이번 실험이 선정하는 것은 **우리 데이터·과제·계산 예산에서 사용할 backbone**입니다. 논문별 대규모 사전학습 성능을 새로 재현하거나 보편적인 SOTA를 판정하는 실험은 아닙니다.

## 디렉토리

```text
/data/gykwak03/capstone2/
├── predictability-guided-ecg-masking/  # 우리 실험 코드
│   ├── preprocessing/
│   └── experiments/
├── external/                         # 별도 공식 Git checkout
│   ├── ST-MEM/
│   ├── PatchECG/
│   └── ecg-fm-benchmarking/           # ICLR 2026 ECG-CPC
├── ptbxl_250hz/                      # 이전 40 Hz 데이터: 과거 결과 재현용
├── ptbxl_250hz_bp067_50/              # 새 50 Hz 데이터
└── logs/
/data/gykwak03/envs/ecg-sota/          # 별도 Python 환경
```

실험 코드는 외부 저장소를 import합니다. 외부 코드를 우리 Git 저장소 안에 복사하거나 원본 학습 스크립트의 SLURM 설정을 그대로 사용하지 않습니다. 기존 `/capstone2/ST-MEM`도 이동하지 않습니다.

새 50 Hz 데이터와 실행 파일을 전송한 뒤에는 저장소 루트에서 `bash experiments/start_sota_profile.sh`로 아래 clone → 별도 환경 설치 → 데이터 헤더 확인 → 한 GPU의 순차 profile 제출을 한 번에 진행할 수 있습니다. 이 명령은 profile 20 update만 제출하며 본 pilot을 자동 제출하지 않습니다.

## 공식 코드 받기

Seraph 로그인 터미널에서 우리 저장소 루트로 이동합니다.

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
/data/gykwak03/envs/ecg-mask-pilot/bin/python experiments/setup_sota_repos.py \
  --external-root /data/gykwak03/capstone2/external
```

`sota_sources.json`의 공식 URL과 확인한 commit을 사용합니다. 기존 checkout은 reset/pull하지 않으며, 다른 버전이나 tracked 수정이 있으면 멈추고 경로를 알려줍니다. 사용 버전은 `external/sota_sources_record.json`과 각 실행의 `run_config.json`에 남깁니다.

## 환경

```bash
bash experiments/setup_sota_env.sh
```

기존 작동 중인 `ecg-mask-pilot`의 Python으로 새 venv를 만들고, 별도 환경에 PyTorch 2.5.1 CUDA 12.1과 `requirements-sota.txt`를 설치합니다. 설치에는 GPU 할당이 필요 없습니다. 로그인 노드에서 모델 학습은 실행하지 않습니다. 기본 경로는 `/data/gykwak03/envs/ecg-sota`이며 `SOTA_BASE_PYTHON`, `SOTA_ENV_DIR`로 바꿀 수 있습니다.

세 upstream의 requirements 전체를 한 환경에 섞지 않습니다. PatchECG의 모델 정의, ST-MEM의 모델 정의, ECG-CPC의 encoder/predictor/head/loss 모듈만 불러옵니다. ECG-CPC의 CUDA extension/PyKeOps가 없으면 수학적으로 같은 conjugate-pair Cauchy sum을 PyTorch로 계산합니다. 속도와 메모리는 native 가속 kernel과 달라질 수 있고 실행 설정에 backend를 기록합니다.

## 새 50 Hz 데이터

필터 상한만 이름으로 바꾸거나 기존 40 Hz 배열을 다시 필터링하지 않습니다. 원본 **500 Hz WFDB**에서 resampling → 0.67–50 Hz 4차 zero-phase Butterworth → ECG별 lead별 z-score를 다시 계산합니다. 환자 분할·라벨 규칙은 그대로 유지합니다.

이전 40 Hz 난이도 점수·학습 가중치·파형 그림은 이전 조건의 결과로 보존합니다. 새 50 Hz 실험의 난이도 정책을 검증할 때는 이 데이터에서 추정기를 다시 학습하고 점수를 새로 계산해야 합니다.

로컬:

```bash
python preprocessing/run_preprocessing.py
```

로컬 기본 저장 위치는 `preprocessing/artifacts/ptbxl_250hz_bp067_50`입니다. 원본이 Seraph에도 있으면:

```bash
/data/gykwak03/envs/ecg-sota/bin/python preprocessing/run_preprocessing.py \
  --data-dir /실제/원본/PTBXL/경로 \
  --output-dir /data/gykwak03/capstone2/ptbxl_250hz_bp067_50
```

또는 로컬에서 생성한 새 폴더 전체를 `/data/gykwak03/capstone2/`에 전송합니다. `.building-*` 임시 폴더 대신 검증 후 확정된 폴더를 전송하세요. 기존 경로가 있으면 파일을 덮어 합치지 말고 manifest와 파일 완전성을 먼저 확인합니다. CLI는 설정이 다른 산출물을 덮어쓰지 않습니다.

작은 pipeline 확인은 `--limit-per-split 4`이며 기본 출력이 별도 `_smoke` 경로가 됩니다. 이 파일을 본 실험에 쓰지 않습니다.

## 첫 GPU 검사: 한 대에서 세 모델 순차 실행

```bash
bash experiments/submit_sota.sh all --mode profile --steps 20 \
  --microbatch 4 --accum-steps 8
```

각 모델은 별도 Python 프로세스에서 차례로 실행되며 **동시에 GPU 세 대를 할당하지 않습니다**. profile은 실제 Small 모델을 사용하지만 64 ECG와 짧은 budget으로 동작 검사를 합니다. 성능 순위를 정하지 않습니다.

기본 할당은 사용자 세션에서 확인한 `ugrad_advisor_x` / `debug_ugrad_advisor_x`, GPU 1개, CPU 4개, RAM 16GB, 최대 4시간입니다. `moana-u2` 등 특정 노드를 고정하지 않고 올바른 `experiments/` 절대 경로를 작업 디렉토리로 사용합니다. 작업 한도는 예상 소요 시간이 아닙니다.

```bash
squeue -u "$USER"
tail -n 100 -F /data/gykwak03/capstone2/logs/sota-all_JOB번호.log
```

메모리 부족 시 같은 유효 batch 32를 유지하며 `--microbatch 2 --accum-steps 16`으로 바꿉니다. CPC의 BatchNorm 통계는 microbatch 기준이므로 accumulation이 큰 실제 batch와 완전히 같지는 않습니다. 다른 microbatch를 쓰면 실행 기록과 비교에서 구분하세요.

## 비교할 모델과 조건

| 모델 | 불러오는 실제 모델 | 입력/목적함수 | 정책 backbone 후보 |
|---|---|---|---|
| ST-MEM | 공식 Small, encoder 384/12층/6heads, decoder 256/4층/4heads | 250 Hz, masked-only waveform MSE | 가능 |
| PatchECG | 공식 small_patch_ecg.sh의 ViT: dim128/6층/8heads/MLP512, head_dim64 | 250 Hz, masked-only waveform MSE | 가능 |
| ECG-CPC | 공식 CNN 4층(512) + causal S4 4층, state8, linear prediction head, official CPC loss | 공통 250 Hz로 적응, 미래 14 latent step 예측, negatives128 | CPC 비교군 |

ECG-CPC는 예전 LSTM-CPC가 아닙니다. 이번 코드의 250 Hz는 원 논문의 240 Hz와 다르며, 원래 summed CPC loss는 예측 위치 수로 나눈 평균으로 계산합니다. 가림 비율을 억지로 50%로 추가하지 않습니다. 공개 pretrained weights도 이번 from-scratch 비교에 혼합하지 않습니다.

두 reconstruction 모델에는 **lead당 50개 중 25개 가리기**라는 동일한 마스크와 batch 순서를 제공합니다. 이는 ST-MEM의 visible-count 제약을 충족하는 공통 조건이며, 과거의 global uniform 300-of-600 난이도 실험과 다릅니다. PatchECG 논문의 세 masking type 혼합도 이번 통제 비교에서는 사용하지 않습니다. 따라서 원래 논문 수치와 직접 비교하지 않습니다.

공통 optimizer는 AdamW LR3e-4, betas(.9,.95), weight decay.05, gradient clipping1, warmup5% update + cosine to3e-5입니다. norm/bias decay 제외와 S4 parameter별 optimizer hint를 보존합니다. reconstruction은 FP16 AMP, 복소 연산이 있는 CPC는 FP32입니다. 같은 data/update는 같은 FLOPs나 GPU 시간이 아닙니다.

## Profile 확인 후 pilot

각 모델의 `summary.json`에서 `median_update_seconds`, peak VRAM을 확인합니다. `median_update_seconds × 3000`에 feature 추출/probe 시간을 더한 값이 제출 시간 제한에 들어가는지 먼저 확인하세요. 현재 runner는 final checkpoint를 저장하며 중간 재개 기능은 없으므로, 제한 안에 끝날 budget만 제출합니다.

```bash
bash experiments/submit_sota.sh stmem --mode pilot --steps 3000 --seed 42
bash experiments/submit_sota.sh patchecg --mode pilot --steps 3000 --seed 42
bash experiments/submit_sota.sh ecgcpc --mode pilot --steps 3000 --seed 42
```

위는 개별 제출 예시이며 한 대만 사용할 때는 한 작업이 끝난 다음 다음 작업을 제출합니다. `all --mode pilot`은 합산 시간이 할당 제한 안에 들어가는 경우에만 사용합니다. 3,000 update는 최종 수렴 budget이 아니라 모델 후보를 비교할 pilot입니다. 결과가 접전이거나 한 모델의 성능이 계속 상승 중이면 같은 budget으로 연장합니다.

pilot에서는 원 train folds1–8 중 같은 seed로 고른 5,000 ECG를 pretrain합니다. 매 update마다 유효 batch를 새로 뽑으므로 epoch가 아니라 update budget입니다. 3,000×32=96,000회 ECG 노출, 5,000 ECG 대비19.2회분입니다. 마스크는 매 update 새로 생성합니다.

Linear probe는 이 5,000개 중 유효 라벨이 있는 ECG를 학습하고, fold9의 유효 라벨 ECG 전체에서 평가합니다. Encoder eval/frozen, train-only feature scaling, 한 개 linear layer, BCEWithLogitsLoss, AdamW LR1e-3/weight_decay1e-4, batch128,100 epochs, validation AUROC로 head 선택, F1 threshold.5입니다. 미학습 encoder도 똑같은 probe로 비교합니다. test 파형·라벨을 열지 않습니다.

결과는 `experiments/artifacts/sota_comparison_bp067_50/<새 실행 폴더>/`에 생깁니다. run config, source commit/sha256, ECG 선택, 학습 곡선, 시간/VRAM, initial/final probe와 validation 확률, final checkpoint를 저장합니다. profile/smoke 또는 validation에 5개 클래스가 모두 지원되지 않는 실행은 모델 선정에서 차단됩니다.

```bash
/data/gykwak03/envs/ecg-sota/bin/python experiments/compare_sota_runs.py \
  /각/STMEM/pilot폴더 /각/PatchECG/pilot폴더 /각/ECGCPC/pilot폴더 \
  --output experiments/artifacts/sota_comparison_bp067_50/comparison_seed42
```

서로 다른 데이터 manifest, 학습 ECG, update, 유효 batch, probe, seed 집합 또는 두 reconstruction 모델의 마스크 순서가 섞이면 비교를 거부합니다. seed42 한 번은 잠정 순위이며42/43/44로 반복해 안정성을 확인합니다. CPC는 downstream 비교표에는 포함하지만 임의의 waveform mask 정책을 적용할 backbone으로 자동 선택하지 않습니다. 그 선택은 ST-MEM/PatchECG에서 수행하고, Random encoder보다 학습 이득이 없는 후보를 경계합니다.
