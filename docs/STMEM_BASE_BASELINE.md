# ST-MEM Base 3000-update baseline

기존 ST-MEM Small / Plain MAE 비교와 같은 P50·50% 마스킹·학습량으로 **ST-MEM Base만 처음부터 학습**한다.
기존 Small·Plain 결과를 덮어쓰거나 두 모델을 다시 실행하지 않는다.
GPU 신청용 `profile_stmem_base.py`의 P75·75%·9초 crop·AMP·patch-normalized loss와 다른 실험이다.
그 자원 측정 checkpoint는 이번 baseline의 초기 가중치로 사용하지 않는다.

## 적용 설정

| 항목 | 이번 Base baseline |
| --- | --- |
| 모델 | 공식 ST-MEM Base 구조: encoder 768 width / 12 layers / 12 heads, decoder 256 / 4 / 4; 현재 구성 총 88,529,558 parameters |
| 입력 | 기존 250 Hz·0.67–50 Hz·ECG별/lead별 Z-normalized artifact, 12×2500 |
| Patch / mask | P50, lead당 50 patches 중 정확히 25 hidden, 전체 600 중 300 hidden |
| SSL train | 같은 subset seed 42로 고른 5000 train ECG, label 미사용 |
| 학습량 | **3000 optimizer updates**, effective batch 32 → 96000 ECG 노출 |
| 기본 배치 | **microbatch 1 × accumulation 32**; 기존 Small/Plain은 4×8 |
| 정밀도 | FP32, TF32 off; 메모리가 부족해도 AMP로 자동 변경하지 않음 |
| AdamW / LR | 기존 비교와 동일: lr 3e-4, betas=(0.9,0.95), wd 0.05, clip 1 |
| Schedule | warmup 100 + 3000-update cosine, 최저 3e-5 |
| Loss | 가린 300개 patch 전체의 waveform MSE; patch target normalization 없음 |
| Checkpoint | 0 / 1000 / 3000 updates, 중단 복구용 optimizer/RNG checkpoint는 200회마다 저장 |
| 복원 평가 | 같은 고정 validation 128 ECG·mask, train-only lead std 분모; NRMSE·상관·RMS 등 |
| Linear probe | 기존과 동일한 full unmasked input·두 pooling·1500 updates, 300/1500회 결과 저장 |
| Primary | 3000회 encoder / 1500회 probe / norm_then_mean Macro AUROC |
| Seed | 기본 42 한 번. Base 사전학습 1회이며 Small·Plain은 실행하지 않음 |
| Test | fold 10 파형·metadata·label 미사용 |

공통 조건은 `configs/backbone_matched.json`과 맞췄으며 테스트에서 차이를 검사한다.
microbatch를 나누기 **전에** effective batch 전체 ECG·mask를 만들므로 1×32와 4×8의 사례·마스크 순서가 같다.
다만 누적 합산 순서의 부동소수점 차이와 모델 크기·실행 시간·FLOPs 차이는 남는다.
따라서 같은 update·노출량의 비교이며 동일 계산량이나 ST-MEM 논문 전체 재현을 주장하지 않는다.

## Seraph 실행

Windows PowerShell:

```powershell
ssh seraph
```

이후 Seraph Bash:

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
git pull --ff-only origin main

export ECG_REPO="$PWD"
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50
export ECG_ST_MEM_REPO=/data/gykwak03/capstone2/ST-MEM
export ECG_LOG_DIR=/data/gykwak03/capstone2/logs
export ECG_JOB_HOURS=24

STUDY="/data/gykwak03/capstone2/results/stmem_base_$(date -u +%Y%m%dT%H%M%SZ)"

"$ECG_PYTHON" experiments/run_stmem_base_baseline.py plan \
  --config configs/stmem_base_baseline.json \
  --study-dir "$STUDY" \
  --artifact-dir "$ECG_ARTIFACT_DIR" \
  --st-mem-repo "$ECG_ST_MEM_REPO"

JOB=$(bash scripts/submit_stmem_base_baseline.sh "$STUDY") && {
  JOB=${JOB##*$'\n'}
  JOB=${JOB%%;*}
  printf 'JOB=%s\nSTUDY=%s\n' "$JOB" "$STUDY"
  squeue -j "$JOB"
}
```

`plan`은 Torch를 import하거나 파형을 읽지 않는다. 실제 학습·평가는 Slurm GPU job에서만 허용한다.
기본 GPU 1장·CPU 4·RAM 32G·24시간·array 동시 실행 `%1`이다.
`ECG_ACCOUNT`, `ECG_QOS`, `ECG_PARTITION`은 기존과 같은 ugrad_advisor_x / ugrad / batch_ugrad_advisor_x가 기본값이다.
리소스 상향 승인이나 특정 GPU 모델을 가정하지 않으며 노드를 고정하지 않는다.

```bash
tail -n 80 -F \
  "/data/gykwak03/capstone2/logs/stmem-base-${JOB}_0.out" \
  "/data/gykwak03/capstone2/logs/stmem-base-${JOB}_0.err"
```

데이터 검증·초기 fixed-bank 평가 후 학습을 시작하며 `mae update=200/3000 ...` 형태로 진행을 출력한다.
사전학습 후 0/1000/3000회 checkpoint의 복원·probe 평가도 수행하므로 3000회 학습 로그 이후에도 job은 계속될 수 있다.
`Ctrl+C`는 로그 보기만 종료하며 job 취소는 `scancel "$JOB"`이다.

## 완료 확인·중단 복구

```bash
cat "$STUDY/seed42/status.json"
cat "$STUDY/seed42/stmem_base/model.json"
cat "$STUDY/seed42/stmem_base/resources.json"

"$ECG_PYTHON" experiments/run_stmem_base_baseline.py analyze --study-dir "$STUDY"
cat "$STUDY/analysis/audit.json"
cat "$STUDY/analysis/reconstruction_summary.csv"
cat "$STUDY/analysis/probe_summary.csv"
```

학습과 지정된 평가가 모두 끝나야 `completed`가 된다. `model.json`의 실제 model_size는 `base`, 폭은 768이어야 한다.
합성 smoke는 `smoke`로 명시되므로 실제 Base 결과와 혼동하지 않는다.

시간 제한·중단이면 **같은 STUDY 경로**로 제출 명령을 다시 실행한다.

```bash
bash scripts/submit_stmem_base_baseline.sh "$STUDY"
```

완료한 Base는 재학습하지 않고, 미완료 학습은 모델·AdamW·RNG·LR 위치를 복구한다.
GPU 기종/메모리·라이브러리·데이터·소스·설정이 달라지면 섞어 이어가지 않는다.
OOM이면 `seed42/stmem_base/error.json`을 확인한다. 기본 microbatch가 이미 1이므로 더 작은 배치를 자동 탐색하거나
기존 plan을 AMP로 바꾸지 않는다. 메모리가 더 큰 GPU에서 시작하려면 새 study를 만든다.

큰 GPU에서 microbatch 4×8 등을 쓰려면 **plan 생성 시에만** `--microbatch 4 --accumulation 8`을 추가한다.
effective batch는 항상 32여야 하며 이미 만든 plan은 수정하지 않는다.
기존 `backbone_comparison.py`, `backbone_models.py`, Phase 1·adapter는 수정하지 않았으므로 기존 Small/Plain 연구의 source hash를 보존한다.

## 기존 결과와 비교할 때

`seed42/stmem_base/evaluations/update_003000/completed.json`의 복원 결과와 primary probe를
기존 matched study의 `seed42/plain` 및 `seed42/stmem`에 대응시킨다.
비교 전 `seed42/identity.json`의 `data_id`, `selection_id`, `banks`, `scale`과 checkpoint별
`sequence_sha256`, probe의 `cohort_sha256`·`head_stream_sha256`을 확인한다.
모델/plan/GPU/physical batch가 달라 전체 `shared_id`는 같지 않다. 같은 값인 것처럼 처리하지 않는다.
이번 analyzer는 Base의 결과만 집계하며 검증하지 않은 타 study 점수를 자동 병합하지 않는다.

## 로컬 코드 검증

```powershell
python -m pytest tests/test_stmem_base_baseline.py
python experiments/run_stmem_base_baseline.py smoke --st-mem-repo ../ST-MEM
```

Smoke는 축소한 공식 클래스·합성 CPU 데이터·2 updates이며 실제 Base 성능 근거가 아니다.
실제 Base 구조 생성, hidden 입력 격리, 공식 pooling과의 수치 일치, microbatch 분할과 무관한 데이터·마스크 순서,
중단 전후 정확한 가중치 복구와 완료 재호출 시 미학습을 검사한다. GPU OOM·처리 속도는 Seraph에서 확인해야 한다.

2026-11-05 로컬 검증: Python 3.12 / Torch 2.14.1+cpu에서 관련 테스트 **8개 통과**.
Slurm 제출 검증은 가짜 `sbatch`를 사용하며 실제 GPU job은 제출하지 않았다.
