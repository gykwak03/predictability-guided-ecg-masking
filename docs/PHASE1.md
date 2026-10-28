# ECG masking Phase 1

이번 구현은 전처리 검증, 개선 Plain ECG-MAE patch pilot, Ridge/CNN/Small Transformer의 공통 마스크 복원까지다.
난이도별 사전학습, soft sampler, downstream 비교, 72회 원인 탐색 재실행은 포함하지 않는다.
실제 Seraph 접속·학습 제출은 수행하지 않았다. 합성 smoke는 PTB-XL 성능 결과가 아니다.

## 구성과 재사용

```text
experiments/run_phase1.py       plan/dry-run, preflight, prepare, smoke, profile,
                               pilot, estimator-fit, estimator-score, analyze
experiments/phase1/
  config.py                    기본값, JSON, 경로 우선순위
  common.py                    source/data identity와 run ID, 상태
  data.py                      50 Hz 검증, train/validation, 환자 pool, fit 통계
  masks.py                     lead-major patch, per-lead mask, bank replay, batch 순서
  model.py                     원문 factor_010 / 기존 추정기 factory
  metrics.py                   기존 NRMSE + hidden MSE, RMS, baseline 진단
  training.py                  FP32 학습, 고정 bank 평가, checkpoint/resume, 자원 기록
  pipeline.py                  MAE pilot / estimator fit / score
  analysis.py                  학습 곡선, 관측 수, map, waveform HTML 보고서
configs/phase1_smoke.json       서버 데이터·CUDA 없이 CPU에서 실행
configs/phase1_seraph.json      확인된 서버 경로와 기본 연구 설정
scripts/submit_phase1.sh        사용자가 직접 실행하는 Slurm 제출기
scripts/phase1_worker.sh        할당 안에서 runner 실행
tests/test_phase1.py            과학적 계약 및 경로·재개 검증
```

- 원문 `archive/20261028_legacy/experiments/mae_cause_model.py`의 `CauseMAE`를 파일 위치로 불러온다. 해당 클래스는 보존된 `Masking_pilot.ipynb`를 읽는다. 이 두 파일을 함께 Git으로 전달해야 한다.
- `patient_pools`, `lead_scale`, `LaggedRidge`, `SmallCNN`, `SmallTransformer`, `normalized_masked_loss`, `reconstruction_metrics`, hidden/observed 변환은 기존 `predictability_estimators.py`를 재사용한다.
- 기존 global mask bank는 per-lead 분포와 다르므로 재사용하지 않는다. 새 bank에 geometry·pair·hash·replay 정보를 추가했다.
- 과거 patch difficulty의 관측별 제곱 오차 집계와 보수 마스크 개념을 참고했다. 과거 percentile/group 함수와 정책은 사용하지 않는다.
- 노트북, 원인 탐색 소스, 기존 checkpoint/CSV/plan은 수정하지 않았다. 보존 snapshot의 57개 파일 SHA-256 검증 결과는 모두 일치했다.

## 전처리와 데이터 경계

입력은 PTB-XL의 500 Hz, 10초, physical mV WFDB다. canonical lead 순서는
`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`이다.

1. `resample_poly(up=1, down=2, axis=1, padtype="line")`로 250 Hz 변환, float32.
2. 4차 Butterworth 0.67–50 Hz SOS + `sosfiltfilt(axis=1)`, float32.
3. ECG·lead마다 전체 2500 samples에서 float64 평균·population std를 계산하고 `(x - mean) / max(std, 1e-6)`를 float32로 저장한다.

출력은 ECG당 `(12,2500)`, float32, 무차원 per-record z 단위다.
마스킹 전 정규화이므로 이후 hidden인 구간도 평균·표준편차에 포함된다.
이는 visible-only 또는 train 통계 정규화가 아니다. 모델에 넣기 직전 hidden 값을 바꾸는 검사와
정규화 전 hidden 값을 바꿔 visible 값도 달라지는 검사를 별도로 통과했다.
Zero-phase filtering 자체도 시간 양쪽을 사용하므로 원본 hidden 구간의 전처리 영향까지 차단했다고 주장하지 않는다.

기존 50 Hz artifact는 원본 WFDB 없이 재사용한다. manifest config, 전처리 순서, lead 순서,
shape/dtype/bytes/count, ECG ID 정렬·유일성, artifact_index, fold, 환자 분리와 train/validation 신호의 유한성·정규화 범위를 검사한다.
읽은 manifest·metadata·waveform 파일의 SHA-256을 identity에 저장한다. 40 Hz artifact나 불일치는 오류다.
새 생성은 `prepare`에 새 `--artifact-dir`를 지정한다. 40 Hz 배열을 재필터링하지 않는다.

과거 schema 1 manifest는 `padtype`, `ddof`, 원본 레코드별 hash를 별도로 기록하지 않았다.
이 항목은 검토한 노트북 규칙에 근거하며, 기존 배열만으로 실제 원본 처리 이력을 다시 입증할 수는 없다.
새 run의 `data.json`에 이 제한과 수식을 남긴다. 원본과 전처리의 수치 일치는 합성 500 Hz WFDB 테스트로 검증했다.

학습/평가 loader는 `signals_train.npy`, `metadata_train.csv`, `signals_validation.npy`, `metadata_validation.csv`, `manifest.json`만 연다.
metadata도 ID·patient·fold·split·index 열만 읽는다. label 배열과 test metadata/파형/label 파일을 열지 않는다.
fold 1–8은 train, 9는 validation, 10은 보존한다. 기존 artifact에 test 파일이 있어도 괜찮다.
새 전처리 CLI도 기본은 train/validation만 생성·검증한다. 프로그램용 `run()`의 기존 전체 split 동작은 호환성을 위해 유지했다.
전체 artifact를 별도로 만들 때만 전처리 CLI의 `--splits train validation test`를 명시한다.

Estimator-fit과 score는 **전체 train 환자를 먼저 80/20으로 나눈 뒤** ECG cap을 적용한다.
fit/score/validation 환자·ECG 중복을 명시적으로 거부한다. MAE pilot은 별도 SSL train subset을 사용한다.
Estimator score 환자가 MAE SSL subset과 겹칠 수는 있지만, 해당 환자는 복원 추정기의 fitting에는 사용하지 않는다.
train 전체 점수화는 후속 patient-wise cross-fitting이 필요하다.

## 모델과 pilot

현재 기본 backbone은 우리 구현의 확대 Plain ECG-MAE `factor_010`이다. 공식 SOTA라는 의미가 아니다.
Encoder 384/12층/6 heads, decoder 256/4층/4 heads, 독립 초기화, encoder·decoder 고정 sin/cos 시간 임베딩,
학습 가능한 lead embedding, 추가 patch 전후 LayerNorm 없음, SEP 없음, 전체 token decoder다.
`all_on`과 구분한다. 공식 ST-MEM checkout은 이 단계의 의존성이 아니다.
원문과 새 factory의 전체 크기 초기 가중치 및 P50 CPU forward는 정확히 일치했다.

기존 `factor_010`의 주 학습은 global 50%였다. 새 per-lead 학습은 새로운 실험이다.
기존 분석 CSV/REPORT와 prefix audit는 참고만 했으며 새 결과나 최적값으로 옮기지 않았다.
과거 prefix audit에는 state digest 불일치가 남아 있으므로 과거 재개가 bitwise 같았다고도 주장하지 않는다.

token index는 `lead * Q + time_patch`, `Q=2500/P`이며 `True=hidden`이다.
token `k`의 lead는 `k//Q`, 시간은 `[k%Q*P, (k%Q+1)*P)` samples, 초는 samples/250다.

| Patch | 초 | lead당 patches | 50% hidden / lead | 전체 tokens / hidden |
|---|---:|---:|---:|---:|
| P25 | 0.1 | 100 | 50 | 1200 / 600 |
| P50 | 0.2 | 50 | 25 | 600 / 300 |
| P125 | 0.5 | 20 | 10 | 240 / 120 |

각 lead에서 같은 개수를 가리고 위치는 독립적으로 무작위 선택한다. Global uniform과 다른 제한된 분포다.
비율이 정수 count를 만들지 못하면 반올림하지 않고 거부한다. 1차 기본은 위 세 patch의 50%다.
60%, 70% sensitivity는 사용자가 선택한 단일 `--patches` 값으로만 실행한다.
Estimator patch 기본값 P50은 실행 예시의 기준값이며, 최적값으로 선정한 결과가 아니다.

MAE AdamW `lr=3e-4`, `betas=(.9,.95)`, `weight_decay=.05`, 100 update warmup 후 cosine,
최저 `3e-5`, gradient clipping 1, FP32, microbatch 4 × accumulation 8 = effective batch 32, 3000 updates다.
MAE 학습 목적함수는 hidden samples의 원래 입력 단위 MSE다.
Estimator 신경망은 같은 AdamW betas, 기본 2000 updates, lr `3e-4`, weight decay `1e-4`, clipping 1, hidden normalized MSE다.
모든 설정은 JSON에 저장되며 smoke에서는 크기·budget을 줄인다.

모든 patch 조건은 같은 subset·ECG 순서·updates를 사용한다. 이것이 동일 FLOPs나 GPU 시간을 뜻하지 않는다.
매 update의 effective batch 데이터·마스크 전체를 만든 뒤 microbatch로 자른다.
데이터·mask RNG는 모델 초기화와 독립적이다. 고정 train/validation bank로 update 0부터 학습 곡선을 남긴다.
선정은 validation 안정성, zero 대비 개선, correlation, prediction/target RMS, 시간·VRAM을 함께 검토한다.
복원 MSE가 낮다는 이유만으로 자동 선택하거나 downstream 표현 품질을 확정하지 않는다.

## 추정기 입력과 오차

세 추정기는 같은 ECG, `mask_id`, hidden target을 사용한다. Hidden waveform 값은 내부에서 0으로 차단하고
availability/observed indicator를 별도로 넣어 실제 visible 0과 구분한다. Label은 입력·target·점수에 사용하지 않는다.

| 모델 | 입력과 문맥 | 기록 |
|---|---|---|
| Ridge | 모든 lead의 visible 값·availability를 지정 lag에서 수집, intercept 포함. 양수 lag도 쓰는 offline 복원 | alpha, lag samples/seconds, 실제 fitting target 수/lead, fitting bank hash |
| CNN | 24채널 = 12 visible + 12 indicator, width 64, dilations 1/2/4/8 | receptive field 97 samples = 0.388초, ±48 samples, parameter 수 |
| Small Transformer | token마다 P visible samples + observed 1개, 학습 lead·고정 sin/cos time | width128/depth4/heads4, 모든 lead-time token 양방향 attention |

Ridge는 시간 offset feature를 쓰는 Ridge 회귀다. 기본 `fixed_seconds`는 P50 기준
`[-.4,-.2,-.004,0,.004,.2,.4]`초 = `[-100,-50,-1,0,1,50,100]` samples를 patch와 무관하게 유지한다.
`legacy_patch_scaled`는 과거 `[-2P,-P,-1,0,1,P,2P]`를 재현한다. 정확한 samples로 변환되지 않는 초 값은 거부한다.
Alpha 기본 `.001`, ECG-mask당 시간 후보 128개를 subsample하고 실제 hidden lead targets만 fit한다.
Alpha 선택을 자동 구현하지 않았으며 변경 후보 비교에는 validation만 사용해야 한다.
CNN/Transformer는 동일 fitting bank와 update별 mask ID·batch 순서를 사용하고 sequence hash를 비교한다.
입력 visible 정보는 같아도 실제 receptive field는 다르며, closed-form Ridge와 신경망의 compute가 같다고 주장하지 않는다.

공통 NRMSE는 `sqrt(mean_hidden(((prediction - target) / s_lead)**2))`다.
`s_lead`는 estimator-fit ECG만으로 계산한 고정 population std다. 모든 모델·validation·score에 같은 분모를 사용한다.
개별 patch나 score/validation std로 다시 나누지 않는다. 분모와 fit ID는 `denominator.json`에 저장한다.

각 bank CSV에는 ECG/patient/split/pool/index, mask ID/seed/repeat, samples/seconds,
requested/realized ratio, policy, lead별 count, SHA-256과 pair/member가 있다. CSV만으로 replay할 수 있다.
기본 50% validation/score는 M과 ~M의 pair를 사용한다. 같은 pair는 독립 표본이 아니다.
Fit bank에는 독립적으로 생성한 지정 반복 수만 사용한다.

mask별 오차는 `mask_metrics.csv`, patch별 오차는 `patch_errors.npz`로 분리한다.
`ecg_maps.npz`에는 ECG별 map·hidden 관측 수를, `aggregate_map.csv`에는 집계 map·관측 수를 저장한다.
관측별 normalized squared error를 평균한 뒤 sqrt한다. 미관측 patch는 NaN/관측 수 0을 유지한다.
낮은 NRMSE가 쉬운 방향이며, percentile이나 Easy/Medium/Hard 정책은 만들지 않는다.
zero baseline보다 나쁜 모델, RMS ratio <.1인 진폭 붕괴 경고, 정의되지 않는 correlation/baseline은 명시적으로 표시한다.
이 .1은 진단용 경고 임계치이며 연구상 검증된 선택 기준이 아니다.

## 출력, 재개, 실패

새 결과 기본값은 `experiments/artifacts/phase1/{stage}_{run-id}/`다. 기존 이름이 있으면 새 실행은 실패한다.
`--run-id`를 생략하면 UTC timestamp + UUID를 만든다. `cache_dir`는 쓰기 가능성을 확인하는 예약 scratch 위치이며,
현재는 외부 50 Hz 배열을 직접 mmap해 사용하므로 중복 waveform cache를 만들 필요가 없다.

- `config.json`, `identity.json`, `data.json`, `ecg_selection.csv`, `denominator.json`
- 조건/추정기별 `banks/`, `checkpoint.pt`, `history.csv`, `updates.csv`, `runtime.json`, `resources.json`
- 평가 pool별 `mask_metrics.csv`, `metrics.json`, `patch_errors.npz`, `ecg_maps.npz`, `aggregate_map.csv`, `examples.npz`
- 최상위 `summary.csv` 또는 `estimator_summary.csv`, `status.json`
- `analysis_<id>/REPORT.html`, 학습 곡선·map·관측 수·파형 이미지. 모든 ECG의 수치는 NPZ에 있고 보고서는 첫 ECG를 미리 본다.

`runtime.json`에 실제 GPU 이름·전체 메모리, Torch/CUDA·패키지 버전, FP32·effective batch를 기록한다.
`updates.csv`의 training_seconds는 batch 준비와 optimizer update까지 포함하고 compute_seconds를 따로 남긴다.
평가·checkpoint·session 시간, parameter 수, peak allocated VRAM을 별도 기록한다.
CPU의 peak VRAM 0은 CUDA 메모리 측정 결과가 아니다. Profile 첫 3 updates는 최종 처리량 보장이 아니다.

Checkpoint는 모델·optimizer·update·Python/NumPy/Torch/CUDA RNG, semantic config,
source/data/selection/mask/분모 identity, 다음 ECG·mask 흐름, history·sequence를 저장한다.
`--resume RUN_DIR` 또는 `--checkpoint .../checkpoint.pt`로 해당 run을 재개한다.
동일 실험을 다른 root로 옮길 수 있도록 경로 자체는 semantic identity에서 제외하지만 내용·설정은 엄격히 비교한다.
Update budget, model, seed, microbatch 등을 바꿔 기존 checkpoint에 덮어쓰는 것은 거부한다.
`--stop-after N`은 **추가 N updates**만 실행하고 partial 상태로 반환한다. 남은 budget과 schedule은 바뀌지 않는다.
SIGTERM/시간 초과 시 마지막 주기 checkpoint부터 재개하므로 checkpoint 주기 이후의 작업은 다시 계산할 수 있다.

Estimator fit 완료는 `fit_completed`, score까지 끝나야 `completed`다. 완료 score는 재실행으로 덮어쓰지 않는다.
한 조건/모델이 실패해도 나머지를 실행하고 `error.json`/제외 사유와 partial 상태·비정상 종료를 남긴다.
기준선보다 나쁜 수치도 보존하며 성공한 성능으로 해석하지 않는다. 설정·검증 실패를 합성 성공으로 대체하지 않는다.
분석은 partial run도 읽고 새 analysis 디렉터리를 만든다.

## 경로 계약

우선순위는 **명시적 CLI > ECG 환경변수 > JSON config > 코드 기본값**이다.
`--raw-dir` / `ECG_RAW_DIR`, `--artifact-dir` / `ECG_ARTIFACT_DIR`,
`--output-root` / `ECG_OUTPUT_ROOT`, `--cache-dir` / `ECG_CACHE_DIR`를 지원한다.
전처리 CLI의 `--data-dir`는 `--raw-dir`의 alias, `--output-dir`는 `--artifact-dir`의 alias다.
Shell 실행 파일은 `ECG_PYTHON`으로 지정한다.

JSON 상대 경로는 **그 JSON 파일의 디렉터리**, CLI·환경변수 상대 경로는 **호출 cwd**가 기준이다.
코드·archive 위치와 기본 output은 `__file__`에서 찾는다. 실행 전에 최종 절대 경로를 출력한다.
기본 raw/artifact는 unset이며 실데이터 모드는 없어진 경로와 필요한 옵션을 구체적으로 표시하고 종료한다.
서버 config의 Linux 절대 경로를 Windows에서 그대로 쓰면 거부한다. 로컬은 smoke config를 사용한다.
원격 경로는 과거 확인 기록이며 현재 존재 여부는 Seraph preflight에서 확인해야 한다.

## Windows PowerShell: 편집 → 검증 → 선택한 파일 전달

로컬 `.venv/Scripts/python.exe`는 삭제된 Microsoft Store Python 3.12 경로를 가리켜 실행되지 않았다.
이번 검증에서는 설치를 바꾸지 않고 **앱의 Python 3.12.14 + 기존 `.venv/Lib/site-packages`**를 사용했다.
아래 경로는 이번 PC에서 확인한 값이다. 다른 PC에서는 실행 가능한 Python과 의존성 위치로 바꾼다.
이 Python/Windows venv를 Seraph로 복사하지 않는다.

```powershell
$repo = 'C:\Users\gykwa\kyunghee2026\capstonedesign2\predictability-guided-ecg-masking'
Set-Location $repo
$env:ECG_PYTHON = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = Join-Path $repo '.venv\Lib\site-packages'
& $env:ECG_PYTHON -c "import sys,torch,numpy,pandas,scipy,wfdb; print(sys.executable); print(torch.__version__,numpy.__version__,pandas.__version__,scipy.__version__,wfdb.__version__)"
& $env:ECG_PYTHON "$repo\experiments\run_phase1.py" plan --config "$repo\configs\phase1_smoke.json"
& $env:ECG_PYTHON "$repo\experiments\run_phase1.py" smoke --config "$repo\configs\phase1_smoke.json"
$phase1Temp = Join-Path $repo ('.pytest_cache\phase1-' + [guid]::NewGuid().ToString('N'))
& $env:ECG_PYTHON -m pytest -q --basetemp $phase1Temp

# 다른 cwd에서 절대 진입점 실행도 가능하다.
Set-Location (Split-Path $repo -Parent)
& $env:ECG_PYTHON "$repo\experiments\run_phase1.py" plan --config "$repo\configs\phase1_smoke.json"
Set-Location $repo
git status --short
git diff -- preprocessing/run_preprocessing.py experiments/README.md
git diff --no-index -- NUL experiments/run_phase1.py
git branch --show-current
git remote -v
```

현재 확인한 branch는 `main`, remote는 `origin=https://github.com/gykwak03/predictability-guided-ecg-masking.git`이다.
자동 commit/push는 하지 않았다. 기존에 archive 이동·정리 변경이 남아 있으므로 먼저 함께 검토해야 한다.
사용자가 보낼 파일을 고른 뒤 다음처럼 **명시한 경로만** stage한다. `git add .`를 사용하지 않는다.

```powershell
git add -- README.md .gitignore experiments/README.md preprocessing/run_preprocessing.py experiments/run_phase1.py experiments/phase1 configs/phase1_smoke.json configs/phase1_seraph.json scripts/submit_phase1.sh scripts/phase1_worker.sh tests/test_phase1.py docs/PHASE1.md docs/PHASE1_VALIDATION.md
# 아래는 기존 정리 변경이지만 import/테스트/보존 파일 전달에 필요하다. 별도 검토 후 선택한다.
git add -- archive/ .gitattributes pytest.ini experiments/io_utils.py experiments/requirements.txt docs/PHASE1_HANDOFF.md
# 원래 위치의 삭제도 archive 이동과 함께 검토한 경우에만 stage한다.
git add -u -- experiments tests
git status --short
git diff --cached --stat
# 사용자가 선택한 변경만 검토한 뒤 직접 실행:
git commit -m "Implement phase 1 ECG masking pipeline"
git push origin main
```

아직 추적되지 않은 archive를 제외하면 새 모델이 필요한 원문을 서버에서 찾지 못할 수 있다.
기존 정리 작업의 삭제·이동과 archive 추가는 묶어서 검토한다. 체크포인트·배열·cache는 전달하지 않는다.
Compact CSV/JSON/Markdown/곡선 이미지는 기존 `.gitignore`의 allowlist를 유지하며, 자동 stage하지 않는다.

## Seraph Bash: 준비 → smoke → profile → pilot → fit/score

아래는 사용자가 서버에서 실행할 명령이다. 이 구현 과정에서는 원격 실행·job 제출을 하지 않았다.
SSH config에서 확인한 alias는 `seraph` (`moana.khu.ac.kr`, user `gykwak03`, port `30080`)다.
로컬 PowerShell에서 접속할 때는 `ssh seraph`를 사용하면 그 설정을 따른다.

```bash
export ECG_REPO=/data/gykwak03/capstone2/predictability-guided-ecg-masking
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_RAW_DIR=/data/gykwak03/capstone2/ptbxl_raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50
export ECG_OUTPUT_ROOT="$ECG_REPO/experiments/artifacts/phase1"
export ECG_CACHE_DIR="$ECG_REPO/experiments/artifacts/phase1/cache"
export ECG_LOG_DIR=/data/gykwak03/capstone2/logs
cd "$ECG_REPO"
squeue -u "$USER"
git status --short
git branch --show-current
git remote -v
# 이 checkout을 읽는 실행 중 job이 없고, 사용자 선택 변경의 commit/push가 끝난 뒤 직접 실행:
git pull --ff-only origin main

test -x "$ECG_PYTHON"
"$ECG_PYTHON" -c 'import sys,torch,numpy,pandas,scipy,matplotlib; print(sys.executable); print(torch.__version__,torch.version.cuda,numpy.__version__,pandas.__version__,scipy.__version__)'
mkdir -p "$ECG_LOG_DIR" "$ECG_OUTPUT_ROOT" "$ECG_CACHE_DIR"
"$ECG_PYTHON" "$ECG_REPO/experiments/run_phase1.py" plan --config "$ECG_REPO/configs/phase1_seraph.json"
"$ECG_PYTHON" "$ECG_REPO/experiments/run_phase1.py" preflight --config "$ECG_REPO/configs/phase1_seraph.json"
```

Preflight는 import, manifest·배열·환자 pool, 출력/cache 쓰기 권한을 검사한다. train/validation 전체 배열을 검증·hash하므로 파일 읽기 시간이 든다.
GPU 연산·전처리는 시작하지 않는다. GPU 실사용 정보는 할당 worker의 `runtime.json`에서 확인한다.
실행 중 job이 소스를 읽는 checkout에는 pull하지 않는다. 긴 실험 중 코드를 바꿔야 하면 별도 checkout을 사용한다.

서버의 기존 개인 환경은 PyTorch 2.5.1+cu121, pandas 3.0.6, NumPy 2.4.6, WFDB 4.3.1 기록이 있다.
현재 설치 상태는 위 import로 확인한다. 기존 환경/base를 downgrade하거나 requirements 전체를 재설치하지 않는다.
새 라이브러리를 추가하지 않았으며, 원본 전처리에만 WFDB가 필요하다. pandas 3에서 실패한 WFDB 4.1.2를 설치하지 않는다.

50 Hz artifact가 있으면 preflight 후 그대로 사용한다. 없을 때만 아래 별도 job을 실행한다.
Artifact가 불일치하면 기존 디렉터리를 지우지 말고 새 경로로 바꾼다. 생성 후에는 같은 경로를 계속 지정한다.

```bash
# 기존 artifact가 없거나 설정이 다른 경우에만: 별도 새 데이터 디렉터리 선택
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50_phase1_new
bash "$ECG_REPO/scripts/submit_phase1.sh" prepare --config "$ECG_REPO/configs/phase1_seraph.json"
# 위 job 완료 뒤 preflight 재실행. labels/test는 기본 생성하지 않는다.
```

기본 partition `batch_ugrad_advisor_x`, account `ugrad_advisor_x`, QOS `ugrad`, `--gres=gpu:normal:1`, 최대24시간이다.
사용자 GPU 총량은 1장이다. 다음 job을 제출하기 전에 앞 job을 확인하거나 scheduler에서 대기시키며 동시에 GPU를 두 장 쓰지 않는다.
GPU 종류/node는 고정하지 않는다. RTX 2080 Ti/A2000 기록이 있어도 속도·메모리를 보장하지 않는다.
`ECG_PARTITION`, `ECG_JOB_HOURS`, `ECG_ACCOUNT`, `ECG_QOS`, `ECG_CPUS`, `ECG_MEMORY`로 조정한다.
Debug partition은 `ECG_PARTITION=debug_ugrad_advisor_x ECG_JOB_HOURS=4`로 지정한다.
제출기는 실제 runner와 같은 경로 우선순위로 출력·cache를 해석하고 로그/출력 디렉터리를 제출 전에 만든다.
Bash LF를 유지하고 argv와 환경변수를 worker에 명시적으로 전달한다.

```bash
TAG=$(date -u +%Y%m%dT%H%M%SZ)

# 합성 CPU smoke도 Slurm worker 안에서 실행하는 예시 (실데이터 성능 아님).
bash "$ECG_REPO/scripts/submit_phase1.sh" smoke --config "$ECG_REPO/configs/phase1_smoke.json" --run-id "$TAG"

# smoke 완료 확인 후, 실제 backbone/데이터로 P25의 자원·속도를 먼저 확인.
bash "$ECG_REPO/scripts/submit_phase1.sh" profile --config "$ECG_REPO/configs/phase1_seraph.json" --patch-samples 25 --run-id "$TAG-p25-profile"

# profile 검토 후 하나씩 실행한다. 기본 grid 전체는 --patches 생략 시 실행된다.
# 3000 budget을 유지하면서 우선 1000 updates만 실행; status는 partial.
bash "$ECG_REPO/scripts/submit_phase1.sh" pilot --config "$ECG_REPO/configs/phase1_seraph.json" --patches 25 --run-id "$TAG-p25" --stop-after 1000
# 위 job 완료 후 같은 설정으로 추가 1000 updates 재개 (남은 budget까지 반복).
bash "$ECG_REPO/scripts/submit_phase1.sh" pilot --config "$ECG_REPO/configs/phase1_seraph.json" --patches 25 --resume "$ECG_OUTPUT_ROOT/pilot_${TAG}-p25" --stop-after 1000
# P50/P125도 같은 seed/subset/budget으로 각각 실행하고, 필요하면 위와 같이 재개한다.
bash "$ECG_REPO/scripts/submit_phase1.sh" pilot --config "$ECG_REPO/configs/phase1_seraph.json" --patches 50 --run-id "$TAG-p50" --stop-after 1000
bash "$ECG_REPO/scripts/submit_phase1.sh" pilot --config "$ECG_REPO/configs/phase1_seraph.json" --patches 125 --run-id "$TAG-p125" --stop-after 1000

# 검토 후 P50을 선택했다고 가정한 sensitivity 예시. 실제 최적값을 뜻하지 않는다.
bash "$ECG_REPO/scripts/submit_phase1.sh" pilot --config "$ECG_REPO/configs/phase1_seraph.json" --patches 50 --patch-samples 50 --mask-ratio 0.6 --run-id "$TAG-p50-r60" --stop-after 1000
# 70%는 --mask-ratio 0.7, 다른 run ID로 명시적으로 실행한다.

# pilot 검토 후 선택한 patch를 명시한다. 예시는 기준 P50이다.
bash "$ECG_REPO/scripts/submit_phase1.sh" estimator-fit --config "$ECG_REPO/configs/phase1_seraph.json" --patch-samples 50 --run-id "$TAG-fit" --stop-after 1000
# fit job 완료 후 남은 신경망 updates 재개. 이미 완료된 Ridge는 다시 fit하지 않는다.
bash "$ECG_REPO/scripts/submit_phase1.sh" estimator-fit --config "$ECG_REPO/configs/phase1_seraph.json" --patch-samples 50 --resume "$ECG_OUTPUT_ROOT/estimators_${TAG}-fit"
# status=fit_completed 확인 후 같은 config로 score한다.
bash "$ECG_REPO/scripts/submit_phase1.sh" estimator-score --config "$ECG_REPO/configs/phase1_seraph.json" --patch-samples 50 --run-dir "$ECG_OUTPUT_ROOT/estimators_${TAG}-fit"
```

위 예시는 각 job의 완료·로그 검토 사이에 실행한다. 여러 seed는 `--seed 42`, `43`, `44`와 별도 run ID로 반복할 수 있다.
Subset/mask seed를 동일하게 두면 공통 고정 평가 bank로 비교한다. Seed 간 통계적 결론이나 최소 seed 수를 자동 판정하지 않는다.
공식 ST-MEM 경로는 필요하지 않다. Estimator 학습에 원본 WFDB, ZIP 추출, 재전처리를 요구하지 않는다.

```bash
squeue -u "$USER"
# 제출기가 출력한 실제 job 번호로 JOB_ID를 지정한다.
JOB_ID=실제_job_번호
tail -f "$ECG_LOG_DIR/phase1-ecg-phase1-pilot-${JOB_ID}.out"
cat "$ECG_OUTPUT_ROOT/pilot_${TAG}-p25/status.json"
cat "$ECG_OUTPUT_ROOT/estimators_${TAG}-fit/status.json"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,MaxRSS,AllocTRES

# 분석은 저장된 결과만 읽으며 GPU/원본 데이터가 필요 없다.
"$ECG_PYTHON" "$ECG_REPO/experiments/run_phase1.py" analyze --run-dir "$ECG_OUTPUT_ROOT/pilot_${TAG}-p25"
"$ECG_PYTHON" "$ECG_REPO/experiments/run_phase1.py" analyze --run-dir "$ECG_OUTPUT_ROOT/estimators_${TAG}-fit"
cat "$ECG_OUTPUT_ROOT/estimators_${TAG}-fit/estimator_summary.csv"
```

## 결과 회수

Git clone/pull은 코드와 **실제로 commit한** compact 결과만 옮긴다. 서버의 untracked 결과·checkpoint·배열·환경은 이동하지 않는다.
분석 HTML/이미지를 포함하는 작은 묶음과 모델/배열은 나눠 전송한다.

서버 Bash에서 필요한 실제 run 이름을 지정한다.

```bash
tar -czf "$ECG_OUTPUT_ROOT/${TAG}-small.tar.gz" \
  --exclude='*.pt' --exclude='*.npz' --exclude='*.npy' --exclude='cache' \
  -C "$ECG_OUTPUT_ROOT" "pilot_${TAG}-p25" "estimators_${TAG}-fit"
# patch/ECG map을 로컬에서 추가 분석하려면 해당 npz를 별도 전송한다.
```

로컬 PowerShell에서는 원격에서 만든 실제 TAG를 사용한다. SSH config의 alias가 host/port를 처리한다.

```powershell
$tag = '서버에서_사용한_TAG'
$dest = Join-Path $repo "experiments\artifacts\phase1\download-$tag"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
scp "seraph:/data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/phase1/$tag-small.tar.gz" $dest
tar -xzf (Join-Path $dest "$tag-small.tar.gz") -C $dest
# 큰 파일은 필요한 것만 별도 회수한다.
scp "seraph:/data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/phase1/estimators_$tag-fit/transformer/checkpoint.pt" $dest
scp "seraph:/data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/phase1/estimators_$tag-fit/transformer/score/ecg_maps.npz" $dest
```

NPZ를 제외한 small 묶음은 서버에서 생성한 HTML을 열어 볼 수 있다. 원래 NPZ를 읽어 보고서를 다시 생성하려면
`examples.npz`, `ecg_maps.npz`, `aggregate_map.csv`, `history.csv`, `status.json` 등을 포함한 원래 디렉터리 구조를 회수한다.
서버 경로를 변경했다면 위 회수 경로도 그 실제 값으로 바꾼다.

아직 모르는 것은 실제 PTB-XL 복원 성능, 수렴·진폭 안정성, GPU별 시간·VRAM, 최적 patch/ratio, downstream 유용성이다.
이 값은 Seraph 실험과 validation 검토 뒤에 기록해야 한다.
