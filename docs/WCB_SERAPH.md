# WCB 파형 묶음 마스킹 실험 실행

현재 실험 진입점은 `experiments/run_wcb.py`다. 기존 GitHub main과 로컬 변경분은
`archive/20261112_pre_wcb/`에 보관했다. 원본 데이터와 기존 결과는 이동·삭제하지 않았다.

## 고정된 실험 조건

- 250Hz, 12 lead x 10초, 기본 P10=40ms, 전체 3,000 tokens 중 1,500 visible.
- Plain MAE Small: encoder 384/12 layers/6 heads, global decoder 256/4/4,
  FF expansion 4, dropout 0, fixed sin/cos time positions, learned lead positions.
- 기존과 같은 전체 hidden waveform MSE. 추가 loss/teacher 없음.
- 모든 seed/방법은 초기화부터 시작. Random masking warmup 없음.
- `lr_warmup=100`은 기존 학습률 schedule이며 Random 전용 마스킹 구간이 아니다.
- 3,000 committed updates, H=50, microbatch 1 x accumulation 32, FP32.
- W/C/B는 독립 logit 좌표, 초기 중심 .5/.5/.5, delta=.405465, 중심 이동 제한 .05.
- 목표 P/QRS/T는 검출 파형 전체를 덮는 고정 patch 묶음. Purity 필터 없음.
- B는 같은 lead의 다른 박동에서 같은 파형을 덮는 token 합집합. 목표와 겹치는
  다른 B 묶음은 통째로 제외. C는 다른 lead의 동일 시간 token. W는 같은 lead의 나머지.
- 목표는 묶음 전체를 가리며 문맥 공개율은 후보 token에 적용한다. 문맥 파형 전체가
  항상 한꺼번에 visible이 되어야 한다는 제약은 두지 않는다.
- 최대 목표 6개, P/QRS/T 순환 균형 선택과 lead 분산 우선. 공동 할당 불가 시
  6/3/1 순으로 같은 목표 묶음을 축소. 요청 비율은 몰래 보정하지 않는다.
- 복수 목표 할당은 겹침 패턴별 token을 압축해 정수 제약으로 해결한다.
  후보 간 동일 목표·우선순위·rounding 난수, 정확한 전체 50%를 유지한다.
- 문맥 방법은 첫 구간부터 구조적 마스크 80% + Global Random 20%.
- 주 보상은 Random Control Probe LP. Balanced Probe는 별도 평가하며 합치지 않는다.
- Random: Control 환자 128명 x complement masks 2개. Balanced: 그중 최대 32명,
  파형별 목표 하나 x W-rich/C-rich/B-rich/low-context 네 조건, 유효한 경우만 포함.
- Balanced는 기본 5개 정책 구간마다와 최종 상태에서 평가. Random은 각 후보 구간마다 평가.
- `reward=balanced`로 새 계획을 만들면 보상 ablation 가능. 두 평가값은 계속 분리한다.
- 검출 실패 ECG도 학습에 유지. 목표가 없으면 Global Random fallback을 기록한다.
- Frozen probe는 norm_then_mean pooling, train 환자의 labels로 linear head만 학습,
  fold 9 AUROC로 head 선택. Test는 별도 `test` 명령에서만 읽는다.

## 1. 로컬 소스를 Seraph로 전송

완성된 전송 파일은 워크스페이스의 `outputs/wcb_source_20261112.tar.gz`다.
Windows PowerShell에서:

```powershell
scp 'C:\Users\gykwa\kyunghee2026\capstonedesign2\outputs\wcb_source_20261112.tar.gz' seraph:/data/gykwak03/capstone2/
ssh seraph
```

이후 명령은 Seraph Bash에서 실행한다. 새 디렉터리를 사용한다.

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
export ECG_REPO="/data/gykwak03/capstone2/code/wcb_${STAMP}"
mkdir -p "$ECG_REPO"
tar -xzf /data/gykwak03/capstone2/wcb_source_20261112.tar.gz -C "$ECG_REPO"
cd "$ECG_REPO"
```

## 2. Python 환경과 데이터 경로

Python 3.11 이상 권장. 기존 환경을 바꾸지 않으려면 새 환경을 만든다.
CUDA wheel은 Seraph 드라이버에 맞아야 한다. 아래는 CUDA 12.4를 지원하는
환경에서 사용할 수 있는 PyTorch 공식 2.6 설치 예시다.
다른 CUDA 환경이면 [PyTorch 설치 표](https://pytorch.org/get-started/previous-versions/)의 wheel을 사용한다.

```bash
python3 -m venv /data/gykwak03/envs/ecg-wcb
export ECG_PYTHON=/data/gykwak03/envs/ecg-wcb/bin/python
"$ECG_PYTHON" -m pip install --upgrade pip
"$ECG_PYTHON" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$ECG_PYTHON" -m pip install -r configs/wcb_requirements.txt
"$ECG_PYTHON" scripts/package_wcb.py --verify "$ECG_REPO"
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50
export ECG_LOG_DIR=/data/gykwak03/capstone2/logs
```

이미 NumPy 2와 호환되는 CUDA PyTorch 환경이 있으면 ECG_PYTHON만 해당 환경으로 지정해도 된다.
준비에는 `signals_train.npy`, `signals_validation.npy`, 정렬된 metadata와 manifest가 필요하다.
이 단계에서 labels/test는 열지 않는다. 원본 ECG 재전처리는 이 새 파이프라인 범위에 포함하지 않았다.

## 3. 계획 생성

```bash
export ECG_ACCOUNT=ugrad_advisor_x
export ECG_QOS=qos_gykwak03_2026_2
export ECG_PARTITION=batch_ugrad_advisor_x
export ECG_MAX_PARALLEL=2
export ECG_JOB_HOURS=24
STUDY="/data/gykwak03/capstone2/results/wcb40_${STAMP}"
"$ECG_PYTHON" experiments/run_wcb.py plan \
  --config configs/wcb_seraph.json --artifact-dir "$ECG_ARTIFACT_DIR" --study-dir "$STUDY"
cat "$STUDY/tasks.json"
```

계정·QOS·partition은 실제 할당값과 다르면 변경한다. 기본 8방법 x 3seed=24 task이며,
seed별 task 하나가 GPU 한 장을 사용한다. `--methods`와 `--seeds`로 새 계획의 범위를 줄일 수 있다.
계획 생성 후 코드/설정이 바뀌면 재개를 거부한다. 변경 실험은 새 STUDY를 만든다.

## 4. CPU 전처리와 GPU 비용 측정

```bash
PREP_JOB=$(bash scripts/submit_wcb.sh "$STUDY" prepare)
PREP_JOB=${PREP_JOB%%;*}
PROFILE_JOB=$(bash scripts/submit_wcb.sh "$STUDY" profile "$PREP_JOB")
PROFILE_JOB=${PROFILE_JOB%%;*}
echo "PREP=$PREP_JOB PROFILE=$PROFILE_JOB"
squeue -j "$PREP_JOB,$PROFILE_JOB"
```

Prepare는 선택된 train/control ECG 모두에 12-lead 파형 검출을 수행하므로 CPU 시간이 걸린다.
이미 완료된 주석은 다음 제출에서 재사용한다. 같은 STUDY에 중복 제출하면 flock으로 차단한다.
선택 사항으로 기존 audit 경계가 같은 artifact에서 생성됐다면 제출 전에
`export ECG_ANNOTATION_CACHE=/path/to/old_audit_run`을 지정할 수 있다.

Profile은 임시 초기화 모델에서 optimizer 갱신 없이 forward/backward 시간과 peak VRAM을 잰다.
그 모델은 학습에 재사용하지 않는다. 완료 후 다음을 읽는다.

```bash
cat "$STUDY/profile.json"
cat "$STUDY/profile_budget.csv"
```

전체 GPU 시간 추정은 실측 microbatch 시간을 기반으로 하며 CPU MILP, optimizer/저장,
linear probe feature 추출 등을 제외하므로 실행시간 보장이 아니다.
OOM이면 microbatch/patch 크기 등을 바꾼 새 계획을 만든다. 자동 설정 변경은 하지 않는다.

## 5. 본 학습과 재개

Profile과 prepare가 정상 종료된 다음:

```bash
RUN_JOB=$(bash scripts/submit_wcb.sh "$STUDY" run "$PROFILE_JOB")
RUN_JOB=${RUN_JOB%%;*}
echo "RUN=$RUN_JOB STUDY=$STUDY"
squeue -j "$RUN_JOB"
tail -n 50 -F "$ECG_LOG_DIR/wcb-run-${RUN_JOB}_0.out"
```

Task 0은 Random seed42다. 정확한 method/seed 매핑은 tasks.json에 있다.
한 task만 먼저 실행하려면 마지막 인자로 index를 지정한다.

```bash
bash scripts/submit_wcb.sh "$STUDY" run - 0
```

중단 후 같은 코드를 사용해 같은 명령을 다시 제출하면 마지막 H 구간 경계부터 재개한다.
끝난 task는 건너뛴다. timeout 중이던 미완료 분기는 마지막 경계부터 다시 계산하며,
학습·optimizer·정책·난수 상태를 복원한다. 기존 job이 종료됐는지 확인한 뒤 재제출한다.
Task 전체 취소는 `scancel "$RUN_JOB"`, tail의 Ctrl+C는 로그 추적만 종료한다.

## 6. 결과 분석

```bash
"$ECG_PYTHON" experiments/run_wcb.py analyze --study-dir "$STUDY"
cat "$STUDY/analysis/summary.csv"
cat "$STUDY/analysis/paired_vs_random.csv"
sacct -j "$RUN_JOB" --format=JobID,State,Elapsed,ExitCode
```

주요 파일:

- `runs/METHOD/seedN/completed.json`: 두 Probe, validation reconstruction, frozen AUROC/AUPRC, 시간.
- `history.json`: 요청/채택 비율, 실제 달성률, 목표 축소, U 공개량, 별도 Probe 점수.
- `schedule.json`: 실제 채택한 공개 비율.
- `checkpoint.pt`: 해당 방법 자신의 학습 상태. 과거 Random checkpoint로 시작하지 않는다.
- `prepared/annotation_coverage.csv`: 검출 누락과 파형별 목표 수.
- `analysis/by_seed.csv`, `summary.csv`, `paired_vs_random.csv`, `auroc.png`, 학습 곡선 PNG.

3seed 평균/표준편차와 paired 차이를 보고하며, 통계적 유의성을 자동 주장하지 않는다.
후보 겹침 때문에 실제 문맥/Background 교환과 목표 축소가 함께 작용할 수 있으므로 로그를 함께 해석한다.

## 7. 추가 비교

같은 계산 시간 예산 비교는 새 config의 `updates`를 충분히 크게 하고 `budget_seconds`를
모든 방법에 동일하게 설정한다. `lr_horizon`을 명시해 초기 LR 경로를 유지한다.
시간 제한은 H 경계에서 확인하므로 한 구간만큼 초과할 수 있다.
최종 linear probe 및 serialization 비용은 별도로 보고하며, GPU 분기 학습의 원시 시간과
Slurm Elapsed도 함께 확인한다. 매 초 정확히 같은 FLOPs를 쓰는 비교라고 주장하지 않는다.

REINFORCE는 1경로, Coordinate/SPSA/Random-best2는 2경로다. committed updates만 맞추면
계산량이 같지 않다. 동일 예산 study와 profile 표를 함께 보고한다.

Replay는 GPU allocation 안에서 실행한다. 다른 seed에 스케줄을 재생하는 예시:

```bash
"$ECG_PYTHON" experiments/run_wcb.py replay --study-dir "$STUDY" \
  --source-method spsa --source-seed 42 --seed 43
"$ECG_PYTHON" experiments/run_wcb.py replay --study-dir "$STUDY" \
  --source-method spsa --source-seed 42 --seed 43 --shuffle
```

Balanced reward, H=25/100, 단일 목표, Random 혼합 제거, P25=100ms 비교는 config를 복사해
각각 새 계획에서 초기화부터 실행한다. `reward`는 random 또는 balanced이며 합산하지 않는다.
Curriculum은 중심 [.5,.5,.5]에서 출발해 W/C/B 강조 상태를 순환하고 같은 중심으로 돌아오는
사전 고정 piecewise-linear 스케줄이다. 쉬운 과제에서 어려운 과제로 검증됐다는 가정은 하지 않는다.

최종 프로토콜 확정 후에만 Test를 실행한다. 모든 task의 완료된 encoder와 validation 선택 head를 사용한다.

```bash
TEST_JOB=$(bash scripts/submit_wcb.sh "$STUDY" test "$RUN_JOB")
```

## 소프트웨어 검증

```bash
"$ECG_PYTHON" -m pip install pytest
"$ECG_PYTHON" -m pytest tests/test_wcb.py tests/test_patch_audit.py -q
```

합성 CPU smoke는 소프트웨어 검사이며 실제 ECG 성능이 아니다. 이 모드에는 CUDA/Slurm이 필요 없다.
실제 학습/profile/test는 Slurm CUDA allocation에서만 실행되며 로그인 노드 CPU로 자동 대체하지 않는다.
