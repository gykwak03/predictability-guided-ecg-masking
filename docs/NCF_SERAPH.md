# Annotation 없는 NCF 동적 마스킹 실험

NCF는 Near / Cross-lead / Far 순서다. 기존 W는 **근거리 시간 문맥(Near)**,
C는 **다른 lead의 동시점 문맥(Cross-lead)**, B는 **원거리 시간 문맥(Far)**으로 표기한다.
Far는 다른 박동의 같은 파형이라는 의미가 아니다.

이 구현은 첨부된 **목표 400ms, 앞뒤 Near 각각 400ms** 안을 따른다.
이후 논의한 4초 창 확대안은 적용하지 않았다. 파형 경계, R-peak, 주기 추정,
NeuroKit2, 파형 annotation JSON, 목표 축소 fallback, MILP는 실행하지 않는다.

## 1. 고정 조건

- 입력: 기존 250Hz / 0.67–50Hz / ECG·lead별 z-score artifact.
- 모델: 기존 `experiments/wcb/model.py`의 PlainMAE를 그대로 사용한다.
- Loss: 기존 `masked_mse`, 모든 hidden patch의 파형 MSE. 목표 가중 loss 추가 없음.
- 초기화부터 학습하며 pretrained checkpoint 및 Random masking warmup은 없다.
- 학습률 warmup 100 updates는 유지한다.
- 목표: 무작위 6개 lead의 같은 400ms 구간. 앞뒤 각 400ms를 확보할 수 있는 위치에서 선택한다.
- Near: 선택한 lead에서 목표 앞뒤 각 400ms.
- Cross-lead: 나머지 6개 lead의 목표와 동일한 400ms.
- Far: 선택한 lead에서 목표와 Near를 제외한 모든 시간.
- Background: 나머지 lead에서 Cross-lead를 제외한 시간.

| P10=40ms 후보 | token 수 |
| --- | ---: |
| Target (항상 hidden) | 60 |
| Near | 120 |
| Cross-lead | 60 |
| Far | 1,320 |
| Background | 1,440 |

전체 3,000개 중 1,500개를 공개한다. N/C/F 공개율은 각자 0.05–0.95 범위에서
움직이고 합을 1로 정규화하지 않는다. 후보별 stochastic rounding과 공개 우선순위를
짝지은 분기 사이에 공유한다. 잔여 예산만 Background에서 채운다. 요청을 조용히 보정하지 않는다.
설정 검증 시 전체 독립 비율 범위가 반올림을 포함해 실현 가능한지 검사한다.
Background 실현 공개율도 기록한다.

전체 시간·lead가 학습 loss에 포함되지만, 목표 위치 자체는 양 끝 400ms에서 선택하지 않는다.
기존 파형 분석의 purity/여백 결과는 이 실험의 patch 크기 최적성 근거가 아니다.

## 2. 보상과 세 최적화 알고리즘

[CurrMask](https://arxiv.org/html/2410.17744v1#S4)의 다중 마스크 평가와 learning progress를 참고한다.
EXP3 대신 Coordinate-wise / REINFORCE / SPSA를 비교한다.

- 각 방법·seed별로 3,000 committed updates, H=50, FP32, microbatch 1 × accumulation 32.
- 정책은 bounded sigmoid로 logit 3개를 N/C/F 공개율에 대응시킨다.
- Coordinate-wise: 한 logit 좌표를 순환해 ±delta.
- SPSA: Rademacher 방향 3개 좌표에 동시에 ±delta.
- REINFORCE: Normal 정책에서 Gaussian noise로 대칭 후보 2개를 생성한다.
  두 표본의 평균 score gradient와 pair-mean baseline을 사용한다.
- 세 방법 모두 후보 2개를 같은 모델·optimizer·RNG 상태에서 시작한다.
  데이터 순서, 목표, rounding·mask 우선순위를 공유한다.
- 두 후보를 같은 Context-Balanced Control Probe에서 평가하고 더 작은 loss의 분기를 이어 학습한다.
- 보상은 `context_loss_before - context_loss_after`이며 전체 hidden MSE다.
- 이전 보상의 P80−P20를 공통 스케일로 사용한다. 이력이 8개 미만이면 최초 Context Probe loss를 사용한다.
  현재 후보 보상은 스케일 계산 후 이력에 추가한다. 클리핑으로 두 후보 차이를 지우지 않는다.
- logit 기울기를 갱신하고 중심의 실제 공개율 이동을 좌표당 최대 0.05로 제한한다.
- Random 전용 warmup이나 초기 고정 마스킹 구간을 삽입하지 않는다.

## 3. Probe, 데이터 분리, 비교군

train 환자 중 control 환자 128명을 분리한다. 해당 환자의 다른 ECG도 MAE 학습에서 제외한다.
그중 고정 표본 32명에 0.2/0.8의 모든 N/C/F 조합 8개를 적용해 256개 Context Probe 사례를 만든다.
한 ECG에서는 8조건의 목표와 mask 난수를 공유한다. 모든 분기에서 같은 고정 bank를 사용한다.

- **Context-Balanced Control**: 모든 후보 분기에서 측정하며 정책 보상에 사용한다.
- **Random Control**: 환자당 상보 mask 2개, 기본 5개 정책 구간마다와 최종 상태에서 측정한다.
- **Validation Random/Context-Balanced**: fold 9의 별도 환자에 별도 고정 bank를 만들고 최종 보고한다.
- Control/validation 신호를 준비할 때 labels와 test는 열지 않는다.
- Balanced의 파형 종류 균형(P/QRS/T)은 없다. 8개 문맥 조건을 동일 가중 평균한다.
- 각 조건의 **전체 hidden MSE**가 보상이다. `target_loss`는 별도 진단 지표이며 정책 보상에 섞지 않는다.
- Frozen linear probe는 기존 norm-then-mean pooling, train-only 표준화, linear head만 학습한다.
  fold 9 macro-AUROC로 head를 선택하고, test는 명시적인 `test` 명령에서만 읽는다.

| Method | 의미 |
| --- | --- |
| random | Global Random 50% 공개 |
| conditional | 목표 400ms 블록을 항상 가리는 Random |
| fixed | N/C/F 공개율 0.5/0.5/0.5 고정 |
| random_policy | H구간마다 세 공개율을 독립 Uniform(0.05,0.95) 표본화 |
| coordinate | Coordinate-wise 동적 갱신 |
| reinforce | 대칭 표본 2개의 REINFORCE |
| spsa | SPSA 동적 갱신 |
| conditional_best2 | 같은 목표의 Conditional Random 두 분기 중 Probe로 선택 |

8방법 × seed 42/43/44 = 24 task다. 동적 3방법과 conditional_best2는 각 구간에
2배의 학습 update를 시도한다. 동일 committed updates 비교는 동일 계산 예산 비교가 아니다.
실측 profile과 Slurm elapsed time도 함께 보고한다. 중립 fixed는 conditional과 비슷할 수 있다.

## 4. 코드 읽는 순서

1. `configs/ncf_seraph.json`: 실제 설정.
2. `experiments/ncf/common.py`: 조건 검증, 후보 수, 전체 비율 범위의 실현 가능성, source hash.
3. `experiments/ncf/masks.py`: 목표·문맥 집합, 공개율별 마스크 생성.
4. `experiments/ncf/data.py`: 환자 분리와 Probe bank. 장시간 annotation 없음.
5. `experiments/ncf/evaluation.py`: 전체 hidden MSE와 target 진단 MSE의 분리.
6. `experiments/ncf/training.py`: 세 정책, 두 분기, 보상 스케일, checkpoint·resume.
7. `experiments/run_ncf.py`: CLI·profile·분석·test·Replay.
8. `scripts/start_ncf.sh` / `submit_ncf.sh` / `ncf_worker.sh`: sbatch 실행.

기존 MAE와 frozen linear head·학습/RNG helper는 `experiments/wcb/`에서 공유한다.
공유 파일은 새 plan source hash에도 포함된다. 기존 WCB core 파일은 수정하지 않았다.
기존 annotation 기반 study/checkpoint는 NCF에서 사용할 수 없다.

## 5. Git으로 전달

로컬 PowerShell에서 변경 파일을 확인하고 이번 파일들을 commit/push한다.

```powershell
cd C:\Users\gykwa\kyunghee2026\capstonedesign2\predictability-guided-ecg-masking
git status --short
git add experiments/ncf experiments/run_ncf.py configs/ncf_seraph.json configs/ncf_requirements.txt scripts/start_ncf.sh scripts/submit_ncf.sh scripts/ncf_worker.sh tests/test_ncf.py docs/NCF_SERAPH.md README.md
git commit -m "Add annotation-free Near Cross-lead Far masking experiments"
git push origin main
```

아래는 Seraph Bash다. 기존 작업이 사용하는 checkout과 분리된 새 checkout을 만든다.

```bash
mkdir -p /data/gykwak03/capstone2/code
export ECG_REPO="/data/gykwak03/capstone2/code/ncf_$(date -u +%Y%m%dT%H%M%SZ)"
git clone --branch main --single-branch https://github.com/gykwak03/predictability-guided-ecg-masking.git "$ECG_REPO"
cd "$ECG_REPO"
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50
"$ECG_PYTHON" -c 'import torch,numpy,pandas,scipy,sklearn,matplotlib; print("torch",torch.__version__,"CUDA",torch.version.cuda)'
```

기존 환경에 위 의존성이 있으면 재설치하지 않아도 된다. NeuroKit2는 필요하지 않다.
새 환경이라면 드라이버에 맞는 CUDA PyTorch를 먼저 설치하고 `configs/ncf_requirements.txt`를 설치한다.
현재 torch wheel의 실제 GPU 호환성은 profile job에서 확인한다.

## 6. Prepare → GPU profile 제출

```bash
cd "$ECG_REPO"
bash scripts/start_ncf.sh
```

이 명령은 새 NCF plan을 만들고 CPU prepare와 GPU profile까지만 제출한다.
`ECG_ACCOUNT`, `ECG_QOS`, `ECG_PARTITION`을 지정하지 않으면 현재 사용한
`ugrad_advisor_x` / `qos_gykwak03_2026_2` / `batch_ugrad_advisor_x`를 사용한다.
Prepare는 MOANA 제한에 맞춰 GPU 없이 CPU 1개, RAM 2G를 사용한다.
GPU job은 normal GPU 1장, CPU 4개, RAM 32G, 24시간이 기본이다.
`ECG_MAX_PARALLEL`의 기본은 1이며 실제 계정 허용 범위에서 바꿀 수 있다.

출력된 `source .../session.env` 명령을 현재 셸에서도 실행한다. 재접속할 때도 같은 파일을 사용한다.
변수는 기존 WCB의 STUDY/PREP/PROFILE과 구분하여 NCF_STUDY/NCF_PREP/NCF_PROFILE을 사용한다.

```bash
# start_ncf.sh 출력의 실제 경로로 실행
source /data/gykwak03/capstone2/results/ncf40_TIMESTAMP/session.env
squeue -j "$NCF_PREP,$NCF_PROFILE"
sacct -j "$NCF_PREP,$NCF_PROFILE" --format=JobID,State,Elapsed,ExitCode
cat "$NCF_STUDY/prepared/geometry.json"
# Profile이 COMPLETED된 후:
cat "$NCF_STUDY/profile.json"
cat "$NCF_STUDY/profile_budget.csv"
```

`mask_preview.png`에서 Near / Cross-lead / Far 영역과 세 가지 실제 공개 마스크를 확인할 수 있다.
Profile 모델은 임시 초기화 후 폐기하며 optimizer step을 수행하지 않는다.
실행 시간은 모델 연산과 Probe·feature 추출의 추정이며 optimizer/CPU mask/파일 IO 등은 제외한다.
Peak VRAM에는 optimizer 상태가 없고 별도 Adam moment 저장량 추정값을 제공한다.
실측 시간·메모리를 실제 학습의 보장값으로 해석하지 않는다.

## 7. 본 학습, 상태 확인, 재개

Profile 결과 확인 후 전체 array를 제출한다.

```bash
NCF_RUN=$(bash scripts/submit_ncf.sh "$NCF_STUDY" run "$NCF_PROFILE")
NCF_RUN=${NCF_RUN%%;*}
printf 'export NCF_RUN=%q\n' "$NCF_RUN" >> "$NCF_STUDY/session.env"
squeue -j "$NCF_RUN"
tail -F "$ECG_LOG_DIR/ncf-run-${NCF_RUN}_0.out"
```

한 task만 먼저 실행할 수도 있다. Task 0은 random seed42이며 tasks.json이 정확한 매핑이다.
전체 array와 같은 task를 동시에 제출하지 않는다.

```bash
bash scripts/submit_ncf.sh "$NCF_STUDY" run - 0
```

중단된 작업은 종료 상태를 확인한 다음 같은 study·코드로 다시 제출하면 마지막 H 구간부터 재개한다.
모델·optimizer·RNG·정책 중심·정규화 이력을 복원한다. 완료 task는 건너뛴다.
seed가 다른 재학습이나 Replay는 초기화부터 시작하며 기존 run checkpoint를 출발점으로 쓰지 않는다.

## 8. 분석, Replay, test

```bash
"$ECG_PYTHON" experiments/run_ncf.py analyze --study-dir "$NCF_STUDY"
cat "$NCF_STUDY/analysis/summary.csv"
cat "$NCF_STUDY/analysis/paired_vs_random.csv"
```

주요 결과는 by_seed.csv, policy_rates.csv, auroc.png와 방법별 curves.png다.
Control과 validation, Random과 Context-Balanced, 전체 hidden과 target MSE를 분리한다.

완료한 source 정책의 실제 적용 비율을 새 초기화 모델에 재생한다. 아래 Python 작업도 실제 데이터에서는
GPU sbatch job 안에서 실행한다. 기존 task와 다른 폴더를 사용한다.

```bash
"$ECG_PYTHON" experiments/run_ncf.py replay --study-dir "$NCF_STUDY" --source-method spsa --source-seed 42 --seed 43
"$ECG_PYTHON" experiments/run_ncf.py replay --study-dir "$NCF_STUDY" --source-method spsa --source-seed 42 --seed 43 --shuffle
```

test는 연구의 방법 선택을 마친 뒤 명시적으로 제출한다.

```bash
bash scripts/submit_ncf.sh "$NCF_STUDY" test - 0
```

## 9. 로컬 소프트웨어 검증

```bash
python -m pytest tests/test_ncf.py -q
python experiments/run_ncf.py plan --smoke --study-dir /tmp/ncf_smoke
python experiments/run_ncf.py prepare --study-dir /tmp/ncf_smoke
python experiments/run_ncf.py run --study-dir /tmp/ncf_smoke --method spsa --seed 42
```

Synthetic smoke는 실제 ECG 성능이나 GPU 실행 검증이 아니다. 실제 데이터에서는 Slurm/CUDA를 요구하며
자동 CPU fallback을 하지 않는다. 새 NCF 구현은 실제 Seraph 본 학습 전 단계다.
