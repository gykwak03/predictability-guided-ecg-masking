# ST-MEM Small ↔ 확대 Plain ECG-MAE 동일 조건 재실험

이번 코드는 과거 서로 다른 실행의 점수를 재사용하지 않고 **두 모델을 처음부터 다시 사전학습**한다.
비교 대상 Plain MAE는 원인 탐색에서 선택한 **factor_010(고정 sin/cos, shared global decoder)**이다.
공식 ST-MEM Small의 patch 입력 LayerNorm·SEP·lead embedding·lead-wise decoding은 유지한다.
ST-MEM의 decoder 가중치는 lead끼리 공유되며, 12개의 독립 decoder를 새로 만드는 것이 아니다.

관련 맥락: [Progress Report - 9주차](https://app.notion.com/p/Progress-Report-9-407f58d59d04800c8957e78de534a816).
재실험 결과는 기존 보고서의 과거 실행과 별도 namespace에 저장한다.

## 1. 왜 다시 실행하는가

기존 `experiments/artifacts/st_mem_grid_v2/.../run_config.json`의 `profile.steps`는 **1,000 optimizer updates**다.
10/28 Plain MAE pilot은 **3,000 optimizer updates**였다. Epoch 1,000/3,000이라는 의미가 아니다.
기존 ST-MEM grid는 여러 patch/ratio를 포함하고 전처리는 40 Hz였으며, probe도 300 updates와
공식 mean-then-norm pooling을 썼다. 서로 다른 실행의 점수 차이를 모델 우열로 해석할 수 없다.
원인 탐색 계획의 `official_stmem`은 별개의 3,000-update 조건이지만, 계획만 있는 것은 완료 결과가 아니다.

이 실험은 **같은 제한된 학습량에서 두 backbone 중 어떤 것이 유리한가**를 확인한다.
논문 전체 데이터·epoch 재현, SOTA 판정, masking 정책 효과를 단독으로 검증하는 실험과 구분한다.

## 2. 공통 설정

| 항목 | 양쪽 모델에 적용하는 값 |
| --- | --- |
| 입력 | PTB-XL train folds 1–8 / validation fold 9, 12 leads × 10초 |
| 전처리 | 원본 500 → 250 Hz, 0.67–50 Hz, ECG별·lead별 전체 10초 Z-normalization |
| SSL cohort | 같은 5,000 train ECG, label 미사용; subset seed 42 고정 |
| Patch / mask | P50 = 0.2초, 600 patches 중 300 hidden; lead마다 정확히 25개 |
| Train mask | update마다 새 마스크, **같은 seed의 두 모델에는 정확히 같은 조합** |
| 학습량 | 각 3,000 optimizer updates; effective batch 32 → 96,000 ECG 노출, 약 19.2 passes |
| 중간 결과 | 동일 3,000-update 학습의 0 / 1,000 / 3,000 checkpoint |
| 배치 / 정밀도 | microbatch 4 × accumulation 8; FP32, TF32 off |
| Optimizer | AdamW, lr 3e-4, betas=(0.9,0.95), weight decay 0.05 |
| LR / clip | warmup 100 updates + cosine(3,000회 종료, 최저 3e-5), clip norm 1 |
| Loss | 동일한 hidden waveform MSE, per-patch target normalization 없음 |
| 복원 평가 | 같은 고정 validation 128 ECG·mask, NRMSE·상관·RMS·zero baseline |
| NRMSE 분모 | 같은 SSL train cohort에서 계산한 lead별 std; validation 통계 미사용 |
| Seeds | 42,43,44 → **총 6회 사전학습** |
| GPU | seed별 한 job 안에서 양쪽 모델을 순차 실행, 단일 GPU |
| Test | fold 10 파형·metadata·label을 열지 않음 |

1,000회 checkpoint는 **같은 3,000회 LR schedule의 중간 관측점**이다.
1,000회에 cosine이 종료되는 독립 단기 실험이 아니다. 두 시점을 각각 공정하게 비교한다.
같은 update·노출량을 맞추므로 모델 간 실제 FLOPs·시간은 다를 수 있다.
Encoder 폭/층/heads는 양쪽 모두 384/12/6, decoder는 256/4/4로 대응한다.
내부 block·norm·SEP·decoder 구조와 초기화 구현은 각 모델의 일부로 남는다.
따라서 결과는 구조 묶음의 비교이며, 특정 한 요소의 인과 효과로 주장하지 않는다.

## 3. Frozen linear probe

- encoder는 eval/no-grad로 고정하고, 전체 unmasked ECG의 patch 특징을 사용한다.
- **Primary: norm_then_mean**, secondary: mean_then_norm. SEP는 양쪽 모두 pooling에서 제외한다.
  ST-MEM의 secondary 결과는 공식 `encoder.forward_encoding` pooling과 수치 일치를 검사한다.
- 같은 SSL train 5,000 ECG 중 진단 라벨이 있는 사례로 head를 학습한다.
  validation은 fold 9의 전체 라벨 유효 ECG를 사용한다(복원 모니터링의 128개와 다름).
- 특징 표준화는 probe train 통계만 사용한다. 같은 seed에서 같은 head 초기화·batch 순서를 쓴다.
- head는 Linear(384,5), BCE, AdamW(lr=1e-3, betas=(0.9,0.999), wd=1e-4), batch ≤256,
  1,500 updates이며 300/1,500회를 저장한다. validation best 선택 없이 정한 시점으로 보고한다.
- Primary endpoint는 **3,000 pretrain updates / 1,500 probe updates / norm_then_mean의 Macro AUROC**.
  Macro AUPRC·F1(threshold 0.5)·클래스별 지표·확률도 저장한다.
- 0회 random encoder의 probe를 함께 측정해 사전학습 전후 이득을 확인한다.
- 다섯 클래스 중 평가 불가능한 클래스가 있으면 Macro AUROC/AUPRC를 비워두고 지원 클래스 수를 기록한다.

## 4. 로컬에서 확인

저장소 루트에서 의존성이 있는 Python으로 실행한다. GPU 학습을 로컬에서 자동 시작하지 않는다.

```powershell
python experiments/run_backbone_comparison.py smoke --st-mem-repo ../ST-MEM
python -m pytest tests/test_backbone_comparison.py
```

Smoke는 축소 모델·합성 ECG·2 updates의 **코드 동작 검증**이며 성능 근거가 아니다.
실제 ST-MEM source checkout은 별도로 필요하다. 이미 있는 저장소를 재사용하며 자동 clone/download하지 않는다.

## 5. Seraph 실행 (로그인 후 저장소 루트)

로컬의 새 코드들을 Git에 반영하여 서버에 전달한 뒤 실행한다. Linux에서 Windows `.venv`를 사용하지 않는다.
장기 계산은 아래 `sbatch` wrapper로 제출한다. `plan`은 파일 설정·해시만 기록하며 학습하지 않는다.

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
git pull

export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50
export ECG_ST_MEM_REPO=/data/gykwak03/capstone2/ST-MEM
export ECG_ACCOUNT=ugrad_advisor_x
export ECG_QOS=ugrad
export ECG_PARTITION=batch_ugrad_advisor_x
export ECG_JOB_HOURS=24

STUDY=/data/gykwak03/capstone2/results/backbone_p50_matched_$(date +%Y%m%d_%H%M%S)
"$ECG_PYTHON" experiments/run_backbone_comparison.py plan \
  --config configs/backbone_matched.json --study-dir "$STUDY"

bash scripts/submit_backbone_comparison.sh "$STUDY"
printf 'STUDY=%s\n' "$STUDY"
```

array task 3개(`0,1,2`)가 seed 42/43/44에 대응하며, `%1`로 동시 실행을 하나로 제한한다.
각 task는 Plain → ST-MEM 순서로 실행한다. 기존 GPU job이 있으면 QOS에 따라 대기한다.
`--nodelist`는 고정하지 않는다. 현재 partition/account 허용 범위는 `show-assoc`·`show-qos`로도 확인한다.

```bash
squeue -u gykwak03
# 예시 제출 번호 123456을 실제 job 번호로 바꾼다.
tail -n 80 -F /data/gykwak03/capstone2/logs/backbone-123456_0.out
sacct -j 123456 --format=JobID,State,Elapsed,ExitCode
```

시간 제한·중단 후 **같은 `$STUDY`를 지정하여 동일한 제출 명령을 재실행**한다.
완료한 arm/probe는 재학습하지 않고, 미완료 arm은 마지막 optimizer/RNG 상태에서 이어간다.
200회마다 checkpoint를 저장하므로 비정상 종료 시 그 이후의 미저장 update만 다시 실행한다.
재시작 때 데이터·코드·마스크·GPU 기종/메모리·라이브러리 환경이 달라지면 혼합하지 않고 거부한다.
다른 환경을 쓰려면 새 study를 만든다. 기존 plan의 updates·LR를 편집해 연장하지 않는다.

## 6. 완료/부분 결과 보기

학습이 끝나거나 중단된 뒤 로그인 노드에서 CSV/그림 집계만 수행한다.
GPU 추론이나 추가 학습을 하지 않으므로 집계에 GPU 할당이 필요 없다.

```bash
"$ECG_PYTHON" experiments/run_backbone_comparison.py analyze --study-dir "$STUDY"
cat "$STUDY/analysis/audit.json"
cat "$STUDY/analysis/reconstruction_summary.csv"
cat "$STUDY/analysis/probe_summary.csv"
cat "$STUDY/analysis/paired_deltas.csv"
```

| 출력 | 확인 내용 |
| --- | --- |
| `seed42/plain/updates.csv` | 매 update loss·LR·학습 시간 (다른 seed/model도 동일 구조) |
| `seed42/plain/history.csv` | 동일 고정 train/validation bank에서의 복원 학습 곡선 |
| `seed42/plain/milestones/update_001000.pt` | 동일 연속 학습의 1,000회 가중치 |
| `seed42/plain/evaluations/update_003000/` | 복원 mask별/patch별 지표·예시 파형 배열·probe 지표와 확률 |
| `analysis/paired_deltas.csv` | 같은 seed/update끼리 ST-MEM − Plain 차이 |
| `analysis/*_summary.csv` | seed 평균·표준편차·실제 관측 수 |
| `analysis/comparison_curves.png` | 같은 학습량에서 복원과 primary probe 곡선 |
| `analysis/audit.json` | 완료/부분 상태, 미완료 arm, synthetic 여부 |

`paired_deltas`에서 NRMSE 차이는 음수일수록 ST-MEM이 낮고, AUROC 차이는 양수일수록 ST-MEM이 높다.
양쪽의 ECG/mask 순서 해시와 probe cohort/head stream이 일치하지 않으면 비교를 중단한다.
복원 오차만으로 최종 backbone을 결정하지 않고 3 seeds의 primary AUROC, random encoder 대비 이득,
실행 시간·VRAM을 함께 검토한다. 일부 seed만 완료되면 partial 결과로 보고한다.

## 7. 로컬 검증 기록 (2026-11-04)

`tests/test_backbone_comparison.py`와 기존 `tests/test_phase1.py`를 함께 실행하여 **33개 통과**하였다.
고정 mask의 hidden 입력 격리, ST-MEM 공식 pooling과의 일치, 연속/중단 학습 가중치의 정확한 일치,
양쪽 모델의 합성 end-to-end 실행, 다른 ECG/mask stream의 비교 거부를 확인하였다.
Slurm 검사는 가짜 `sbatch`로 GPU 1장·array `%1`·공백 경로 전달을 검사했으며 실제 제출은 하지 않았다.
검증 환경은 Python 3.12.14 / Torch 2.14.1+cpu이며, Seraph의 CUDA 학습과 실제 ECG 성능 검증은 별도 실행 대상이다.
