# Within/Cross Reconstruction Gain 사전 검증

2026-11-04. 별도 teacher 없이 Plain MAE 하나로 진행하는 **탐색적 기전 실험**이다.
합성 CPU 검증과 실제 PTB-XL 실험은 구분한다. `status.json`이 실제 데이터에서
completed가 되기 전에는 연구 결과가 나온 것으로 기록하지 않는다.

## 검증할 질문

같은 checkpoint에서 측정한 국소 Within/Cross Reconstruction Gain 차이가,
Within 지원 마스크와 Cross 지원 마스크로 짧게 추가 학습한 뒤의 **공통 검증 과제 개선 차이**와
안정적인 관계를 가지는가? 높은 Gain이 더 좋은 학습을 뜻한다고 가정하지 않는다.

이 실험은 Dynamic Masking 정책을 구현하기 전의 진단이다. Teacher/student 비교,
ST-MEM과의 공정한 backbone 비교, 최종 분류 실험은 포함하지 않는다.

## 고정할 조건

| 항목 | 설정 |
| --- | --- |
| 모델 | 기존 Phase 1 Plain MAE, fixed sin/cos, shared decoder, 약 24.6M parameters |
| 입력 | 동일한 250 Hz, 0.67–50 Hz, per-record/lead Z-normalized artifact |
| Patch / 비율 | P50 = 0.2초, 12×50 = 600 tokens 중 300 hidden |
| Lead별 비율 | **모든 lead에서 정확히 25/50 hidden** |
| 학습 ECG | 5,000 ECG, 모든 seed·분기에서 동일한 ID 집합 |
| Gain 진단 | train 내부 128 ECG, 해당 환자들은 모든 gradient 학습에서 제외 |
| 개선량 평가 | validation fold 9의 128 ECG; test fold 10은 열지 않음 |
| Seeds / checkpoints | 42,43,44 × 500,1500,3000 optimizer updates |
| 공통 Random 사전학습 | seed마다 한 번 3,000 updates, 같은 cosine horizon에서 중간 checkpoint 저장 |
| 추가 학습 | checkpoint마다 Within / Cross / Random 각각 200 updates |
| 추가 학습 LR / optimizer | 모든 분기 1e-4 고정, 동일한 새 AdamW 상태에서 시작 |
| Effective batch | microbatch 4 × accumulation 8 = 32, FP32 |
| 총 update 수 | 사전학습 9,000 + 추가학습 5,400 = 14,400 |

1000/3000 등의 숫자는 epoch나 모델 depth가 아니라 **optimizer update 수**다.
진단 환자 제외와 checkpoint별 조건을 통제하기 위해 새 Random trajectory를 만든다.
이전에 받은 결과 폴더에 checkpoint가 없거나 환자 분리가 맞지 않으면 그대로 재사용하지 않는다.
과거 결과·데이터를 덮어쓰지 않고 별도 output에 저장한다.

## 동일 목표 patch, 네 관측 조건

각 ECG의 목표 `(lead,time)` patch 하나는 항상 hidden이다. 목표 시간은 2–47
(0-based)에서 추출해 양쪽 이웃을 확보한다. 첫/마지막 두 time patch는 목표에서 제외된다.

- W: **목표 lead에서 바로 앞 2개·뒤 2개 patch**의 관측 여부.
  양쪽 각각 0.4초의 국소 문맥이다. 목표 lead의 나머지 시간까지 모두 제거하는 것은 아니다.
- C: **나머지 11 lead에서 목표와 같은 시간 patch**의 관측 여부.
  다른 시점의 cross-lead 문맥은 남아 있다.
- 모든 조건에 같은 난수 우선순위를 주고, 위 제약을 적용한 뒤 같은 lead의 나머지 위치에서
  hidden 수를 보정한다. 정확한 전체/lead별 50%를 유지한다.

| 조건 | 국소 Within 정보 W | 동시점 Cross 정보 C | 용도 |
| --- | --- | --- | --- |
| w0_c0 | 제한 | 제한 | Gain 측정 |
| w1_c0 | 제공 | 제한 | **Within 지원 분기** |
| w0_c1 | 제한 | 제공 | **Cross 지원 분기** |
| w1_c1 | 제공 | 제공 | Gain 측정 |
| random | 목표만 hidden 고정, 나머지 random | 동일 | Random 대조군 |

"Within 중심"은 **Within 정보를 이용하도록 지원하는 과제**라는 뜻이다.
Within 정보를 많이 가려 어렵게 한다는 뜻으로 해석하지 않는다.
Random도 동일 목표를 반드시 숨기는 조건부 random이며 일반 무조건 random과 구분한다.

총 visible 수를 고정하려면 다른 patch를 교환해야 한다. 따라서 값은 순수한 정보원 제거 효과나
생리학적 인과효과가 아닌 **이 마스크 교환 규칙에 대한 모델의 반응**이다.
ECG별 4개 독립 anchor/교환 seed를 사용한다. 이는 몇 개의 고정 문제만 본 우연을 줄이는 설정이며,
교환 규칙·시간 이웃 크기에 대한 민감도 실험을 대신하지 않는다.

## Gain 정의와 중요한 수학적 한계

Lwc는 같은 목표 patch에서의 standardized MSE다. 분모는 gradient-training ECG에서만 구한
고정 lead std다. NRMSE 차이가 아닌 **MSE 차이**로 Gain을 계산한다.

```
G_within = 0.5 * [(L00 - L10) + (L01 - L11)]
G_cross  = 0.5 * [(L00 - L01) + (L10 - L11)]
Interaction = L10 + L01 - L00 - L11
```

음의 Gain도 보존한다. 관측 정보 추가가 이 모델/마스크 교환에서 항상 오차를 낮춘다는 보장은 없다.
이 값은 Shannon mutual information, entropy 또는 임상적 정보량 추정치가 아니다.

**G_within − G_cross = L01 − L10**이다. 즉 두 Gain의 차이는 두 지원 조건의 현재 loss 차이와
수학적으로 같다. 이 관계가 재현되더라도 "단순 loss와 독립적인 새로운 신호"를 발견했다고
주장할 수 없다. 이후 정책 연구에서는 단순 high-loss 선택과 반드시 비교해야 한다.

## 추가 학습과 평가

주 실험은 `--fork-loss anchor`다. Within/Cross/Random에서 **동일 목표 patch만 supervised loss에
포함**한다. 전체 hidden patch를 모두 loss에 넣으면 조건마다 정답 patch 집합이 바뀌므로,
먼저 같은 정답으로 문맥에 따른 학습 효과를 비교한다. 300 hidden 중 나머지 patch도 입력에서는
제거하지만, 이 분기의 학습 target에는 넣지 않는다. 이는 표준 MAE 사전학습과 다른 기전 검증이다.

세 분기는 같은 시작 가중치, 동일한 새 optimizer, dropout RNG, ECG 순서, 목표 patch 순서와
마스크 생성 우선순위를 공유한다. 매 update에서 새 목표와 마스크를 생성한다.
시간에 따라 목표를 고정한 문제집으로 추가 학습하는 방식은 아니다.

Gain은 held-out train 진단 panel에서만 계산한다. 추가 학습 전후의 비교는 별도 validation
환자의 **동일한 5종 마스크 × ECG당 4 targets**에서 수행한다. 한 분기만 자기 조건에서 평가하지 않는다.

주 결과는 5종 조건을 같은 가중치로 평균한 **목표 patch MSE 감소량**이다.
추가로 조건별 효과와 전체 hidden-patch MSE 감소량을 보관한다. 후자는 숨긴 정답 구성이
다른 조건을 섞으므로 보조 결과로 해석한다.

```
Predictor X = mean_probe(G_within - G_cross)
Outcome   Y = common_validation_improvement(Within) - common_validation_improvement(Cross)
```

`hypothesis_units.csv`에는 seed×checkpoint별 X/Y, 두 Gain, interaction과 Random 대비 효과가 나온다.
9개 행은 독립된 9개 학습 실험이 아니다. 같은 seed의 세 checkpoint는 같은 학습 trajectory다.
`association.json`의 Spearman은 탐색 통계이며 독립 행을 가정한 p-value를 출력하지 않는다.
ECG/anchor 행을 독립 학습 seed로 취급해 유의성을 부풀리지 않는다.
Checkpoint가 늦어지면서 전체 오차 규모가 함께 감소하는 영향을 살펴보기 위해 baseline loss와
baseline으로 나눈 상대 Gain/개선량도 출력한다. 이 보조 분석만으로 교란 요인이 모두 통제되지는 않는다.

방향과 seed별 일관성, Random 대비 실질적 개선 크기를 먼저 본다. 방향을 사후 선택했다면
**새 학습 seed와 새 평가 마스크에서 그 방향을 고정해 재검증**해야 한다. 현재 3개 seed만으로 일반화나
동적 정책의 우월성을 확정하지 않는다. 관계가 흐리면 먼저 측정/추가 학습 budget의 민감도를
검토한 후 정책 신호에서 제외한다. 복원 개선만으로 downstream 분류 향상을 주장하지 않는다.

주 실험이 유망하면 별도 output에서 `--fork-loss all-hidden`으로 표준 MAE 목적함수에 가까운
후속 실험을 한다. 이때는 조건별 supervised target 구성 차이가 다시 생긴다는 한계를 기록한다.

## 실행

로컬 소프트웨어 검사:

```powershell
python experiments/run_dependency_gain.py --smoke --output ../outputs/dependency_gain_smoke
python -m pytest tests/test_dependency_gain.py -q
```

이 명령은 synthetic 작은 모델만 사용하고 Slurm job을 제출하지 않는다.
실제 데이터는 CUDA와 SLURM_JOB_ID가 있어야 실행된다. CPU로 자동 대체하지 않는다.

서버에 코드가 있을 때 **Seraph의 Bash**에서:

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
RUN=/data/gykwak03/capstone2/dependency_gain_$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/submit_dependency_gain.sh \
  --config configs/dependency_gain_seraph.json --output "$RUN"
# 위 출력의 실제 숫자로 설정한다.
JOB=실제_JOB번호
squeue -j "$JOB"
tail -n 80 -F "/data/gykwak03/capstone2/logs/dependency-gain-${JOB}.out" \
              "/data/gykwak03/capstone2/logs/dependency-gain-${JOB}.err"
```

로그 추적의 Ctrl+C는 `tail`만 종료한다. Job 취소는 `scancel "$JOB"`이다.
중단된 실험은 같은 코드·설정·output으로 제출하며 `--resume`을 추가한다.
원래 job이 끝났거나 취소됐는지 확인한 뒤 재제출한다. 두 job이 같은 output에 쓰면 안 된다.
`--resume`에서도 기본 seed/checkpoint/옵션을 바꾸면 거부한다.

실험 종료 후:

```bash
cat "$RUN/status.json"
cat "$RUN/association.json"
cat "$RUN/hypothesis_units.csv"
sacct -j "$JOB" --format=JobID,State,Elapsed,ExitCode
```

`REPORT.md`, `hypothesis_units.csv`, `association.json`, `compute_cost.json`이 우선 해석 대상이다.
체크포인트와 masks.npz는 재현·재개용이다. Default 전체 checkpoint 저장량은 대략 10–15GB다.
GPU 시간은 실제 할당 GPU와 panel 평가 비용에 따라 달라지며 완료 후 training 시간과
Slurm Elapsed를 함께 기록한다. 12시간 normal GPU 1장을 요청하며, 이는 예상 실행시간 보장이 아니다.

Git push/pull 없이 실행하려면 함께 제공되는 로컬 START_SERAPH.ps1을 사용한다.
코드 snapshot을 새 서버 폴더에 업로드하고 job 하나로 3 seeds를 순차 실행한다.
사용자가 직접 SSH 비밀번호를 입력하며 비밀번호를 파일에 저장하지 않는다.

## 구현 위치

- experiments/dependency_gain.py: 마스크, 동일 목표 loss, 분기 학습, Gain/효과 분석, 재개.
- experiments/run_dependency_gain.py: CLI와 read-only `--plan`.
- configs/dependency_gain_seraph.json: 실제 데이터·모델·학습 기본 설정.
- scripts/submit_dependency_gain.sh: Slurm 제출; worker는 할당 노드에서만 실행.
- tests/test_dependency_gain.py: 마스크 제약, 동일 target, Gain 식, 환자 분리, 재개, 합성 end-to-end.

이 구현은 기존 Phase 1 학습 결과나 원본 전처리 artifact를 수정하지 않는다.
