# 기존 checkpoint를 이용한 Within/Cross 추가 학습 재검증

2026-11-04. 기존 `dependency_gain_results_*`의 **Random 사전학습 checkpoint**를 읽어
추가 학습 절차를 통제한다. 사전학습을 다시 실행하지 않으며 기존 run에는 쓰지 않는다.
이 단계는 동적 마스킹 정책이나 downstream 분류 성능을 검증하는 본 실험에 앞선 대조 실험이다.

## 기본 비교: D / E / F × optimizer 유지·초기화

| Case | 마스크 | 학습 loss 대상 | 목적 |
| --- | --- | --- | --- |
| D | 기존 MAE의 무조건 Random | hidden 300개 전체 | 원래 MAE 학습을 이어가는 대조군 |
| E | Within 지원, 동시점 Cross 제한 | hidden 300개 전체 | 문맥 마스크 변경의 효과 |
| F | Cross 지원, 가까운 Within 제한 | hidden 300개 전체 | 문맥 마스크 변경의 효과 |

각 case는 `keep` / `reset` 두 optimizer 조건을 실행한다.

- `keep`: source checkpoint의 AdamW 누적 상태를 복원한다. 각 분기에는 상태를 깊게 복사해
  이전 분기의 update가 다음 분기의 공통 시작 상태를 변경하지 못하게 한다.
- `reset`: 같은 가중치에서 동일 hyperparameter의 새 AdamW를 사용한다.
- **두 조건 모두 동일한 학습률 일정**을 사용한다. 기본값은 원래 3,000-update cosine schedule의
  다음 update부터 이어간다. 3,000 이후에는 기존 scheduler 규칙대로 min_lr=3e-5를 유지한다.
  warmup이나 cosine horizon을 다시 시작하지 않는다.
- 학습 loss의 lead 분모도 기존 사전학습처럼 ones를 사용한다. 현재 입력은 per-record/lead Z-normalized다.
  검증 지표는 기존 train-only lead scale로 계산한다.
- 같은 seed/checkpoint 내 모든 분기는 같은 가중치·RNG·ECG 순서·update 수·LR로 시작한다.
  ECG 순서는 사전학습의 absolute update 다음부터 이어간다.

기본값은 **seeds 42/43/44, checkpoints 1500/3000, 각 분기 200 updates**다.
총 추가 update는 `3 × 2 × 3 cases × 2 optimizer modes × 200 = 7,200`이다.
source의 checkpoint 구성이 다르면 마지막 두 저장 checkpoint가 기본 선택된다.

## 마스크가 의미하는 것

모두 P50, 전체 600 tokens 중 300 hidden, 각 lead에서 정확히 25/50 hidden이다.

- **D**는 원래 사전학습의 다음 Random mask를 그대로 생성한다. 별도 목표 patch를 hidden으로
  강제하지 않는다. `keep + continue LR`는 원래 사전학습을 이어가는 조건이다.
- **E/F**는 ECG마다 anchor 하나를 선택하고 해당 anchor를 반드시 숨긴다. E에서는 같은 lead의
  앞뒤 각 2 patches를 보여주고, 다른 11 lead의 같은 시간 patch를 숨긴다.
  F에서는 그 반대로 배치한다. 다른 위치의 patch를 교환해 lead별 50%를 유지한다.
- 따라서 E/F는 특정 anchor 주변에서 문맥을 바꾼 **전체 마스크 정책**이다. 나머지 299개 target이
  각각 Within-only/Cross-only가 되는 것은 아니다. 다른 시간의 cross-lead 정보나 먼 within-lead 문맥도 남는다.
- D와 E/F의 비교에는 anchor를 반드시 가리는 조건의 차이도 포함된다. 그 영향을 분리하려면 아래 G를 포함한
  matched loss ablation을 실행한다. E/F의 성능 차이를 특정 정보원의 순수 인과효과로 해석하지 않는다.

## 단일 patch loss와 optimizer의 영향을 나누는 추가 비교

`--losses all-hidden anchor`를 지정하면 다음까지 실행한다.

| Case | 마스크 | loss 대상 |
| --- | --- | --- |
| A | E와 동일한 Within 지원 마스크 | anchor 1개 |
| B | F와 동일한 Cross 지원 마스크 | anchor 1개 |
| C | anchor를 숨긴 조건부 Random | anchor 1개 |
| G | C와 동일한 조건부 Random | hidden 300개 전체 |

E−A, F−B, G−C는 **마스크·데이터·LR·optimizer mode를 맞춘 상태에서 loss 대상의 차이**를 비교한다.
각 case의 keep−reset은 loss·마스크·데이터·LR를 맞춘 optimizer 상태의 효과다.
D−G는 무조건 Random과 anchor-conditioned Random의 차이를 살펴보는 데 쓸 수 있다.
무조건 Random에서 anchor loss를 계산하면 목표가 보일 수 있으므로, D의 anchor 버전은 만들지 않는다.

모든 7 cases × 2 modes를 실행하면 총 16,800 additional updates다.
이 A/B/C는 **새 프로토콜 안의 matched control**이며, 예전 run의 A/B/C와 완전히 같은 재실행은 아니다.
예전 실험은 fixed LR=1e-4, 새 optimizer, 별도 ECG 순서, train-only scale을 사용했다.
따라서 예전 C와 새 D 차이만으로 단일 patch loss가 유일한 원인이라고 결론 내리지 않는다.
새 paired 비교를 우선한다. LR의 영향까지 보려면 다른 output에
`--lr-policy constant --lr 1e-4`를 명시해 동일 suite를 비교한다.

단일 patch loss를 쓰더라도 ECG마다 목표는 새로 추출된다. 고정 patch 하나만 계속 암기시키는 실험은 아니다.
전체 target 평균보다 noisy할 가능성은 있지만, 단일 target 선택 자체가 반드시 성능 저하를 일으킨다고 가정하지 않는다.

## 공통 평가와 진단

**주 지표:** source와 같은 validation ECG, 고정된 **무조건 Random** 마스크 묶음에서
전체 hidden patch의 standardized MSE를 비교한다. 기본 ECG당 2 complement pairs = 4 masks다.
서로 반대인 mask를 쌍으로 써 모든 시간·lead가 같은 횟수만큼 target에 포함되게 한다.

- 추가 학습 전과 50/100/150/200 updates에서 동일 bank로 평가한다.
- `improvement = MSE_before - MSE_after`; **양수는 개선, 음수는 악화**다.
- `validation_nrmse = sqrt(mean standardized hidden MSE)`다.
- 주 표는 ECG-mask 평균이며, `mean_patient_improvement`를 환자 가중치 민감도 확인용으로 함께 저장한다.
- 기존 5종 문맥 panel은 추가 학습 전후의 **보조 진단**으로 보존한다.
  조건별 anchor/전체 hidden patch 개선량을 따로 출력한다. 예전 표의 5조건 anchor 평균과 새 주 지표는 다르다.
- Gain은 같은 held-out TRAIN 환자와 기존 panel 정의에서 다시 계산한다. validation을 마스크 선택 정책 조정에 쓰지 않는다.
- Downstream probe와 test fold 10은 실행하지 않는다. 이 결과만으로 분류 성능 향상이나 임상적 유효성을 주장하지 않는다.

주 검증 전에 no-update replay로 가중치를 다시 읽어 같은 오차가 나오는지 확인한다.
또한 D/keep/continue가 원래 사전학습 범위 안에 있으면 source `updates.csv`와
ECG·mask·LR가 일치하는지 검사하고 training loss 차이를 기록한다.
GPU/소프트웨어가 달라질 수 있으므로 실제 데이터 training loss의 bitwise 일치는 강제하지 않는다.

같은 seed의 checkpoint는 서로 독립인 학습 실험이 아니다. seed별 방향과 효과 크기를 보고,
ECG나 checkpoint 행을 독립 seed로 간주한 pooled p-value는 계산하지 않는다.

## Seraph 실행: 기존 job이나 데이터는 수정하지 않음

서버에서 `git pull --ff-only origin main` 후:

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
export ECG_REPO="$PWD"
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50

SOURCE=/data/gykwak03/capstone2/dependency_gain_results_20261104T101928Z
RUN=/data/gykwak03/capstone2/dependency_continuation_$(date -u +%Y%m%dT%H%M%SZ)

JOB=$(bash scripts/submit_dependency_continuation.sh \
  --source-run "$SOURCE" --output "$RUN" \
  --artifact-dir "$ECG_ARTIFACT_DIR")
JOB=${JOB##*$'\n'}
JOB=${JOB%%;*}
echo "JOB=$JOB"
echo "RUN=$RUN"
squeue -j "$JOB"

tail -n 80 -F "/data/gykwak03/capstone2/logs/dependency-cont-${JOB}.out" \
              "/data/gykwak03/capstone2/logs/dependency-cont-${JOB}.err"
```

기본 요청은 normal GPU 1장, 최대 12시간이며 위 조건을 순차 실행한다.
로그 추적의 Ctrl+C는 tail만 종료한다. 학습 완료는 status.json과 sacct를 함께 확인한다.

처음에 작은 실제 데이터 확인부터 하려면 새 output에 다음과 같이 제출한다.

```bash
bash scripts/submit_dependency_continuation.sh \
  --source-run "$SOURCE" --output "${RUN}_pilot" \
  --artifact-dir "$ECG_ARTIFACT_DIR" \
  --seeds 42 --checkpoints 1500 --updates 50 --optimizer-modes keep
```

옵션을 바꾸는 실험은 항상 새 output을 사용한다. 이미 실행 중인 job과 output을 공유하지 않는다.
14개 factorial 조건까지 실행하려면 최초 제출 명령에 다음을 추가한다.

```bash
--losses all-hidden anchor
```

일부 seed만 완료됐다면 `--seeds 42 43`을 명시할 수 있다. 선택한 모든
`seed_N/pretrain/update_N.pt`가 있어야 제출된다. 새 checkpoint를 만들거나 없는 seed를 조용히 생략하지 않는다.
원래 run 전체가 completed일 필요는 없지만, 선택한 source checkpoint를 동시에 변경해서는 안 된다.

재개는 **동일한 source·설정·새 RUN**에 `--resume`을 추가한다. 이 옵션은 기존 실험의 목적함수를
바꾸는 기능이 아니다. 완료된 seed/checkpoint는 건너뛰고, 중단된 분기는 저장된 model/optimizer/RNG에서 이어간다.
code/data/checkpoint identity가 바뀌면 재개를 거부한다.

## 읽을 결과

```bash
cat "$RUN/status.json"
"$ECG_PYTHON" - "$RUN" <<'PY'
from pathlib import Path
import sys
import pandas as pd
p = Path(sys.argv[1])
df = pd.read_csv(p / 'summary.csv')
print(df[['seed','checkpoint','case','optimizer_mode','validation_mse','improvement']]
      .to_string(index=False, float_format=lambda x: f'{x:.6f}'))
PY
```

- `summary.csv`: 완료된 각 분기의 최종 공통 Random 검증 결과.
- `learning_curves.csv`: 분기별 평가 시점 변화. 최종 값만 보고 중간 개선을 놓치지 않도록 한다.
- `contrasts.csv`: E−D, F−D 및 matched loss / optimizer 차이. 양수는 왼쪽 조건에 유리하다.
- `context_summary.csv`: 기존 5종 문맥의 조건별 진단. 문맥별 tradeoff 확인용.
- `seed_*/base_*/gains.csv`: 같은 정의의 학습 전 Gain.
- `no_update_control.json`, `random_replay_audit.json`: 평가 재생과 원래 Random 학습 흐름 점검.
- 분기별 `initial_state.json`: keep/reset이 실제로 다른 AdamW 상태로 시작했는지 확인.
- `compute_cost.json`: 이번 추가 학습 비용만 기록; `pretraining_updates=0`.
- `REPORT.md`: 지표와 해석 한계.

판단 순서는 D/keep의 변화 → E/F 대 D → keep/reset 차이 → 필요하면 새 matched loss 비교다.
D도 악화된다면 먼저 학습 continuation 및 평가 분포를 점검한다. D가 개선되고 E/F가 악화되면
이번 문맥 마스크 정책이 해당 평가에서 불리했다는 근거다. 결과를 얻기 전 성능 개선을 약속하지 않는다.

## 로컬 검증

```powershell
python -m pytest tests/test_dependency_continuation.py tests/test_dependency_gain.py -q
```

합성 source run에서 `--smoke`를 지정하면 CPU 작은 모델로 최대 2 updates만 실행한다.
실제 데이터의 CPU 대체 실행은 허용하지 않는다. Slurm job은 Python smoke가 자동 제출하지 않는다.
실제 모델, 데이터 선택, mask semantics가 source와 동일한지 검증하고 source artifact는 read-only로 유지한다.

원래 producer의 과학적 코드 hash가 바뀐 경우 provenance 보호를 위해 재사용을 거부한다.
이번 변경은 producer 파일을 수정하지 않고 별도 runner로 구현했다.
