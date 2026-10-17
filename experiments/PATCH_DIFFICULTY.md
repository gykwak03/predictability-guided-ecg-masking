# 같은 ECG 안의 patch 난이도: 50% masking 검증

`run_patch_difficulty.py`는 기존 predictability의 Ridge/CNN/Transformer checkpoint를
고정하고 **추가 학습 없이** 추론한다. Estimator-fit과 환자가 겹치지 않는 기존 score
ECG만 사용하며 test/validation 파형을 열지 않는다.

## 두 질문을 따로 측정한다

1. **Patch 복원 난이도**: 다른 절반이 무작위로 가려지는 조건에서, 이 patch 자체의
   hidden-only 복원 오차가 반복해서 큰가?
2. **전체 문제와의 연관성**: 이 patch가 hidden인 mask의 전체 복원 오차가, visible인
   mask보다 평균적으로 큰가?

두 결과는 다를 수 있다. 1이 재현되어도 2를 자동으로 주장하지 않는다.
2는 정확히 50%를 유지하는 무작위 마스크 분포에서의 **포함 연관성**이다. Patch를
포함하면 context와 복원 target 구성 모두 바뀌므로 단일 patch의 인과 효과는 아니다.
‘2번 patch 때문에 전체 문제가 어려워진다’까지 주장하려면 후속 matched-swap 실험이
필요하다. 예를 들어 나머지 299개 hidden 위치를 고정하고, 2번과 대조 patch 하나를
교환하며 공통 target에서의 오차도 비교할 수 있다. 이 runner는 그 개입을 수행하지 않는다.

## 코드 흐름

1. 원래 `run_config.json`의 P, 50% mask, 전처리 manifest hash, 고정 train lead std와
   source `ecg_selection.csv`를 검증한다. Fit/score 환자 중복과 metadata 불일치는 차단한다.
2. 같은 ECG에 독립적인 **discovery / confirmation** context bank를 만든다.
3. 각 bank에서 무작위 50% mask M과 그 보수 ~M을 한 쌍으로 생성한다. P50에서는
   모든 context가 600개 중 정확히 300개를 가린다. 각 patch는 한 쌍에서 정확히 한 번
   hidden이므로 관측 수가 위치마다 같다. 쌍 사이의 무작위 추출은 독립이며 bank seed
   namespace도 다르다. 보수 mask는 다른 `half` 규칙을 사용하므로 기존
   `masks_from_bank` 대신 `patch_difficulty.replay_bank`로 재현한다.
4. 기본 pilot의 bank당 32쌍은 64 masks이다. 두 bank를 합치면 ECG당 128 masks,
   patch당 discovery 32회 / confirmation 32회 hidden 관측을 얻는다.
5. Estimator별로 patch의 normalized squared error를 context에 걸쳐 평균한 뒤 제곱근을
   취한다. 분모는 원래 estimator-fit ECG에서 구한 lead별 std이며, 해당 patch의 자체
   std로 나누지 않는다. Context별 NRMSE 평균·표준편차도 별도로 저장한다.
6. 각 ECG 내부에서 estimator별 patch 오차 순위를 percentile로 만들고 평균한다.
   **이번 점수는 높을수록 Hard**다. 원래 whole-mask `predictability`는 높을수록 Easy이므로
   방향을 혼동하지 않는다. Discovery에서만 세 그룹을 만들고 confirmation에는 고정 적용한다.
   P50은 ECG마다 Easy/Medium/Hard 각 200 patches이다. 점수가 같은 경계는 별도 표시한다.
7. 새 context에서 순위 상관, Hard 겹침, frozen 그룹별 오차와 순서를 확인한다.
   Discovery 그룹을 무작위 섞은 대조군과도 비교한다. 같은 평가 결과에서 그룹을 다시
   최적화하지 않는다. Bootstrap은 먼저 ECG를 환자별로 평균한 다음 환자 단위로 한다.

`patch_number=2`는 **1-based lead-major 전체 token 번호**다. P50에서 lead I의 두 번째
0.2초 patch를 의미한다. CSV에는 `lead`, `time_patch_index`(0-based), 시작/끝 초도 저장한다.
다른 ECG의 2번 patch가 같은 심장 박동 위상이라는 뜻은 아니다.

## 실행

기존 pilot의 다음 파일들이 필요하다.

- `run_config.json`, `ecg_selection.csv`
- `ridge_weights.npz`, `cnn.pt`, `transformer.pt`

Git으로 받은 CSV만으로는 새 context에서 재평가할 수 없다. Checkpoint는 원래 서버
run에 남아 있어야 한다. 없는 파일은 학습으로 자동 대체하지 않고 구체적인 오류를 낸다.
직접 만든 기존 checkpoint만 사용한다. Neural checkpoint는 `weights_only=True`로 읽으며
기존 NumPy std 필드에 필요한 타입만 허용한다.

서버 clone 루트에서 먼저 짧은 smoke를 실행한다. Smoke는 코드 확인용으로 항상
`smoke_only`이며 과학적 성공으로 판정하지 않는다. CPU도 가능하다.

```bash
python experiments/run_patch_difficulty.py \
  --mode smoke \
  --source-run experiments/artifacts/predictability/20261014T050200Z_pilot_seed42_eb872a85 \
  --artifact-dir ../ptbxl_250hz \
  --score-ecgs 2 --pairs-per-bank 4 --batch-size 4
```

실제 pilot은 기존 Slurm GPU 설정을 재사용한다. 이 문서는 명령 예시이며 자동 제출하지 않는다.

```bash
bash experiments/submit_pilot.sh patch-difficulty \
  --source-run "$PWD/experiments/artifacts/predictability/20261014T050200Z_pilot_seed42_eb872a85" \
  --score-ecgs 64 --pairs-per-bank 32 --batch-size 4 --seed 42
```

결과는 `experiments/artifacts/patch_difficulty/<timestamp>_pilot_seed42_<id>/`,
로그는 clone 부모의 `logs/patch-difficulty_JOBID.log`에 저장한다.
처음 64개는 source score pool의 ECG ID 순서로 고른 label-blind pilot subset이며 전체
score pool의 대표성을 보장하지 않는다. 확대하려면 `--score-ecgs 0`으로 기존 score
pool 전체를 사용한다. Mask seed를 43/44로 바꿔도 ECG subset은 같게 유지된다.
Estimator 학습 seed robustness는 별도 source run checkpoint로 비교해야 한다.

64 ECG × 128 masks = estimator당 8,192번의 ECG-mask 평가다. 기존 원본 pilot의
4,096 masks보다 scoring 양이 두 배다. 512 ECG 전체는 estimator당 65,536번이다.
GPU 시간과 VRAM을 실제 측정한 수치는 아직 없다. 메모리 부족 시 batch size를 줄인다.

## 결과 파일

| 파일 | 읽는 방법 |
| --- | --- |
| `context_mask_bank.csv` | ECG·bank·pair·half·seed·mask digest. 항상 50%인지 재현 가능 |
| `patch_scores.csv` | ECG × patch 한 행. 위치, discovery 그룹, 두 bank의 estimator별 NRMSE·관측 수·context 표준편차·whole-mask 포함 연관성 |
| `ecg_validation.csv` | ECG마다 순위 상관·Hard overlap·그룹 오차·포함 점수 재현성·진폭 대조군 |
| `summary.json` | 환자별 동일 비중의 평균, 환자 bootstrap 95% 구간, 미리 정한 pilot 기준과 두 질문별 상태 |
| `run_config.json` | checkpoint/전처리 hash, 설정·점수 정의·단위·누수 방지 정책 |
| `completion.json` | 코드 실행 완료 여부. 과학적 성공 여부와 구분 |

`--save-patch-cache`를 주면 ECG별 원오차 NPZ를 저장한다. NPZ는 Git에서 제외된다.
확대 실행의 `patch_scores.csv`도 커질 수 있으므로 Git에 추가하기 전에 파일 크기를
확인하고, 우선 `ecg_validation.csv`와 `summary.json`을 공유한다.

## 어떤 결과면 가능성이 있다고 보는가

다음은 **이번 pilot에서 미리 정한 탐색 기준**이며 보편적인 통계적 정답이 아니다.
Confirmation 결과를 본 뒤 통과하도록 기준을 바꾸지 않는다. 임계값 아래라고 해서
불가능하다는 뜻도 아니며, 특히 whole-mask 포함 연관성은 작은 효과를 추정하므로
32 hidden 관측으로 부족할 수 있다.

### A. Patch 자체를 상중하로 나눌 수 있는가

- 최소 10 patients, 순위 상관을 계산할 수 있는 환자도 최소 10명. Patch마다 각 bank에서
  최소 16 hidden 관측.
- 환자별 평균을 다시 평균한 discovery–confirmation consensus Spearman ≥ 0.60.
- Discovery Hard의 confirmation Hard 잔존 비율 평균 ≥ 0.60. 무작위 기준은 약 1/3.
- 세 estimator 모두 confirmation에서 Easy < Medium < Hard인 ECG 비율을 환자별로
  평균한 값 ≥ 0.80.
- 실제 Hard−Easy 차이에서 shuffled-group 차이를 뺀 값의 환자 bootstrap 95% 하한 > 0.
- Estimator 각각의 discovery–confirmation Spearman 평균 ≥ 0.50.
- 그룹 경계 동점이 발생한 ECG 비율의 환자 평균 ≤ 0.10.

모두 충족하면 `status=supports_conditional_patch_ranking`이다. 이는 이 고정 estimator와
50% 무작위 context 분포에서의 반복 가능한 **오차 순위**를 지지한다.
큰 진폭만으로도 raw error 순위는 안정될 수 있으므로 `energy_spearman`, estimator별
`energy_only_gap`, `energy_adjusted_spearman`도 반드시 확인한다. 진폭을 통제한 순위가
무너지면 ‘진폭을 넘어서는 predictability’를 확인했다고 주장하지 않는다.

### B. 해당 patch가 포함된 전체 문제가 더 어려운가

포함 점수는 `mean(전체 mask NRMSE | patch hidden) − mean(전체 mask NRMSE | patch visible)`다.
Discovery 포함 점수로 별도 그룹을 고정한다. 이 점수의 consensus 순위 상관 평균 ≥ 0.60,
confirmation 포함 점수의 Hard−Easy 차이의 환자 bootstrap 95% 하한 > 0을 검사한다.
최소 환자·관측 수 조건도 충족해야 `whole_mask_inclusion_status=supports_inclusion_association`이다.

A 통과 / B 불확실이면 **patch 자체 복원 난이도는 반복되지만 전체 mask 난이도에 대한
위치별 예측력은 아직 불확실**하다고 해석한다. 둘 다 통과해도 downstream SSL 효용은
별도 검증해야 한다. 이후 난이도 기반 policy를 만들면 다른 patch가 동시에 hidden이 되어
context가 바뀌므로 새 policy mask를 다시 평가해야 한다.

## 검증과 제한

`tests/test_patch_difficulty.py`는 정확한 50%·동일 coverage·CSV replay, 독립 bank에서
안정 신호 재현, 무작위 오류의 false success 방지, 진폭만의 효과 노출, 환자 단위 집계,
실제 모델 checkpoint를 사용하는 작은 CPU inference 흐름을 검증한다.
합성 신호와 무작위 checkpoint의 smoke 성공은 실제 ECG의 난이도 분리가 확인되었다는
근거가 아니다. Estimator·학습 seed·context seed·mask 구조를 바꾸는 robustness는 남는다.
