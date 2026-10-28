# 다음 Codex에 전달할 통합 프롬프트

첨부 원문의 [1]–[10] 요구사항을 기준으로 작성했다.
정리된 저장소의 참고 경로를 반영하고, [11]에 Windows 로컬 개발·Git 전달·Seraph 실행 조건을 통합했다.
새 결과의 기본 namespace는 `experiments/artifacts/phase1/`로 통일한다.
아래 구분선 이후 전체를 전달한다. 구현 범위와 상세 학습·평가·검증 요구사항을 요약해서 생략하지 마라.

---

이 저장소에서 ECG 마스킹 연구의 1단계 코드를 정리하고 구현해줘.
설명이나 계획만 제시하지 말고, 기존 구현을 확인한 뒤 필요한 코드를 작성하고 로컬 테스트까지 완료해줘.

이번 작업 범위는 다음 세 가지다.
1. 전처리 규칙 정리 및 검증
2. 기본 MAE를 사용한 patch length·mask ratio pilot
3. Ridge / 1D CNN / Small Transformer의 동일 마스크 복원 성능 평가

난이도별 MAE 사전학습, predictability-aware sampler, 최종 downstream 비교는 이번 범위에 포함하지 않는다.


[1. 작업 위치와 참고 파일]

로컬 저장소:
C:\Users\gykwa\kyunghee2026\capstonedesign2\predictability-guided-ecg-masking

서버 저장소:
/data/gykwak03/capstone2/predictability-guided-ecg-masking

먼저 AGENTS.md가 있으면 읽고, git status를 확인해 기존 변경사항을 보존해줘.

주요 참고 파일:

전처리:
- preprocessing/PTBXL_preprocessing.ipynb
- preprocessing/run_preprocessing.py
- tests/test_preprocessing_cli.py

기존 MAE와 원인 탐색:
- archive/20261028_legacy/experiments/Masking_pilot.ipynb
- archive/20261028_legacy/experiments/mae_cause_model.py
- archive/20261028_legacy/experiments/mae_cause_data.py
- archive/20261028_legacy/experiments/mae_cause_eval.py
- archive/20261028_legacy/experiments/mae_cause_plan.py
- archive/20261028_legacy/experiments/run_mae_cause_study.py
- archive/20261028_legacy/experiments/MAE_CAUSE_STUDY.md
- archive/20261028_legacy/tests/test_mae_cause_study.py

기존 추정기:
- experiments/predictability_estimators.py
- experiments/run_predictability.py
- tests/test_predictability.py

반복 마스킹·patch 난이도·시각화:
- archive/20261028_legacy/experiments/patch_difficulty.py
- archive/20261028_legacy/experiments/run_patch_difficulty.py
- archive/20261028_legacy/experiments/PATCH_DIFFICULTY.md
- experiments/visualize_predictability.py
- archive/20261028_legacy/tests/test_patch_difficulty.py

기존 결과:
- experiments/artifacts/predictability/
- experiments/artifacts/st_mem_grid_v2/

로컬에 존재하면 다음 원인 탐색 결과도 참고:
../outputs/mae_cause_analysis_150746_20261027/REPORT.html
../outputs/mae_cause_analysis_150746_20261027/reconstruction_summary.csv
../outputs/mae_cause_analysis_150746_20261027/probe_summary.csv
../outputs/mae_cause_analysis_150746_20261027/prefix_identity_audit.csv

archive/20261028_legacy/experiments/README.md와 ECG_JEPA_PILOT.md에는 과거 모델 선정 기록이 남아 있다.
이번 작업에서는 아래에 명시한 개선 Plain ECG-MAE를 사용한다.
과거 기록을 보고 ECG-JEPA나 다른 모델로 자동 변경하지 마.


[2. 연구 목적과 현재 결정]

연구 질문은 “어떤 난이도의 마스킹으로 사전학습해야 유용한 ECG 표현을 학습하는가?”다.

현재는 그 연구를 위한 기본 모델과 마스크 단위, 난이도 추정기를 준비하는 단계다.

탐색용 기본 모델:
확대 Plain ECG-MAE + encoder·decoder의 고정 sin/cos 시간 임베딩

이 모델은 우리 구현의 개선 MAE다. 공식 SOTA 모델이라고 표현하지 않는다.
공식 ST-MEM은 후속 최종 검증 모델로 사용할 예정이다.

기존 원인 탐색에서는 시간 임베딩만 변경한 factor_010이 다음 결과를 보였다.
- Encoder 384차원 / 12층 / 6 heads
- Decoder 256차원 / 4층 / 4 heads
- 독립 초기화
- Encoder와 decoder의 시간 임베딩을 고정 sin/cos로 적용
- 학습 가능한 lead embedding 유지
- 추가 patch 전·후 LayerNorm 없음
- SEP 없음
- lead-wise decoder 없음
- 일반 전체-token decoder 유지

이 설정을 이번 pilot의 기본 backbone으로 사용해줘.
“모든 옵션을 적용한 all_on”과 혼동하지 마.

기존 factor_010의 주 학습 마스크는 global 50%였다.
이번에 도입하는 per-lead 제약에서 같은 성능이 나온다고 가정하지 말고 새로운 실험으로 기록해줘.


[3. 기존 코드와 결과 보존]

기존 원인 탐색 실행은 source hash를 검증하므로, 그 실험의 핵심 소스·노트북·결과를 수정하면 재개가 어려워질 수 있다.

- 기존 mae_cause_* 및 run_mae_cause_study.py는 이번 작업에서 수정하지 않는 방향을 우선한다.
- 기존 노트북과 checkpoint, 결과 CSV, plan.json을 덮어쓰지 않는다.
- 필요한 새 기능은 새 모듈과 진입점으로 분리한다.
- 기존 추정기의 순수 함수와 모델은 검증 후 재사용한다.
- 같은 기능을 이름만 바꿔 중복 구현하지 않는다.
- 새 기본 모델을 별도 모듈로 추출한다면 기존 factor_010과 초기 가중치·forward 출력의 일치 여부를 CPU에서 검증한다.
- README에 새 진입점과 현재 기본 모델을 안내하고, 과거 실험 기록은 보존한다.


[4. 전처리 규칙]

기본 입력:
- PTB-XL 10초 12-lead ECG
- 원본 500 Hz WFDB의 physical mV
- 출력 shape: (12, 2500), float32
- canonical lead 순서 유지

처리 순서:
1. 기존 코드와 같은 polyphase resampling으로 500 → 250 Hz (resample_poly, padtype="line")
2. 0.67–50 Hz, 4차 Butterworth zero-phase band-pass
3. ECG별·lead별 Z-normalization

기본 Z-normalization:
- 각 ECG의 각 lead에서 전체 10초 평균과 표준편차 사용
- population std, ddof=0
- std 하한 epsilon=1e-6
- 정확한 수식은 기존 normalize_ecg와 일치시킨다.

이 방식은 마스킹 전에 적용되므로 정규화 통계에 이후 hidden이 되는 구간도 포함된다.
이 특성을 문서와 설정에 명시하고, visible-only 정규화 또는 train 통계 정규화와 혼동하지 마.
모델 입력에서 hidden 값이 차단되는 검사와 전처리 통계 의존성 검사를 구분해줘.

기존 40 Hz 배열을 다시 필터링해 50 Hz 데이터로 사용하지 않는다.
artifact의 manifest, sampling rate, filter, lead 순서, metadata 정렬을 검증한다.
조건이 맞는 50 Hz artifact가 있으면 재사용한다.

환자 분할:
- folds 1–8: train
- fold 9: validation 및 pilot 선택
- fold 10: 최종 test로 보존

이번 학습·pilot·추정기 평가에서는 test 파형과 label 파일을 열지 않는다.
기존 전체 전처리 artifact에 test 파일이 존재하는 것은 허용하지만, 이번 runner가 이를 읽지 않도록 해줘.


[5. Mask 단위와 pilot]

기본 단위는 lead × time patch다. bool mask에서 True는 hidden으로 통일한다.
lead-major 순서와 patch index ↔ lead·시간 구간 변환을 명시한다.
길이는 samples와 초를 함께 기록한다.

1차 pilot:
- P25 = 0.1초
- P50 = 0.2초
- P125 = 0.5초
- mask ratio는 50%를 primary로 고정

2차 sensitivity:
- 선택한 patch length에서 60%, 70%를 선택적으로 비교할 수 있게 구현
- 기본 실행에서 전체 조합을 자동으로 대규모 학습하지 않는다.

향후 공식 ST-MEM으로 정책을 옮길 수 있도록 primary 마스크 공간은 다음으로 통일한다.
- 각 lead에서 정확히 같은 수의 patch를 가림
- lead별 위치는 독립적으로 무작위 선택
- P50·50%는 lead마다 50개 중 25개, 총 600개 중 300개 hidden
- global uniform masking과 다른 제한된 분포임을 기록
- 지정 비율을 정확히 구현할 수 없는 조합은 조용히 반올림하지 말고 거부하거나 별도 명시

같은 조건에서 seed별 ECG 순서와 전체 effective-batch mask를 먼저 정한 뒤 microbatch로 분할한다.

기본 MAE 학습 설정:
- AdamW, lr=3e-4
- betas=(0.9, 0.95)
- weight decay=0.05
- warmup 100 updates + cosine decay, 최저 lr=3e-5
- gradient clipping=1
- FP32 기본
- microbatch 4 × accumulation 8 = effective batch 32
- 3,000 updates를 기본 pilot budget으로 설정하되 config로 변경 가능

patch 조건별로 동일한 환자·ECG subset과 update budget을 사용한다.
같은 updates가 같은 FLOPs나 GPU 시간이라는 주장은 하지 않는다.

pilot 결과:
- hidden-only reconstruction MSE / NRMSE
- zero baseline 대비 개선
- hidden waveform correlation
- prediction / target RMS
- 고정 train·validation mask bank 학습 곡선
- 실제 파형과 hidden 구간 복원 예시
- update당 시간, 전체 단계별 시간, peak VRAM

서로 다른 patch length와 mask ratio의 복원 MSE만 비교해서 “가장 쉬운 조건”을 자동으로 최적 설정으로 고르지 않는다.
학습 안정성, 진폭 붕괴 여부, 기준선 대비 개선, 비용을 함께 보여줘.
실제 결과가 없으면 최적 patch·ratio가 확정됐다고 쓰지 마.


[6. 공통 데이터·mask bank]

Ridge, CNN, Small Transformer는 같은 ECG, 같은 mask_id, 같은 hidden target으로 비교한다.

저장할 정보:
- ecg_id, patient_id, split/pool
- mask_id, seed, repeat
- patch_samples, patch_seconds
- requested/realized mask ratio
- mask policy, lead별 hidden count
- mask hash와 재생 가능한 mask 정보

독립 RNG를 사용해서 모델의 초기화나 실행 순서가 mask bank를 바꾸지 않게 한다.
저장된 bank를 나중에 그대로 replay할 수 있어야 한다.

Estimator용 데이터는 train 환자 안에서 fit pool과 score pool을 분리한다.
기존 patient_pools 구현을 참고하되 환자 중복을 명시적으로 검사한다.

- estimator-fit: 모델 학습과 NRMSE 분모 통계 계산
- validation: hyperparameter·학습 상태 점검
- held-out train score pool: mask별 난이도 측정

fit한 환자에 대한 오차를 그대로 일반화 난이도라고 사용하지 않는다.
향후 train 전체를 점수화할 수 있도록 patient-wise cross-fitting 확장 가능성을 고려하되, 이번에는 명확한 fit/score 분리부터 구현한다.

50% mask 평가에서는 필요하면 M과 보수 mask ~M을 사용해 patch별 hidden 관측 수를 맞춘다.
두 mask는 독립 표본이 아니므로 pair 정보를 기록한다.
미관측 patch를 오차 0으로 채우지 않는다.


[7. Ridge / CNN / Small Transformer]

기존 predictability_estimators.py의 다음 구현부터 검토해줘.
- LaggedRidge
- SmallCNN
- SmallTransformer
- normalized_masked_loss
- reconstruction_metrics

세 추정기는 공통으로:
- hidden waveform 값을 입력에서 차단
- mask/observed indicator 제공
- visible의 실제 0과 hidden을 구별
- hidden samples에서만 loss와 평가 계산
- label을 입력·복원 target·난이도 점수 계산에 사용하지 않음

Ridge:
- visible 값과 availability indicator를 사용하는 선형 baseline
- lag 범위, ridge alpha, target subsampling 수를 설정에 기록
- alpha를 조정한다면 validation만 사용
- 사용한 ECG-mask bank와 실제 fitting target 수를 저장
- 해석 가능한 baseline으로 구현하고, hidden target이 feature에 유입되지 않도록 검증

Lagged Ridge는 별도의 회귀 목적함수가 아니라 시간 offset 특징을 사용하는 Ridge다.
현재 기본 lag는 [-2P, -P, -1, 0, 1, P, 2P] samples다.
P가 바뀌면 문맥의 초 단위 범위도 변하므로, patch pilot 이후 추정기 비교에서는 동일한 초 단위 lag를 고정할지 명시적으로 정하고 저장해줘.
P50/250 Hz의 기존 offset은 [-0.4, -0.2, -0.004, 0, 0.004, 0.2, 0.4]초다.
과거 재현 옵션은 구분해서 보존하고, 양수 offset도 쓰는 오프라인 복원 baseline임을 명시해줘.

CNN:
- mask-aware 1D CNN
- 입력 채널과 receptive field를 문서화
- validation loss, zero baseline, 진폭 붕괴 여부를 확인

Small Transformer:
- 기본 MAE와 별개의 경량 추정기
- 기존 구현에 있는 고정 sin/cos 시간 위치 정보와 lead 정보 재사용 가능
- 크기·층 수·heads를 config에 명시
- neural estimator 두 모델의 학습 mask ID와 batch 순서를 동일하게 유지

“동일 visible input”과 “동일한 실제 receptive field”는 구분한다.
Ridge/CNN/Transformer의 문맥 활용 범위 차이는 기록한다.
Ridge의 폐쇄형 fitting과 신경망 학습이 동일 compute라는 주장은 하지 않는다.


[8. 지표와 시각화]

NRMSE:
sqrt(mean_hidden(((prediction - target) / s_lead) ** 2))

s_lead는 estimator-fit ECG에서 계산한 고정 lead별 표준편차다.
validation·score ECG나 개별 target patch의 표준편차로 분모를 다시 계산하지 않는다.
모든 추정기에 같은 분모를 사용하고, 단위와 수식을 저장한다.

필수 결과:
- mask별 각 estimator의 NRMSE
- zero baseline 대비 개선
- reconstruction correlation 및 prediction/target RMS
- ECG별 lead × time patch NRMSE map
- 집계 map의 hidden 관측 수
- 실제 ECG 위 hidden 구간과 복원 예시
- neural estimator 학습 곡선
- 시간·VRAM·모델 parameter 수

patch 수준 오차와 전체 mask 오차를 다른 항목으로 저장한다.
낮은 NRMSE=Easy라는 방향을 통일한다.
percentile을 제공하면 높은 점수가 Easy인지 Hard인지 명시한다.

이번 단계에서는 원시 NRMSE와 map을 신뢰할 수 있게 만드는 데 집중한다.
Easy/Medium/Hard 정책 사전학습과 soft sampler는 구현 범위에서 제외한다.
기존 run_predictability.py가 자동 생성하던 그룹을 그대로 이번 연구의 확정 난이도 정의로 채택하지 마.


[9. 실행 구조와 서버]

새 코드 구조는 저장소 스타일에 맞춰 결정하되 다음 역할을 분리한다.
- config / data / mask bank / model / estimator / metrics
- plan 또는 dry-run
- CPU smoke
- MAE pilot
- estimator fit / score
- 결과 분석

원본 데이터와 checkpoint를 CLI에 명시할 수 있게 하고, 현재 작업 디렉토리에 따라 경로가 우연히 바뀌지 않도록 한다.

서버:
- Python: /data/gykwak03/envs/ecg-mask-pilot/bin/python
- 50 Hz artifact: /data/gykwak03/capstone2/ptbxl_250hz_bp067_50
- 원본: /data/gykwak03/capstone2/ptbxl_raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
- 로그: /data/gykwak03/capstone2/logs
- account: ugrad_advisor_x
- QOS: ugrad
- debug partition: debug_ugrad_advisor_x, 최대 4시간
- batch partition: batch_ugrad_advisor_x
- 사용자 GPU 총량 1장, QOS 최대 24시간
- gres: gpu:normal:1

실제로 RTX 2080 Ti가 할당된 적이 있으며 A2000도 사용한 적이 있다.
GPU 종류와 실제 메모리를 기록하고 특정 GPU 속도·메모리 사용량을 보장하지 않는다.

기존 개인 환경:
- PyTorch 2.5.1+cu121
- pandas 3.0.6
- NumPy 2.4.6
- WFDB 4.3.1

WFDB 4.1.2는 pandas 3에서 실제 import 오류가 났으므로 다시 설치하지 않는다.
기존 환경을 무조건 downgrade하거나 공용/base에 설치하지 않는다.
새 의존성이 필요하면 호환성을 확인하고 최소한으로 안내한다.

기존 40 Hz 결과와 다른 새 output namespace를 사용한다.
기본 namespace: experiments/artifacts/phase1/

checkpoint에 모델·optimizer·update·RNG·config·mask/data/source identity를 저장해 재개할 수 있게 한다.
완료 결과를 덮어쓰지 않고 부분 완료와 전체 완료를 구분한다.
한 비교가 실패해도 다른 결과를 볼 수 있도록 오류와 제외 사유를 남긴다.
검증 실패를 조용히 무시해서 결과를 정상으로 표시하지 않는다.


[10. 검증과 완료 조건]

실제 GPU 장기 학습은 자동 제출하지 말고, 로컬 CPU smoke와 필요한 테스트까지만 실행한다.
축소 모델·합성 데이터 smoke를 실제 ECG 성능 결과와 구분한다.

필수 검증:
- patchify/unpatchify와 lead·시간 index 일치
- 초 단위 환산 및 정확한 lead별 mask count
- mask bank 저장·재생 일치
- fit/score/validation 환자 분리
- hidden 입력 perturbation 후 세 estimator 예측 불변
- hidden-only loss 및 NRMSE 수식 검증
- fit-only denominator 통계
- 고정 시간 임베딩은 학습되지 않고 encoder·decoder의 나머지 학습 대상 가중치는 학습됨
- 기존 factor_010 모델과 새 기본 모델의 CPU parity
- 같은 effective batch에서 microbatch 분할이 mask를 변경하지 않음
- checkpoint 재개 후 update·데이터·mask 흐름 유지
- test 파일을 읽지 않음
- 실제 데이터에 평가 가능한 mask가 없거나 estimator가 기준선보다 못하면 명시적으로 표시

마지막으로 다음을 제공해줘.
1. 기존 구현 중 재사용한 부분과 새로 구현한 부분
2. 변경 파일과 디렉토리 구조
3. 확정한 전처리 규칙과 남은 제한
4. pilot 구성 및 선정 기준
5. estimator별 입력·target·NRMSE 정의
6. 로컬 테스트 결과
7. Seraph에서 준비 → smoke → profile → pilot → estimator fit/score → 로그·결과 확인까지 복사 가능한 명령
8. 실제 GPU 실험 전에는 아직 알 수 없는 성능·시간·최적 설정

이번 작업은 기존 72회 원인 탐색을 다시 돌리는 작업이 아니다.
앞으로의 난이도별 마스킹 연구를 위한 재사용 가능한 1단계 파이프라인을 완성해줘.


[11. 로컬 개발 → Git 전달 → Seraph 실행과 경로 계약]

**편집은 Windows 로컬 PC에서, 실제 GPU 학습은 Git으로 코드를 전달한 Seraph에서 수행한다.**
로컬 파일 시스템과 서버 파일 시스템은 다르다. 로컬에 서버 데이터가 있다고 가정하거나,
Git clone/pull이 데이터·환경·checkpoint까지 옮겨 준다고 가정하지 마라.

| 용도 | Windows 로컬 | Seraph 서버에서 확인된 경로 |
| --- | --- | --- |
| Git 저장소 | `C:/Users/gykwa/kyunghee2026/capstonedesign2/predictability-guided-ecg-masking` | `/data/gykwak03/capstone2/predictability-guided-ecg-masking` |
| 실험용 Python | 로컬에서 실제 실행되는 Python을 확인해 사용 | `/data/gykwak03/envs/ecg-mask-pilot/bin/python` |
| 원본 PTB-XL WFDB | 로컬 유무를 가정하지 않음; 필요 시 명시적으로 지정 | `/data/gykwak03/capstone2/ptbxl_raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3` |
| 사용할 250 Hz / 0.67–50 Hz 전처리 데이터 | 로컬 유무를 가정하지 않음; CPU smoke는 합성 데이터 사용 | `/data/gykwak03/capstone2/ptbxl_250hz_bp067_50` |
| 과거 0.67–40 Hz 데이터 | 과거 결과 재현 시에만 명시적으로 지정 | `/data/gykwak03/capstone2/ptbxl_250hz` |
| 공식 ST-MEM checkout | 이 단계의 실행에 필수로 요구하지 않음 | `/data/gykwak03/capstone2/ST-MEM` |
| 기존 실험 결과 | 각 clone의 `experiments/artifacts/` | `/data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/` |
| Slurm 로그 | 서버에서 생성한 뒤 필요 시 다운로드 | `/data/gykwak03/capstone2/logs/` |

위 서버 경로는 기존 사용 기록이다. 원격 파일의 현재 존재 여부는 서버 preflight에서 확인해야 한다.
로컬 전용 `.venv`의 실행 파일을 Linux에 복사하거나, 서버에서 Windows 절대 경로를 사용하지 마라.

다음 경로 계약을 새 코드·config·제출 스크립트·README에 함께 적용해라.

1. **코드 위치**는 Python 모듈/스크립트 자신의 위치에서 저장소 루트를 찾는다. 터미널의 현재 디렉토리에 의존하지 마라.
   archive 참고 경로는 그 저장소 루트 기준으로 찾는다. 로컬 사용자 이름이나 서버 절대 경로를 Python 본문에 박지 마라.
2. **데이터 위치**는 실행 config에서 분리한다. 원본이 필요한 명령에는 `--raw-dir`, 전처리 입력에는 `--artifact-dir`,
   결과에는 `--output-root`, 임시/재사용 캐시에는 `--cache-dir`를 제공해라.
   기존 전처리 CLI의 `--data-dir`는 유지하거나 `--raw-dir`와 호환되는 alias로 지원해라.
3. 경로 우선순위는 **명시적 CLI > 해당 `ECG_*` 환경변수 > 선택한 config > 문서화된 기본값**으로 통일해라.
   대응 환경변수는 `ECG_RAW_DIR`, `ECG_ARTIFACT_DIR`, `ECG_OUTPUT_ROOT`, `ECG_CACHE_DIR`로 한다.
   Python 실행 파일은 shell의 `ECG_PYTHON`으로 지정한다. 상대 경로 config 값은 config 파일 디렉토리를 기준으로 해석하고,
   CLI 상대 경로는 호출 시 cwd 기준이라는 규칙을 명시해라. 실행 직전에 최종 절대 경로를 출력해라.
4. **로컬 smoke와 서버 실데이터 실행을 분리**해라. 기본 smoke는 합성 ECG를 만들고 서버 경로·CUDA·원본 PTB-XL 없이 통과해야 한다.
   서버 실데이터 모드에서 입력이 없으면 찾으려던 경로와 필요한 옵션을 보여 주고 종료해라.
   임의의 로컬/과거 40 Hz 데이터로 fallback하거나 실제 데이터 실패를 합성 smoke 성공으로 처리하지 마라.
5. 서버에 이미 만들어진 50 Hz artifact는 manifest와 설정이 맞으면 재사용한다.
   추정기 학습에 원본 WFDB를 불필요하게 요구하거나 매 실행마다 ZIP 추출·전처리를 반복하지 마라.
   manifest 불일치는 명시적으로 알리고, 새 전처리 생성은 별도 명령·별도 출력 경로로 수행해라.
6. 새 결과의 기본 출력은 저장소의 `experiments/artifacts/phase1/` 아래로 하고,
   run ID로 구분해 기존 결과를 덮어쓰지 마라. 데이터는 기존 외부 디렉토리에 둔 채 읽는다.
   checkpoint·대용량 배열·cache는 Git 전송 대상에서 제외하고, 필요한 경우 별도 다운로드 대상으로 안내해라.
   compact CSV/JSON/보고서의 Git 추적 여부는 기존 `.gitignore` 규칙과 맞춰라.

README에는 아래 흐름을 **Windows PowerShell용 / Seraph Bash용으로 나눠** 실제 명령으로 제공해라.
새로 만드는 실행 파일·config의 실제 이름과 옵션에 맞춰 작성하고, 존재하지 않는 runner 명령을 완성된 것처럼 제시하지 마라.

- **로컬:** 코드 편집 → CPU smoke/테스트 → 변경 내용 확인 → 사용자가 선택한 파일 commit/push.
  자동 commit/push는 하지 말고, 기존 remote/branch 이름을 확인해서 안내해라. 모델·데이터 전체를 `git add .`로 전송하도록 안내하지 마라.
- **서버:** 저장소로 이동 → 실행 중인 job과 수정 사항을 확인한 뒤 사용자 요청에 따라 `git pull --ff-only`
  → 위 Python으로 import·데이터 manifest·출력 쓰기 권한을 preflight → Slurm 제출 → 로그/진행 상황 확인 → 결과 분석.
  실행 중 job이 소스를 읽고 있을 때 같은 checkout을 pull해 파일을 바꾸는 경우는 피하도록 안내해라.
- **결과 회수:** 분석용 작은 결과와 checkpoint/배열의 별도 전송 방법을 설명한다.
  Git에 추적되지 않은 결과는 `git pull`로 로컬에 생기지 않는다는 점을 명시해라.
  접속 host/port는 실제 SSH 설정을 확인해서 사용하고 임의의 주소를 만들지 마라.

Seraph용 config 예시와 실행 스크립트도 함께 제공해라. Bash 환경변수 예시는 다음 확인된 값으로 한다.

```bash
export ECG_REPO=/data/gykwak03/capstone2/predictability-guided-ecg-masking
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_RAW_DIR=/data/gykwak03/capstone2/ptbxl_raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50
export ECG_OUTPUT_ROOT="$ECG_REPO/experiments/artifacts/phase1"
export ECG_CACHE_DIR="$ECG_REPO/experiments/artifacts/phase1/cache"
```

이 값은 config/예시에서 제공하며, 사용자가 다른 디렉토리로 바꾸면 그대로 실행되어야 한다.
제출 스크립트는 저장소 위치와 환경변수를 명시적으로 worker에 전달하고 Bash LF 줄바꿈을 사용해라.
출력/로그 디렉토리는 제출 전에 생성해야 한다. 로그인 노드에서 실데이터 전처리·GPU 학습을 시작하지 마라.
partition은 기존에 사용한 `batch_ugrad_advisor_x`를 예시로 하되 override를 지원하고,
account=`ugrad_advisor_x`, QOS=`ugrad`, `--gres=gpu:normal:1`, 한 job 최대 24시간을 반영해라.
특정 GPU 이름·node를 고정하지 말고 실제 할당 GPU/VRAM, Torch/CUDA 버전과 effective batch를 로그에 남겨라.
긴 실험은 실행 단위를 나누거나 resume할 수 있게 하고, checkpoint의 실험 설정 일치 여부를 검사해라.
실제 서버 명령 실행·원격 접속·job 제출은 이 구현 작업에 포함하지 않는다.

[12. 통합 실행 지침의 추가 완료 조건]

- 위 [1]–[10]의 상세 구현·검증·최종 보고 항목과 [11]의 개발·배포 조건을 모두 충족해줘.
- plan/dry-run, CPU smoke, profile, MAE pilot, estimator fit/score, 분석의 진입점을 구분하고 실제 생성한 파일과 맞는 명령을 제공해줘.
- config 상대 경로, CLI 상대 경로의 기준과 CLI > 환경변수 > config > 기본값의 우선순위를 검증해줘.
- 다른 cwd에서의 실행, 데이터 누락 시 구체적인 오류, Windows 경로와 Linux 경로의 분리를 확인해줘.
- 로컬 smoke config, Seraph 실행 config, Slurm 제출 스크립트, 결과 회수 안내를 제공해줘.
- 실제로 검증한 로컬 결과와 사용자가 서버에서 실행해야 하는 항목을 구분해 보고해줘.
- 공식 ST-MEM checkout은 이번 탐색 단계의 필수 실행 의존성으로 만들지 말고 필요할 때 명시적으로 경로를 지정하게 해줘.
