# ECG-JEPA: mask 단위·길이·비율 pilot

사용 모델은 **Sehun Kim, arXiv:2410.08559v1**의 ECG-JEPA이다.
Weimann–Conrad의 동명 JEPA나 xECG로 바꾸지 않는다.
공식 소스: https://github.com/sehunfromdaegu/ECG_JEPA
고정 commit: `d937ad2c2c8a1e22856ce7e4a23a30f84a71217c`.

## 목적과 해석 범위

주 질문은 "어떤 patch 길이에서 난이도별 마스킹의 효과를 비교할 것인가"이다.
기존 결정대로 **50% masking이 primary**이며 길이 0.1·0.2·0.5초를 먼저 비교한다.
60·70%는 후속 sensitivity 분석이다. 난이도별 Easy/Medium/Hard 정책은 이 pilot의
random baseline을 정한 뒤, 길이·비율을 고정하고 별도로 비교한다.
이 파일을 작성하거나 CPU smoke가 통과한 것은 성능 검증 결과가 아니다.

| 조건 | P25 | P50 | P125 |
|---|---:|---:|---:|
| 250 Hz에서 길이 | 0.1초 | 0.2초 | 0.5초 |
| lead당 patch | 100 | 50 | 20 |
| 12-lead 전체 token | 1,200 | 600 | 240 |
| 50% hidden token | 600 | 300 | 120 |

각 ECG와 update마다 새 mask를 생성한다. 각 lead에서 정확히 같은 개수를
가리되 위치는 lead별 독립적으로 뽑는다. **모든 전체 조합에 대한 균일 추출이 아닌,
lead별 hidden count를 고정한 제한된 분포**이다. 학습·결과에 이 제약을 기록한다.
같은 P50/50%에서 모든 lead의 시간을 함께 가리는 topology anchor도 제공한다.
12-lead anchor는 topology만 분리하기 위한 대조군이며 원논문 전체 재현은 아니다.

## 원본과의 차이, checkpoint

원본은 I·II·V1–V6 **8 leads**, P50=0.2초, 같은 시간 구간의 모든 lead를 가린다.
현재 주 실험은 **12-lead, 독립적인 lead × time 선택**으로 확장했다.
공식 encoder/predictor Transformer와 CroPA, 초기화, masked latent SmoothL1을
그대로 불러오고, visible token 선택과 lead별 predictor 재조립만 일반화한다.
upstream checkout은 수정하지 않는다. 같은 시간 mask를 주었을 때 공식 forward와
동일한 loss가 나오는지 검사한다. Teacher는 EMA student로 업데이트하며
drop-path를 꺼 일관된 target을 사용한다(원본 model.train()과 다른 적용 방식).
입력은 기존 **250 Hz / 0.67–50 Hz / ECG별·lead별 Z-normalization** artifact이다.
공식 CODE-15/Chapman/Ningbo pretraining의 성능을 재현했다고 주장하지 않는다.

공식 README가 제공하는 **encoder-only** checkpoint:

- Random: https://drive.google.com/file/d/1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx/view
- Multi-block: https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view

링크·파일명은 확인했으나 현재 이 작업에서 가중치를 내려받거나 실물 tensor를
검사한 것은 아니다. 공식 저장 코드는 `encoder`와 `epoch`만 저장하며 predictor,
teacher, optimizer 복구용 checkpoint가 아니다. 사용자 파일은 weights_only=True와
strict=True로 읽는다. **12-lead나 다른 patch 길이에 억지로 호환시키지 않는다.**
`--checkpoint FILE --leads 8 --patch-samples 50`은 공개 encoder에서 출발하는
**추가 사전학습 진단**에만 사용한다. Teacher는 encoder를 복사하고 predictor는
새로 초기화한다. scratch pilot과 섞어 순위를 계산할 수 없다.
주 pilot은 checkpoint를 쓰지 않고 동일 seed에서 새로 사전학습한다.

## 학습·평가 조건

- Train folds 1–8 중 고정 seed로 고른 5,000 ECG; validation fold 9 전체.
- Fold 10의 waveform·label을 열지 않는다. 환자 overlap·artifact 정합성 검사를 수행한다.
- Pilot 기본 1,000 optimizer updates, microbatch 4 × accumulation 32 = effective batch 128.
  총 ECG 노출 128,000회, 5,000 ECG에 대해 평균 25.6회이다. 균등한 epoch 순회가 아니라
  update별 무작위 추출이므로 "정확히 25.6 epochs"라고 부르지 않는다.
- AdamW LR 2.5e-5, betas=(0.9,0.999), weight decay=0.05(1D·bias 제외),
  5% updates warmup + cosine to 1e-6, gradient clipping 1.
- EMA 0.996→1.0, auto BF16 on supported GPUs / FP16 otherwise; FP16 GradScaler.
- 원본 Base 설정: encoder dim768/depth12/heads16, predictor dim384/depth6/heads12,
  drop-path 0.1. `smoke`만 dim32/depth1로 줄인다. Profile/pilot은 Base이다.
- 같은 seed의 모든 patch 조건에서 P50 기준 모델을 먼저 초기화한다. 길이가 바뀌면
  입력 projection·고정 sin/cos positions만 새로 만들고 공유 encoder/predictor 가중치는
  동일하게 유지한다. 공유 encoder 초기 tensor hash와 학습 ECG 순서 hash를 저장한다.
- 같은 updates·effective batch·ECG 노출로 비교한다. 길이·비율에 따라 token 수가 달라져
  **동일 FLOPs/시간 비교는 아니다**. 시간·최대 VRAM을 함께 기록한다.
- 초기 encoder와 사전학습 후 student encoder를 각각 freeze하고 **한 개 Linear**를 학습한다.
  사전학습에 선택한 같은 5,000 ECG 중 label이 있는 데이터로 head를 학습한다.
  train feature 통계로 scaling, BCE, AdamW 1e-3/WD1e-4/batch128, 100 epochs;
  validation AUROC로 probe epoch 선택. 모든 조건에서 동일하다.
- Primary: validation Macro AUROC; secondary: Macro AP(AUPRC), F1@0.5,
  초기 encoder 대비 AUROC gain, across-ECG feature std, VRAM/시간.
- **latent loss/NRMSE로 patch 조건의 승자를 정하지 않는다.** JEPA는 waveform을 복원하지
  않는다. 낮은 latent loss와 feature collapse를 구별해야 한다.
- Seed 42로 자원/오류를 먼저 확인하고, 후보 비교는 같은 seed 42·43·44로 반복한다.
  작은 차이·seed 편차가 비슷하면 승자를 단정하지 않고 자원 비용도 고려한다.
  최종 정책 실험에서 길이·비율을 고정한 뒤 별도의 test 평가를 진행한다.

## Linux / Vast.ai에서 실행

GPU 결제·인스턴스 생성은 이 코드가 수행하지 않는다. CUDA 호환 PyTorch가 이미 설치된
Linux 환경에서 프로젝트와 **50 Hz artifact 전체**를 옮긴 뒤 repo root에서 실행한다.
GPU·드라이버마다 PyTorch 설치가 달라지므로 기존 PyTorch를 임의로 교체하지 않는다.

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.5.2 pytest==8.3.5
# 모델에 필요한 timm leaf만 불러와 torchvision 전체 설치/충돌을 피한다.
python -m pip install --no-deps timm==1.0.8
python experiments/setup_sota_repos.py --only ECG_JEPA
python experiments/run_ecg_jepa_pilot.py --mode plan
```

기본 external 위치는 repo의 형제 디렉토리인 `../external/ECG_JEPA`이다.
다른 위치는 setup의 `--external-root`, runner의 `--jepa-repo`로 지정한다.

우선 P25/P50/P125의 Base GPU profile. 아래 첫 명령은 계획만 출력하고 두 번째가 실행한다.

```bash
python experiments/run_ecg_jepa_grid.py --stage primary --mode profile
python experiments/run_ecg_jepa_grid.py --stage primary --mode profile --execute
```

Profile은 조건당 100 updates, 64 ECG, microbatch4/accumulation1이고 linear probe를 생략한다.
처음 5회 이후 median update seconds와 peak VRAM을 확인한다.
OOM이면 모든 비교 조건에 동일한 microbatch를 쓰고 accumulation을 늘려 effective batch를
유지한다. 실제 profile 없이 A100 속도나 전체 완료 시간을 보장하지 않는다.

자원을 확인한 뒤 50% masking의 길이 3개 × seed3개를 한 GPU에서 순차 실행:

```bash
python experiments/run_ecg_jepa_grid.py --stage primary --mode pilot --execute
```

60·70% sensitivity 및 topology anchor는 별도로 실행한다.

```bash
python experiments/run_ecg_jepa_grid.py --stage sensitivity --mode pilot --execute
python experiments/run_ecg_jepa_grid.py --stage topology_anchor --mode pilot --execute
```

각 grid 실행의 `grid_*/job_*.log`에 stdout/stderr가 저장된다. 한 조건이 실패하면
나머지 조건을 자동 시작하지 않는다. `tail -f PATH/TO/grid_*/job_01.log`로 진행을 볼 수 있다.

완료된 primary pilot 9개를 명시해서 비교한다. 폴더 전체 glob으로 grid 로그나 profile,
smoke 결과를 섞지 않는다.

```bash
python experiments/compare_ecg_jepa_pilot.py RUN1 RUN2 RUN3 RUN4 RUN5 RUN6 RUN7 RUN8 RUN9 \
  --output experiments/artifacts/ecg_jepa_pilot_bp067_50/primary_comparison
```

`validation_ranking.csv`는 validation 탐색 순위이며 통계적 유의성/논문 SOTA 결과가 아니다.
세 seed, 동일 train/validation·예산·source·GPU·precision·초기 backbone·ECG 순서를 확인한다.
여러 GPU를 쓰면 속도 순위와 성능 순위를 분리해 별도 설계를 해야 한다.
출력의 `final.pt`는 평가용 가중치이며 중단 후 exact resume 기능은 제공하지 않는다.

## 코드 검사

```bash
python -m pytest tests/test_ecg_jepa_pilot.py -q
python experiments/run_ecg_jepa_pilot.py --mode smoke --device cpu --probe-epochs 2
```

Smoke는 실제 train/validation ECG를 사용하지만 축소 모델·2 updates만 실행하므로
성능 비교 대상으로 사용할 수 없다. 테스트는 token 단위 hidden-value isolation,
native time-mask 계산 일치, exact mask counts, teacher freeze/EMA, 초기 가중치 pairing을 확인한다.
