# Mask 단위 확정 pilot

## 앞으로의 서버 실행과 Git 결과 저장

`experiments/submit_pilot.sh`는 서버의 기존 `run_stmem.sbatch`에서 GPU·계정·파티션
설정만 재사용하고, 결과를 **현재 clone의 `experiments/artifacts/` 아래**에 저장합니다.
실행 파일이나 결과 경로를 clone 밖의 예전 파일로 지정하지 않습니다.
서버의 clone 루트에서 다음처럼 제출하세요. 별도 `--output-root`는 전달하지 않습니다.

```bash
bash experiments/submit_pilot.sh mae \
  --model-size stmem-small --initialization legacy \
  --steps 1000 --seed 42 --subset-seed 42 --eval-mask-seed 42 \
  --patch-samples 50 --mask-ratio 0.75
```

기존 MAE는 `--model-size baseline`, ST-MEM은 다음처럼 실행합니다.

```bash
bash experiments/submit_pilot.sh stmem \
  --model-size small --steps 1000 --seed 42 \
  --patch-samples 50 --mask-ratio 0.75
```

| 실행 | 결과 기본 경로, clone 루트 기준 |
| --- | --- |
| 기존 MAE, single/multi-seed | `experiments/artifacts/masking_pilot/` |
| 확대 MAE, single/multi-seed | `experiments/artifacts/mae_stmem_size_pilot/` |
| ST-MEM | `experiments/artifacts/st_mem_pilot/` |

Python runner를 GPU 노드에서 직접 실행할 때도 `--output-root`를 생략하면 같은
저장소 내부 기본 경로를 사용합니다. Runner의 명시적 경로 override는 유지하지만,
새 Slurm submitter는 `--output-root`를 차단하여 항상 저장소 안에 저장합니다.
기존 서버 root의 `run_stmem.sbatch`를 직접 실행하면 그 파일에 적힌 예전 외부 경로가
사용될 수 있으므로, 앞으로는 위 submitter를 사용하세요.

기본 template은 clone 부모의 `run_stmem.sbatch`, Python은
`/data/gykwak03/envs/ecg-mask-pilot/bin/python`입니다. 경로가 다른 서버에서는
`PILOT_SBATCH_TEMPLATE`, `PILOT_PYTHON` 환경변수로 지정할 수 있습니다.
로그는 기존 서버 `logs/mae_JOBID.log` 또는 `logs/stmem_JOBID.log`로 저장합니다.

결과의 CSV·JSON·곡선 PNG는 `.gitignore`에서 허용돼 있으므로 완료 후
`git add experiments/artifacts`, `git commit`, `git push`하고 로컬에서 `git pull`합니다.
Git은 자동 commit/push하지 않습니다. Checkpoint와 NPY/NPZ 파형 배열은 계속 제외합니다.

## 기본 MAE를 ST-MEM small 크기로 확대

`run_masking_pilot.py --model-size stmem-small`은 PlainECGMAE의 encoder/decoder
폭·깊이·head 수를 ST-MEM small과 맞춥니다. `--model-size baseline` 또는 옵션 생략은
기존 smoke/pilot 크기를 사용합니다. 데이터 수와 update 수는 `--mode`/`--steps`로
별도로 정하며, 큰 모델도 `--mode smoke`로 짧게 코드 검증할 수 있습니다.

| 항목 | 기존 MAE pilot | 확대 MAE | ST-MEM small |
| --- | --- | --- | --- |
| Encoder 차원 / 층 / heads | 192 / 6 / 6 | 384 / 12 / 6 | 384 / 12 / 6 |
| Decoder 차원 / 층 / heads | 128 / 2 / 4 | 256 / 4 / 4 | 256 / 4 / 4 |
| 파라미터 수, patch 50 | 3,127,282 | 24,624,818 | 24,624,278 |

확대 MAE는 기존의 global random mask, learned lead/time embedding, global decoder,
masked-only MSE, optimizer, pooling을 유지합니다. 파라미터 수는 약 7.87배이며,
ST-MEM과의 차이는 540개입니다. 크기 선택과 실제 차원은 `run_config.json`과
checkpoint에, 크기 이름과 파라미터 수는 `results.csv`에 기록합니다.
과거 checkpoint도 저장된 profile로 재평가합니다.
Global decoder 때문에 ST-MEM과 메모리 사용량·실행 시간까지 같아지는 것은 아닙니다.

과거 pilot과 **크기만** 비교할 때는 과거와 같은 `--initialization legacy`,
patch/mask, subset/evaluation seed, batch 32, 1,000 update를 사용합니다.
기본 초기화는 계속 `independent`이며, 새 초기화로 비교하면 두 크기 모두에 같은 값을 지정하세요.

할당된 GPU 노드에서, 서버 clone의 `experiments` 폴더 기준:

```bash
/data/gykwak03/envs/ecg-mask-pilot/bin/python -u run_masking_pilot.py \
  --mode pilot --train --model-size stmem-small --initialization legacy \
  --steps 1000 --seed 42 --subset-seed 42 --eval-mask-seed 42 \
  --patch-samples 50 --mask-ratio 0.75 \
  --artifact-dir /data/gykwak03/capstone2/ptbxl_250hz \
  --output-root /data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/mae_stmem_size_pilot
```

이 실험은 모델 용량을 늘렸을 때의 효과를 확인합니다. ST-MEM과는 mask 정책·위치
embedding·decoder 구조 등이 여전히 다르므로, ST-MEM의 우위가 용량 때문인지
단독으로 확정하는 실험은 아닙니다. 로컬에서는 두 초기화의 full-size
forward/backward·가중치 갱신과 1-update smoke 학습·checkpoint 저장/재평가를 확인했습니다.

2026-10-13 서버의 patch 50 / nominal mask 75%, seed 42, legacy, 1,000 update 결과:

| 지표 | 기존 MAE | 확대 MAE | ST-MEM small |
| --- | --- | --- | --- |
| Validation masked MSE | 0.991805 | 0.991643 | 0.741784 |
| Zero 예측 대비 오차 감소 | 0.5205% | 0.5367% | 25.8676% |
| Prediction / target RMS | 0.0761 | 0.0815 | 0.5093 |
| Masked correlation | 0.0723 | 0.0737 | 0.5087 |
| Trained encoder macro AUROC | 0.7786 | 0.7860 | 0.8035 |
| Random encoder macro AUROC | 0.7726 | 0.7876 | 0.7439 |
| Pretraining AUROC gain | +0.5962 pp | -0.1598 pp | +5.9529 pp |

두 MAE의 train/validation ECG IDs, 평가 mask seed·생성 규칙·per-ECG 메타데이터,
optimizer·schedule·batch/update, 초기화 정책을 맞췄습니다. 크기만 키워서는 이
1-seed/1,000-update 조건의 복원 부진이 해소되지 않았습니다. 확대 MAE의 train도
MSE 약 0.99이므로 validation만 나쁜 overfitting 패턴이 아닙니다.
AUROC의 절대 상승은 더 큰 random encoder의 성능 상승보다 작아 SSL 이득이 확인되지 않습니다.
ST-MEM은 실제 mask 76%와 lead별 mask/decoder를 사용하므로 architecture-only 비교가 아닙니다.
남은 초기화·학습 budget·위치 embedding·mask/decoder 요인의 원인은 아직 분리하지 않았습니다.
원본 결과는 `artifacts/mae_stmem_size_pilot/20261013T140035Z_pilot_seed42_c50d99c2/`,
비교 표·곡선·검증 기록은 `artifacts/mae_capacity_analysis_20261013/`에 있습니다.

## 기본 ECG-MAE의 초기화와 결과 보관 (2026-10-13 점검)

`PlainECGMAE`는 기존 global random mask, encoder/decoder 크기, learned lead/time
embedding, masked-only MSE, optimizer와 pooling을 유지합니다. 새 실행의 기본
`--initialization independent`는 Transformer block의 attention/Linear 가중치만
층별 Xavier로 다시 초기화하여 PyTorch가 복제한 층의 초기값이 모두 같은 문제를 정리합니다.
Embedding과 reconstruction head는 기존 초기화를 유지합니다. 이 수정이 일반화 성능을
개선했다고 주장하지 않습니다. `--initialization legacy`로 이전 방식과 대조할 수 있습니다.
설정은 `profile.initialization`에 저장합니다. 해당 키가 없는 과거 checkpoint는
legacy로 복원하며, 초기화 설정이 다른 실행을 같은 seed 비교로 합산하지 않습니다.

실제 pilot 크기(encoder 192차원/6층, decoder 128차원/2층)의 모델을 CPU에서도
train ECG 2개·고정 mask·constant LR로 검사할 수 있습니다. 일반 pilot 학습의 CUDA
요구는 그대로이며 `--allow-cpu-tiny-overfit`은 이 진단에만 적용됩니다.

```bash
python -u run_masking_pilot.py --mode pilot --tiny-overfit --tiny-steps 1000 \
  --allow-cpu-tiny-overfit --initialization independent \
  --patch-samples 50 --mask-ratio 0.75 --subset-seed 42 --eval-mask-seed 42 \
  --artifact-dir /data/gykwak03/capstone2/ptbxl_250hz \
  --output-root /data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/plain_model_audit/independent
```

같은 명령에서 `--initialization legacy` 및 별도 output root로 이전 초기화와 비교합니다.
GPU 할당 노드에서는 CPU 허용 옵션 없이 실행할 수 있습니다. Tiny-overfit은 학습
ECG의 memorization 검사이며 validation/test/probe 성능의 근거가 아닙니다.
`summary.json`, `initialization.json`, 학습 곡선, initial/final 복원 예시를 저장합니다.

로컬 CPU에서 ECG ID 11/22, patch 50·mask 75%, seed 42, 1,000 update를 맞춰
실행한 결과는 다음과 같습니다. 실제 target 및 mask 배열도 동일함을 확인했습니다.

| 초기화 | 초기 masked MSE | 최종 masked MSE | 최종 예측/정답 RMS | 최종 상관계수 |
| --- | --- | --- | --- | --- |
| legacy | 1.263233 | 0.004770 | 0.992250 | 0.997599 |
| independent | 1.248040 | 0.004772 | 0.996716 | 0.997586 |

두 방식 모두 2개 ECG를 거의 외웠습니다. 따라서 이 검사에서 초기화가 기존 pilot 부진의
주원인이라는 근거는 나오지 않았으며, 수정 전 모델도 복원 과제를 학습할 능력이 있습니다.
새 초기화의 random-mask pilot 및 downstream 개선은 GPU에서 별도로 확인해야 합니다.
측정 원본은 `artifacts/model_audit_20261013/initialization_comparison.csv`,
`audit_report.json`, `initialization_comparison_curves.png`에 있습니다.
초기화·과거 checkpoint 호환성·9조건·결과 저장을 포함한 로컬 회귀 테스트는 79개 통과했습니다.

Git은 `experiments/artifacts/`의 CSV·JSON·Markdown과 학습/진단/gradient/조건 비교
곡선 PNG를 추적할 수 있습니다. Checkpoint, NPY/NPZ 파형·예측 배열, 복원 예시 PNG와
`preprocessing/artifacts/`의 원본 전처리 데이터는 계속 제외합니다. 기존 기록도
직접 add/commit/push해야 원격 저장소에 반영되며, ignore 변경은 서버 파일을 다운로드하지 않습니다.
과거 서버 명령이 저장한 `/data/gykwak03/capstone2/results/`는 clone 밖이므로,
기존 결과만 `experiments/artifacts/`로 옮겨야 Git으로 관리할 수 있습니다.
새 실행은 위 submitter 또는 저장소 내부 기본 경로를 사용하여 이 복사 과정을 생략합니다.

현재 기존 MAE의 9조건·3-seed·진단 기록, ST-MEM `st_mem_grid_v2` 9조건 기록,
확대 MAE의 1조건 pilot 기록을 받았습니다. Git에서 제외한 서버 checkpoint/NPZ까지
모두 내려받았다는 의미는 아닙니다.

## ST-MEM 기본 모델 실행 (새 경로)

`ST_MEM_pilot.ipynb`에 설명과 실행 셀이 있습니다. 서버에서는 `run_st_mem_pilot.py`를
사용합니다. 기존 custom MAE 노트북/결과와 ST-MEM 원본 clone은 변경하지 않습니다.
기존 노트북의 데이터 계약·subset·선형 probe를 재사용하므로 `Masking_pilot.ipynb`도 필요합니다.

필요한 파일은 새 notebook, `run_st_mem_pilot.py`, `st_mem_adapter.py`,
기존 `Masking_pilot.ipynb`, 최신 `masking_diagnostics.py`입니다.
모델 import만 사용하므로 기존 환경에 추가할 패키지는 `einops==0.6.0`입니다.
**upstream requirements 전체를 설치해서 현재 torch/numpy를 옛 버전으로 내리지 마세요.**
별도 clone의 `LICENSE`(CC BY-NC 4.0)와 논문/저자 attribution을 유지합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install einops==0.6.0
.\.venv\Scripts\python.exe experiments\run_st_mem_pilot.py --mode smoke --train --st-mem-repo ..\ST-MEM
```

smoke 기본 모델은 축소형이며 CPU 동작 검사만을 위한 것입니다. GPU에서 실제 공식 Small의
메모리/시간을 먼저 측정하려면 `--mode smoke --model-size small --train --steps 8`을 사용하세요.
데이터 subset/probe도 작은 smoke 설정이며 연구 근거가 아닙니다.

Windows PowerShell에서 최신 실험 파일 전송(데이터는 다시 전송하지 않음):

```powershell
$stmemExperiments = "C:\Users\gykwa\kyunghee2026\capstonedesign2\predictability-guided-ecg-masking\experiments"
scp -P 30080 "$stmemExperiments\ST_MEM_pilot.ipynb" `
    "$stmemExperiments\run_st_mem_pilot.py" `
    "$stmemExperiments\st_mem_adapter.py" `
    "$stmemExperiments\Masking_pilot.ipynb" `
    "$stmemExperiments\masking_diagnostics.py" `
    "gykwak03@moana.khu.ac.kr:/data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/"
```

서버 예시: `/data/gykwak03/capstone2/ST-MEM`에 upstream을 별도 clone하고,
기존 `ecg-mask-pilot` 환경에서 `python -m pip install einops==0.6.0`을 실행합니다.
그 다음 **GPU 할당 노드에서** 실행하거나 아래처럼 master에서 `sbatch` 제출하세요.
아래 명령은 잡을 자동 제출하지 않으며, 사용자가 복사해 제출해야 합니다.
GPU 메모리 검사는 아래 `--wrap` 안에서 `--mode pilot`을 `--mode smoke`로,
`--steps 1000`을 `--steps 8`로 바꿔 먼저 제출하세요. `--model-size small`은 유지합니다.

```bash
mkdir -p /data/gykwak03/capstone2/logs
sbatch --account=ugrad_advisor_x \
  --partition=debug_ugrad_advisor_x \
  --job-name=ecg-stmem \
  --nodes=1 --ntasks=1 --nodelist=moana-u2 \
  --gres=gpu:normal:1 --cpus-per-task=4 --mem=16G \
  --time=04:00:00 \
  --chdir=/data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments \
  --output=/data/gykwak03/capstone2/logs/stmem_%j.log \
  --wrap="srun /data/gykwak03/envs/ecg-mask-pilot/bin/python -u run_st_mem_pilot.py --mode pilot --train --model-size small --batch-size 4 --accum-steps 8 --steps 1000 --patch-samples 50 --mask-ratio 0.75 --st-mem-repo /data/gykwak03/capstone2/ST-MEM --artifact-dir /data/gykwak03/capstone2/ptbxl_250hz --output-root /data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/st_mem_pilot"
```

4시간은 실행 상한이며 예상 소요 시간이 아닙니다. 실제 GPU 실행·VRAM은 아직 확인하지 않았습니다.
메모리가 부족하면 `--batch-size 2 --accum-steps 16`으로 유효 batch 32를 유지할 수 있습니다.
첫 GPU smoke에서 소요 시간과 `gpu_peak_allocated_mb`를 확인하세요.
Runner는 한 번에 한 조건을 실행합니다. 여러 조건은 아래처럼 별도 잡으로 제출하며,
자동 best 선택은 하지 않습니다.

### ST-MEM 9조건 비교

서버에 앞서 준비한 `run_stmem.sbatch`(마지막에 `"$@"`를 전달하는 제출 스크립트)가 있으면
다음 명령으로 patch 50/125/250 × masking 50/75/90%의 9조건을 제출합니다.
각 조건은 같은 5,000 ECG subset, seed 42, 1,000 update를 사용합니다.
전체 train 데이터로 학습하는 명령이 아니며, 조건마다 새 실행 폴더에 저장합니다.

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments
for patch in 50 125 250; do
  for ratio in 0.50 0.75 0.90; do
    sbatch --job-name="stmem-p${patch}-r${ratio}" run_stmem.sbatch \
      --mode pilot --steps 1000 --seed 42 --subset-seed 42 --eval-mask-seed 42 \
      --patch-samples "$patch" --mask-ratio "$ratio"
  done
done
```

수정한 adapter와 runner를 서버에 다시 전송한 뒤 제출하세요.

| Patch samples | 설정 50%의 실제 비율 | 설정 75%의 실제 비율 | 설정 90%의 실제 비율 |
| --- | --- | --- | --- |
| 50 | 50% | 76% | 90% |
| 125 | 50% | 75% | 90% |
| 250 | 50% | 80% | 90% |

이전 binary-float 계산으로 실행한 patch 50/90%, 125/90%는 실제 92%, 95%였습니다.
수정 후 각각 90%로 실행되며, patch 250/90%도 lead당 1개를 보여주어 실행 가능합니다.
새 `run_config.json`의 `policy`는 `per-lead-uniform-decimal-floor-visible-count-v2`입니다.
이전 결과는 보존하고 실제 비율과 policy가 다른 실행을 같은 조건으로 합산하지 마세요.

### ST-MEM과 이전 실험의 차이

- 공식 Small: encoder 384차원/12층/6heads, decoder 256차원/4층/4heads.
  `--model-size base`도 지원하지만 12GB에서 실행되는지는 검증하지 않았습니다.
- **lead별 random mask + lead-wise decoder**가 공식 구조입니다.
  이전의 global mask와 다르므로 architecture만 교체한 비교는 아닙니다.
- 각 lead의 visible 개수는 `floor(time_patches*(1-mask_ratio))`입니다.
  Decimal로 계산해 정수 경계의 소수점 오차를 제거하며, 원본 binary-float 동작과는 다릅니다.
  patch 50/mask 75%는 lead당 50개 중 12개만 보여서 **실제 mask는 76%**입니다.
  `run_config.json`/`results.csv`에 설정값과 실제 비율을 따로 기록합니다.
- 입력 2500 sample, patch 50은 기존 데이터에 맞춘 변경입니다.
  원본의 2250 sample/patch 75와 추가 전처리는 적용하지 않습니다.
- **`norm_pix_loss=False`**로 기존 전처리 파형을 직접 복원합니다.
  원본 pretrain YAML의 True와 다르므로 논문 설정 그대로의 재현은 아닙니다.
- 공식 pooling(SEP 제외 후 평균 → 최종 LayerNorm)과 고정 encoder의 5-class BCE 선형 probe를 사용합니다.
  같은 초기 가중치의 미학습 encoder에도 동일한 분할/표준화/probe 설정을 적용합니다.
- microbatch 4 × accumulation 8 = 유효 batch 32, 총 1,000 optimizer update입니다.
  encoder/decoder 구조가 달라 이전과 같은 연산량·시간은 아닙니다.
- 기존 0.67–40 Hz 필터·lead별 z-score를 유지하며 필터/정규화를 중복 적용하지 않습니다.
- 공개 사전학습 가중치를 자동으로 받지 않습니다. PTB-XL train부터 새로 학습합니다.

### 저장 결과와 중단

매번 새 `*_stmem_*` 폴더에 초기/최종 복원 PNG·NPZ, 학습/고정 평가 곡선,
gradient CSV/PNG, 0/선형 복원 기준, 초기/중간 latest/최종 checkpoint,
학습/미학습 encoder의 분류 확률·클래스별 지표·특징 통계를 저장합니다.
고정 sin-cos 위치 encoding에 gradient가 없는 것은 공식 모델의 정상적인 동작입니다.
`--no-probe`는 복원만 실행하므로 분류 효과의 근거로 사용하지 마세요.
`--tiny-overfit --tiny-steps 200`은 train ECG 2개·고정 mask·고정 학습률을 사용하는 별도 진단입니다.
`--train`과 함께 사용하지 않으며, memorization과 일반화를 구분합니다.
신뢰할 수 있는 checkpoint만 사용하고, 기존 `inspect_masking_pilot.py`에는 전달하지 마세요.
ST-MEM checkpoint는 `config`와 `step`을 갖는 별도 형식이며 optimizer/RNG resume는 지원하지 않습니다.

주 파일은 `Masking_pilot.ipynb`입니다. 기존 preprocessing notebook 및 산출물을 수정하지 않습니다.
실험은 train/validation만 사용합니다. test는 열지 않습니다.

## 로컬 CPU

기존 저장소 `.venv` 커널에서 notebook을 위에서 아래로 실행하세요.
기본값은 `MODE="smoke"`, `RUN_TRAINING=False`라서 patch/mask 검증과 시각화만 합니다.
`RUN_TRAINING=True`로 바꾸면 64 ECG, 작은 모델, 8 update의 동작 검사만 실행합니다.
이 결과로 최적 patch length나 mask ratio를 고르지 마세요.

필요한 학습 패키지는 torch와 scikit-learn입니다. GPU 서버에 CPU 전용 torch를 설치하지 마세요.
로컬에서는 저장소 루트에서 다음 명령으로도 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe experiments\run_masking_pilot.py --mode smoke --train
```

## Seraph 등 GPU 노드

계정/사용 권한/스케줄러 및 GPU 할당은 학교 안내를 확인해야 합니다.
공식 연구실 페이지는 Seraph를 공유 GPU 클러스터로 소개하지만 특정 접속·제출 명령은 확인하지 않았습니다.
로그인 노드에서 학습하지 말고 할당된 GPU 노드에서 CUDA 사용 가능 여부를 확인하세요.

1. `Masking_pilot.ipynb`, `run_masking_pilot.py`, `masking_diagnostics.py`와 기존
   `preprocessing/artifacts/ptbxl_250hz` 폴더 **전체**를 서버로 옮깁니다.
   데이터가 이미 서버에 있다면 다시 전송하지 않아도 됩니다.
2. 서버에 맞는 CUDA PyTorch, numpy, pandas, matplotlib, scikit-learn, IPython을 준비합니다.
3. 먼저 `--mode smoke --train`으로 동작을 확인합니다.
4. 다음 명령의 예시 경로를 실제 읽기/쓰기 가능한 경로로 바꾸고 GPU pilot을 실행합니다.

```bash
python experiments/run_masking_pilot.py --mode pilot --train \
  --artifact-dir /your/path/ptbxl_250hz \
  --output-root experiments/artifacts/masking_pilot --seed 42
```

Notebook에서는 `MODE="pilot"`, `RUN_TRAINING=True`, `ARTIFACT_DIR_OVERRIDE`, 필요시
`OUTPUT_ROOT_OVERRIDE`를 설정하고 위에서 아래로 다시 실행하면 같은 코드를 사용합니다.
CUDA GPU가 없으면 pilot 실행을 기본적으로 차단합니다.

## 비교 설계

- Patch: 50/125/250 samples = 0.2/0.5/1초, 겹치지 않는 lead × time 단위.
- Mask ratio: 50/75/90%, 전체 lead-time 격자에서 정확한 개수를 무작위 선택.
- Pilot: 9조건, 공통 5,000 ECG subset, batch 32, 조건별 1,000 update, seed 42.
- Encoder: 6층/192차원/6 heads. Decoder: 2층/128차원/4 heads.
- Pretraining: visible-only encoder, masked-only MSE, AdamW/warmup/cosine.
- Probe: frozen encoder, 전체 token 평균 pooling, train 특징 통계로 표준화, 선형층만 학습.
- 평가: validation macro AUROC/AUPRC, 클래스별 결과, 고정 threshold 0.5의 F1.
- 같은 update/ECG 노출 수이지 같은 FLOPs는 아닙니다. 조건별 시간/메모리도 저장합니다.
- 현재 입력 산출물은 **0.67–40 Hz, ECG별·lead별 z-score**입니다. 노션의 50 Hz 계획으로
  조용히 변경하지 않습니다. 입력 manifest/hash/ECG ID를 결과에 기록합니다.

매 실행은 새 폴더를 만들고 설정, subset ID, 조건별 학습 곡선, validation 예측,
클래스별 지표, checkpoint 및 `results.csv`를 저장합니다. 자동 resume은 지원하지 않습니다.
Checkpoint는 자신이 생성한 신뢰할 수 있는 파일만 로드하세요.

각 조건의 학습과 평가가 끝나면 해당 조건 폴더(예: `p50_r75/`)에 다음을 자동 저장합니다.

- `pretrain_history.csv`: 매 update의 `step`, `masked_mse`, `grad_norm`, `lr` 원본 기록.
- `learning_curves.png`: loss 원본과 trailing 50-update 이동평균, gradient norm, 학습률의 3-panel 그래프.

이동평균은 시작 구간에서 확보된 update만 사용하며, smoke처럼 50 update보다 짧으면
창을 전체 기록 길이로 줄입니다. Gradient norm은 **clipping 전** 값이고 점선은 clipping 기준 1.0입니다.
기존 CSV의 학습률 기록 시점은 유지합니다. `lr`은 `scheduler.step()` 직후 값이므로 **다음 update의 학습률**입니다.
`learning_curves.png`는 서로 다른 학습 batch의 loss입니다. 고정 샘플의 train/validation
복원 곡선은 진단 활성화 시 아래의 `diagnostic_curves.png`에 별도로 생성합니다.
저장용 그래프는 Jupyter 화면이나 GUI가 없어도 runner/`sbatch` 실행에서 생성됩니다.
학습 도중에도 진단 주기마다 CSV를 갱신합니다. 곡선 PNG는 조건이 완료되면 저장합니다.
중단된 조건의 CSV는 마지막으로 저장된 진단 주기까지만 있고, 이어 학습하는 resume은 없습니다.

## 왜 학습이 약한지 확인하는 진단 (기본 활성화)

모델 초기화/architecture, optimizer, 새 training mask, subset, batch/update 수는 바꾸지 않습니다.
`DIAGNOSTICS` 설정은 notebook 6번째 셀, 학습과 진단 호출은 18번째 셀, 최종 저장은 20번째 셀에 있습니다.
외부 helper `masking_diagnostics.py`는 평가/통계/그림 및 별도 tiny-overfit만 담당합니다.
진단을 끄려면 runner에 `--no-diagnostics`, 추가 random probe만 생략하려면 `--no-random-probe`를 붙이세요.

| 확인할 문제 | 저장 파일 / 읽는 방법 |
| --- | --- |
| train도 못 맞추는지, train만 좋아지는지 | `evaluation_history.csv`, `diagnostic_curves.png`: 0·200·400…·최종 update에 **같은** train/validation 각 최대 32 ECG·고정 mask 1회 평가. Train subset은 실제 학습 subset 안에 있음. |
| 0 근처로 수축하는지 | `final_reconstruction_metrics.json`: masked MSE, `zero_masked_mse`, `skill_vs_zero = 1 − MSE/zero MSE`, 예측/정답 RMS·표준편차·비율·bias·상관계수. **가린 sample만** 계산. RMS가 작고 skill이 0에 가까우면 진폭 수축을 의심. |
| 실제로 어떤 파형을 복원하는지 | `initial_reconstruction_samples.npz`, `final_reconstruction_samples.npz`, `initial/final_reconstruction_ecg*.png`: validation 앞 3 ECG의 정답·raw prediction·patch/sample mask·합성 파형·ECG ID·mask seed·lead 순서·fs 저장. PNG는 가린 구간의 예측만 표시. 진폭은 **z-score이며 mV가 아님**. |
| 특정 lead/ECG/마스크가 어려운지 | `final_reconstruction_per_lead.csv`, `final_reconstruction_per_ecg.csv`: 최종 평가 전체, repeat별 ECG 오차·진폭·상관계수·완전히 가린 lead 수·모든 lead가 가려진 time-patch 수. 마스크 단위는 기존 global uniform 그대로. |
| 입력에 복원 가능한 정보가 있는지 | `cross_lead_baseline.npz`, 최종 metrics의 `linear_masked_mse`: train subset 최대 256 ECG로만 fit한 covariance/ridge(1e-3)를 이용해 같은 시간의 visible lead에서 hidden lead를 선형 예측. **가린 target/validation 통계는 fit·예측 입력에 사용 안 함**. visible lead가 전혀 없으면 train 평균. lead redundancy를 쓰는 baseline으로, MAE 표현 학습의 성공을 뜻하지 않음. |
| gradient가 끊기거나 가중치가 안 바뀌는지 | `gradient_updates.csv/png`: 1·50·100…·최종 update에 patch embed, encoder/decoder 각 층, positional parameters, reconstruction head 등의 clipping 전 gradient norm, grad 없는 parameter 수, 실제 update norm과 weight norm 대비 비율. `pretrain_history.csv`의 `clipped`는 전체 clipping 발생 여부. |
| SSL 없이도 나온 분류 성능인지 | `initial_checkpoint.pt`, `random_encoder_probe_metrics.json`, `random_encoder_probe.pt`, `random_encoder_probabilities.npy`, `results.csv`의 `auroc_gain_vs_random`: **동일한 학습 전 가중치**를 가진 random encoder에 같은 subset/feature 표준화/probe seed/update로 대조 실험. 별도 RNG 보존으로 기존 학습을 교란하지 않음. |
| 표현이 비슷한 상수로 붕괴하는지 | `feature_statistics.json`: probe train의 feature 차원별 std, near-constant 차원 수, centered covariance effective rank, ECG 간 평균 cosine. 작은 subset의 rank 한계에 유의; 단일 지표로 collapse 확정 불가. |
| 초기 layer 값이 같은지 | `initialization.json`: encoder/decoder layer 초기값 동일 여부. 새 independent 실행은 층별 초기값을 분리하며 legacy 실행은 과거 방식을 유지. 동일 초기값은 공유 parameter나 실패 원인의 확정이 아님. |

최종 주 결과는 기존과 같은 validation 최대 128 ECG × 3 mask repeats입니다.
**mask seed뿐 아니라 RNG 값의 argsort 방식까지 기존 평가와 일치**하도록 했습니다.
NPZ 예시는 이 평가의 repeat 0 앞 3 ECG이고, 가중 평균한 전체 최종 지표를 모두 대표하지는 않습니다.
`composite`는 visible input을 복사한 확인용 합성 파형입니다. 이 파형 전체의 MSE를 복원 성능으로 보고하지 않습니다.
정답/예측의 mask 영역이 상수라 Pearson correlation을 정의할 수 없으면 CSV는 빈 값, JSON은 `null`입니다.

`diagnostic_summary.json`은 weak reconstruction, amplitude shrinkage, small random-probe gain 등을
**디버깅용 휴리스틱**으로 표시할 뿐 원인을 자동 확정하지 않습니다. Test split은 여전히 열지 않고,
validation 곡선으로 best/early stopping checkpoint를 고르지 않습니다. 최종 checkpoint만 평가합니다.
중간 모델 checkpoint는 따로 없으며, `initial_checkpoint.pt`는 random-probe 설정이 켜져 있을 때 저장합니다.

추가 평가, 원본 파형 PNG 저장, 두 번째 feature extraction/probe 때문에 예전보다 느립니다.
`results.csv`에 `diagnostic_seconds`, `random_probe_seconds`, `pretrain_wall_seconds`, `condition_seconds`를 기록합니다.
`train_seconds`는 periodic 평가 시간을 제외하지만 per-update 진단 수집 비용은 포함하므로 예전과 정확한 속도 비교는 안 됩니다.
`condition_seconds`는 condition 함수까지의 시간이며 마지막 learning-curve/checkpoint 저장 및 run 전체 준비 비용은 제외합니다.
`gpu_peak_allocated_mb`도 진단용 snapshot/추가 모델·평가까지 포함한 allocator peak이므로 기존 기록과 직접 비교하지 마세요.
RNG/model state 보호는 CPU 테스트로 검증했고 GPU 수치·메모리는 할당 노드에서 smoke로 확인해야 합니다.

PyTorch의 layer 초기값 clone과 clipping 의미는
[TransformerEncoder 문서](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.transformer.TransformerEncoder.html),
[clip_grad_norm_ 문서](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html)를 참고합니다.

## 별도 tiny-overfit / 더 긴 학습

먼저 **할당된 GPU 노드**에서 모델이 ECG 2개와 고정 mask를 반복해 맞출 수 있는지 확인할 수 있습니다.
아래는 pilot과 같은 모델 크기지만, constant LR 3e-4·고정 train ECG 2개·고정 mask의 **별도 진단**입니다.
기존 warmup/cosine pilot·validation·probe와 섞지 않습니다. 작은 학습 샘플을 외우는 결과이지 일반화 증거가 아닙니다.
새 `*_tiny_overfit_*` 폴더에 initial/final 파형·MSE·곡선·checkpoint·`summary.json`을 저장합니다.

```bash
python -u run_masking_pilot.py --mode pilot --tiny-overfit --tiny-steps 200 \
  --patch-samples 50 --mask-ratio 0.75 \
  --artifact-dir /data/gykwak03/capstone2/ptbxl_250hz \
  --output-root /data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/masking_diagnostics
```

CPU에서는 `--mode smoke`로 작은 모델을 검사할 수 있지만, full pilot 모델 검증을 대체하지 않습니다.
`--tiny-overfit`과 `--train`은 함께 쓰지 않습니다. `--tiny-steps`는 tiny의 update 수,
`--steps`는 regular smoke/pilot의 update 수입니다.

먼저 50 samples/75% 한 조건을 기존 1,000 update로 다시 실행해 진단을 보세요:

```bash
python -u run_masking_pilot.py --mode pilot --train \
  --patch-samples 50 --mask-ratio 0.75 \
  --artifact-dir /data/gykwak03/capstone2/ptbxl_250hz \
  --output-root /data/gykwak03/capstone2/predictability-guided-ecg-masking/experiments/artifacts/masking_diagnostics
```

`--steps 3000`을 추가하면 해당 조건을 더 오래 학습합니다. 이때 schedule도 3,000 update 길이에 맞추므로
1,000 update checkpoint의 단순 연장이 아니라 새 학습입니다. 정식 grid 비교에서는 모든 조건에 같은 수를 사용하세요.
`--patch-samples`/`--mask-ratio`를 생략하면 기존 9조건을 실행합니다. 조건 지정은 후보 검증이지 최적 mask 확정이 아닙니다.

기존 실행의 CSV를 다시 그리려면 다음 명령을 사용할 수 있습니다(재학습하지 않습니다).
이 보조 스크립트는 조건 비교용 `training_curves.png`와 마지막 20% 확대용 `training_curves_tail.png`를 저장합니다.

```powershell
.\.venv\Scripts\python.exe experiments\plot_masking_pilot.py "저장된 run 폴더 경로"
```

1-seed 단기 pilot은 최종 결론이 아닙니다. 상위 두 조건은 동일 모델의 더 긴 학습 및
seed 3개에서 재검증하세요. MSE는 조건별 과제 난이도가 달라 최적 설정 선택 기준이 아닙니다.
Easy/Medium/Hard 및 predictability-aware 실험은 이 pilot 이후 단계입니다.

## 기본 조건을 선택하기 위한 비교

MSE의 절대값으로 적당한 난이도를 정하지 않습니다. `validation_skill_vs_zero`는 같은
ECG/가린 sample에서 `1 - model_MSE / zero_MSE`를 계산한 값입니다. 0이면 0 예측과 같고,
음수이면 더 나쁩니다. `auroc_gain_vs_random` / `auprc_gain_vs_random`은 동일한 초기 encoder와
동일한 라벨 표본/probe 학습을 사용한 대조군 대비 개선입니다. 통계와 파형은 학습 진단용이며,
기본 조건은 validation 분류 성능·seed 변동·계산 비용을 함께 보고 선택합니다. Test는 사용하지 않습니다.

### 기존 checkpoint 분석: 복원 학습 다시 실행하지 않음

`inspect_masking_pilot.py`와 최신 notebook, runner, `masking_diagnostics.py`를 함께 옮기세요.
서버의 `experiments` 폴더에서 기존 실행 폴더를 지정합니다.

```bash
python -u inspect_masking_pilot.py \
  artifacts/masking_pilot/20261008T010245Z_pilot_seed42_33f3976c \
  --conditions p50_r75 p50_r90 \
  --artifact-dir /your/path/ptbxl_250hz
```

기존 `run_config.json`의 ECG ID, 모델 크기, 평가 마스크 seed, 전처리 manifest hash를 복원합니다.
새 inspection 폴더에 `results.csv`, 조건별 `final_reconstruction_samples.npz`,
12-lead 복원 PNG, zero/cross-lead baseline 지표와 분류 대조군 결과를 저장합니다.
MAE 가중치는 갱신하지 않고, 공정한 비교를 위해 trained/random encoder의 선형 분류층만 각각 새로 학습합니다.
`--no-probe`이면 복원 진단만 수행합니다. `--batch-size 8`은 추론용 batch만 줄입니다.
초기 checkpoint가 없는 옛 실행은 seed와 현재 모델로 초기값을 재생성하므로,
`random_initialization_source=regenerated-current-model`을 명시합니다. 과거 초기값과 동일함을 보장하지 않습니다.
중간 checkpoint가 없었던 실행의 validation 학습 곡선은 사후 복구할 수 없습니다.
기존 results/checkpoint를 덮어쓰지 않습니다.

### 새 학습: 동일 표본·동일 검증 마스크에서 seed 비교

```bash
python -u run_masking_pilot.py --mode pilot --train --seeds 42 43 44 \
  --subset-seed 42 --eval-mask-seed 42 \
  --patch-samples 50 --mask-ratios 0.75 0.90 \
  --artifact-dir /your/path/ptbxl_250hz
```

위 명령은 두 후보를 같은 suite에서 비교합니다. 한 조건만 보려면 `--mask-ratio 0.75`를 사용합니다.
`--patch-samples`/`--mask-ratios`를 생략하면 전체 9조건을 실행합니다.
`--seeds`는 `--train`과 diagnostics/random-probe 활성화를 요구합니다.
기본적으로 첫 seed로 ECG subset과 평가 mask를 고정하고, 모델 초기화·학습 샘플링·probe seed만 바뀝니다.
원래 단일 `--seed` 실행은 기존 기본 동작을 유지합니다.

새 학습에서는 0 update와 매 200 update 및 마지막에 고정 train/validation ECG·mask를 평가합니다.
`evaluation_history.csv`/`diagnostic_curves.png`, initial/final 파형, matched random-probe 결과를 저장합니다.
기존 모델 크기와 1,000-update 학습 설정을 사용하며 `--steps`로 모든 조건의 학습량을 함께 변경할 수 있습니다.

각 seed가 완료될 때 suite 폴더의 `seed_results.csv`, `condition_summary.csv`,
`condition_comparison.png`를 갱신합니다. 표준편차는 **seed 간 표본 표준편차**이고 신뢰구간이 아닙니다.
1 seed의 표준편차는 미정(빈 값)이며, 미완료 seed는 `complete_requested_seeds=False`로 표시합니다.
자동 최적값 선택이나 통계적 유의성 판정은 하지 않습니다.

서로 다른 suite의 실행을 함께 비교할 때 각 seed 실행 폴더를 지정하세요.

```bash
python pilot_comparison.py /path/to/run_seed42 /path/to/run_seed43 /path/to/run_seed44 \
  --output-dir /path/to/comparison
```

표본·평가 mask·모델/학습/평가 설정이 다른 실행이나 같은 조건/seed 중복은 합산을 차단합니다.

## Mask predictability: Ridge / 1D CNN / Small Transformer

`run_predictability.py`는 세 estimator 학습부터 공통 mask 평가, lead × time map,
Spearman·consensus·disagreement, 전체 mask의 Easy/Medium/Hard 3등분까지 실행합니다.
모델·지표·난이도 함수는 `predictability_estimators.py`에 있습니다.

기본 조건은 **P=50 samples (250 Hz에서 0.2초), masking ratio=0.50**입니다.
ST-MEM의 이 조건은 validation MSE=0.3913, macro AUROC=0.8068이고,
기본 MAE의 AUROC=0.7752도 해당 grid 최고값 0.7799에 가까워 두 모델에 공통으로 쓸
탐색 조건으로 선택했습니다. ST-MEM의 AUROC만 보면 P125/r50=0.8138이 더 높지만
이때 MSE=0.4350이고 기본 MAE는 P125에서 AUROC가 낮았습니다.
MSE와 AUROC가 비례한다거나 이 조건이 최종 최적값이라는 의미는 아닙니다.
`--patch-samples 50/125/250` 및 `--mask-ratio`로 바꿀 수 있고 실제 hidden 비율도 기록합니다.

### 데이터·입력·학습 규칙

- 원래 train folds 1–8의 **환자**를 seed 기반 80/20으로 나눕니다. 80%에서 estimator-fit ECG를,
  20%에서 mask score ECG를 선택하므로 두 pool의 환자가 겹치지 않습니다.
  기본 pilot은 fit 5,000 ECG, score 512 ECG이며, validation fold 9의 128 ECG는 학습 진단만 합니다.
  Test 배열은 열지 않습니다. fit/score/validation ECG 수 `0`은 각 reserved pool 전체를 뜻합니다.
- fit ECG당 4개, score ECG당 8개의 random **전체 lead-time mask**를 만듭니다.
  P50/r50에서는 600 patch 중 정확히 300개를 가립니다. 특정 lead 하나만 가리는 과제가 아닙니다.
  mask마다 ECG ID·repeat·seed·bit digest를 CSV에 기록하며 순서를 바꿔도 동일 mask를 재생성할 수 있습니다.
  학습 및 평가 mask bank는 세 estimator가 공유하고 CNN/Transformer는 같은 batch 순서를 사용합니다.
- 모든 estimator가 내부에서 hidden 값을 0으로 제거하고 explicit visibility를 입력에 포함합니다.
  mask 변경 없이 hidden 값만 바꾸어도 예측은 같아야 합니다. Loss와 지표는 hidden sample만 사용합니다.
- Ridge는 train target이 가려진 sample만으로 lead별 선형 회귀를 학습합니다. 입력은 다른 lead의 같은 시점,
  전체 lead의 lag `[-2P, -P, -1, 0, 1, P, 2P]` visible 값·availability·intercept입니다.
  시간 전체에서 계수를 공유합니다. 값·availability에 대한 선형 모델이며 mask마다 계수를 새로 fit하는 모델은 아닙니다.
  기본 각 fit mask에서 32개 time point를 추출하며 정규화된 목적함수는
  `mean(masked squared error) + alpha * ||w||²`, alpha=1e-3, intercept 제외입니다.
  CNN·Transformer처럼 모든 time target을 매 update 사용하는 것은 아니며 이 차이를 config에 기록합니다.
- CNN은 12 lead를 channel로 보고 visible/availability 24 channel에서 width 64,
  kernel 7 residual convolution 4개(dilation 1/2/4/8)를 사용합니다.
  Small Transformer는 raw patch+visibility를 width 128로 투영해 4 block/4 head의 전역 attention,
  fixed sinusoidal time position 및 learned lead embedding으로 복원합니다. ST-MEM/기본 MAE와는 별도 estimator입니다.
- neural pilot 기본 batch 16, 각 3,000 update, AdamW 3e-4, weight decay 1e-4,
  warmup 후 cosine LR, gradient clip 1입니다. 학습 곡선은 저장하지만 validation으로 checkpoint를 선택하지 않습니다.
  마지막 모델을 평가합니다. 이것은 동일 학습 budget의 첫 탐색이며 estimator별 충분한 학습을 보장하지 않습니다.

NRMSE 분모 `sigma_c`는 **선택된 estimator-fit ECG만** 이용해 lead별로 ECG×time의
population std를 한 번 계산한 값입니다. target patch 자체의 std로 나누지 않습니다.
현재 입력은 per-record/lead z score이므로 std는 대략 1이며 mV 단위가 아닙니다.
전체 mask NRMSE는 `sqrt(mean_hidden_samples(((prediction-target)/sigma_c)^2))`,
lead-time patch NRMSE는 해당 lead patch의 같은 계산입니다.
masked R²/Pearson도 lead별 std로 표준화한 hidden sample을 pooling하며 평탄 target의 미정 값은 빈 값/null입니다.

### 순위·난이도·map 정의

각 estimator의 **score ECG-mask 전체 분포**에서 ascending NRMSE average rank를 계산하고
`predictability = 100 * (N - rank + 0.5) / N`으로 변환합니다. 작을수록 낮은 오차,
높을수록 쉬운 mask입니다. 동률은 동일 percentile이고 모든 값이 같으면 50입니다.
세 percentile의 동일 가중 평균이 consensus, population std 및 max-min이 disagreement입니다.
세 estimator의 결과가 모두 있어야 consensus를 생성하며 누락·NaN error·중복 결과는 차단합니다.
Spearman은 동일 mask들의 NRMSE 순위 간 rho이며 상수 입력은 미정입니다.
한 ECG의 여러 mask가 종속이므로 단순 mask 수를 독립 표본으로 간주한 p-value는 제공하지 않습니다.

Consensus 내림차순으로 전체 score mask를 정렬해 Easy/Medium/Hard로 3등분합니다.
그룹 크기는 최대 1개 차이이며 consensus 동률은 mask_id로 순서를 정합니다.
경계에서 동률 mask가 나뉘면 난이도 차이의 증거가 아니므로 tie 수를 함께 기록합니다.
전역 3등분은 ECG·class·lead 구성을 맞추지 않으므로 그룹별 구성 및 ECG coverage CSV를 꼭 확인하세요.
후속 MAE 비교에서는 각 그룹이 같은 ECG를 충분히 공유하도록 표본을 맞추고,
같은 backbone·총 masked tokens·updates·optimizer·probe로 비교해야 합니다.
모든 train ECG의 mask를 이 코드의 score subset 결과로 정의할 수는 없습니다.
fit ECG까지 포함해 난이도를 정의하려면 추가 patient cross-fitting이 필요합니다.

Map은 **score mask에서 해당 lead/time patch가 hidden이었던 경우만** 집계합니다.
각 estimator의 전체 hidden **patch** NRMSE 분포에서 percentile을 구하고 lead×time 평균을 그립니다.
전체 mask 난이도용 percentile과 patch-map용 percentile은 서로 다른 ranking population입니다.
빈 coverage는 미정이며 visible-copy 점수를 넣지 않습니다. consensus map은 세 map의 평균입니다.
개별 mask의 hidden patch map 예시도 저장합니다.

### 실행과 결과

로컬 CPU에서 코드 흐름을 확인합니다. smoke 결과는 연구의 난이도 확정 근거가 아닙니다.

```powershell
.\.venv\Scripts\python.exe -u experiments\run_predictability.py --mode smoke
```

서버에서 clone 루트로 이동하고 최신 코드를 받은 뒤 GPU 작업으로 제출합니다.
`submit_pilot.sh`는 기존 `/data/gykwak03/capstone2/run_stmem.sbatch`의 GPU·partition·account 지시문을
재사용하며 결과는 clone 내부에 저장합니다. 기존 sbatch 본문의 ST-MEM 실행 코드는 사용하지 않습니다.

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
git pull
bash experiments/submit_pilot.sh predictability \
  --patch-samples 50 --mask-ratio 0.50 \
  --seed 42 --subset-seed 42 --mask-seed 42 \
  --steps 3000 --fit-ecgs 5000 --score-ecgs 512 \
  --train-masks 4 --score-masks 8
```

로그는 `../logs/predictability_JOBID.log`, 결과는
`experiments/artifacts/predictability/<UTC timestamp>_pilot_seed42_<id>/`에 저장됩니다.
결과 파일은 다음과 같습니다.

| 파일 | 내용 |
| --- | --- |
| `run_config.json`, `ecg_selection.csv`, `*_mask_bank.csv` | 전처리 hash·고정 train std·pool/seed·공통 mask 재현 정보 |
| `mask_metrics.csv`, `estimator_summary.csv` | mask별 NRMSE/R²/Pearson/0 예측 대비 개선 및 pooled 요약 |
| `*_learning_history.csv`, `estimator_learning_curves.png` | 0 update~final validation NRMSE·skill |
| `mask_difficulty.csv` | mask별 세 NRMSE·percentile·consensus·disagreement·Easy/Medium/Hard |
| `easy_masks.csv`, `medium_masks.csv`, `hard_masks.csv` | 후속 실험용 mask ID·ECG ID·재현 seed 목록 |
| `spearman_correlations.csv`, `difficulty_definition.json` | estimator rank agreement·난이도 정의·tie/분포 진단 |
| `difficulty_groups.csv`, `ecg_group_coverage.csv`, `difficulty_lead_coverage.csv` | 그룹별 ECG·class·lead 구성 |
| `lead_time_predictability.csv`, `lead_time_predictability_curves.png` | lead×time 평균 map·hidden 관측 수 |
| `example_predictability_curves.png` | 공통 평가 mask 하나의 hidden patch map |
| `completion.json` | 완료 여부·총 시간·품질 지표·공통 첫 batch 및 전체 train sampling sequence hash |

CSV/JSON/curves PNG는 현재 Git ignore 규칙상 저장할 수 있습니다. `.pt`/`.npz` weights,
packed masks 및 patch error cache는 Git에서 제외됩니다. CSV seed/digest로 mask를 재현할 수 있습니다.
후속 코드에서는 `predictability_estimators.masks_from_bank(pd.read_csv(bank_path), 600)`으로
P50 mask bank를 복원하고 digest를 검증할 수 있습니다. P125는 240, P250은 120 token입니다.
서버에서 `git add experiments/artifacts/predictability`, commit/push 후 로컬 pull하면 작은 결과가 전달됩니다.
분류 AUROC는 이 estimator 단계에서 계산하지 않습니다. 후속 동일 MAE의 난이도별 표현 학습 실험에서 평가합니다.

선형 회귀 목적함수와 rank 상관 정의는 [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) 문서를 참고합니다.

## 같은 ECG 내부의 patch 난이도 (50% masking)

`run_patch_difficulty.py`는 기존 predictability checkpoint를 고정하고, 같은 ECG에
독립적인 discovery/confirmation 50% mask context를 반복 적용합니다. Patch 자체의
복원 난이도와 ‘해당 patch가 포함된 전체 mask가 더 어려운가’를 따로 검증합니다.
추가 학습·test 배열 접근 없이 실행하며, GPU pilot은 다음처럼 제출할 수 있습니다.

```bash
bash experiments/submit_pilot.sh patch-difficulty \
  --source-run "$PWD/experiments/artifacts/predictability/20261014T050200Z_pilot_seed42_eb872a85" \
  --score-ecgs 64 --pairs-per-bank 32 --seed 42
```

Checkpoint 세 개가 원래 run 폴더에 있어야 합니다. 출력 정의, CPU smoke 명령,
고정된 pilot 판정 기준과 해석 범위는 [실험 설명](PATCH_DIFFICULTY.md)에 있습니다.
현재 코드 검증과 실제 ECG의 과학적 검증을 구분합니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

로컬 전처리 데이터가 필요합니다. GPU 경로는 CPU 테스트로 검증할 수 없으며 GPU smoke를 먼저 실행하세요.

참고: [PyTorch 설치](https://pytorch.org/get-started/locally/),
[경희대 VLL의 Seraph 소개](https://vll.khu.ac.kr/joinus.html).
