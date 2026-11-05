# ST-MEM Base 실제 학습 자원 측정

Seraph 리소스 상향 신청에 첨부할 **실제 학습의 GPU 이용률·VRAM 사용량**을 기록한다.
공식 ST-MEM Base 구조로 기존 PTB-XL train ECG를 복원하며, 안전하게 들어가는 physical batch를 탐색한 뒤
워밍업을 거쳐 **최소 90초 및 성공한 optimizer update 20회** 동안 측정한다.
신청서에서 요구한 10초 이상 `nvidia-smi` 기록, 원본 CSV, 전체 측정 구간의 그래프 이미지를 남긴다.

이용률 100%를 보장하지 않는다. 더미 텐서로 VRAM을 채우거나 별도의 부하를 걸지 않는다.
낮은 이용률도 원본과 요약에 모두 포함한다. 완료 상태는 측정의 완료이며 신청서 기준 충족이나 승인을 의미하지 않는다.

## 모델과 논문 근거

| 항목 | 이번 실행 | 논문·공식 구현 근거 |
| --- | --- | --- |
| 모델 | 공식 `ST_MEM`의 Base 구성, encoder 12 blocks / 768 width / 12 heads | [공식 구현](https://github.com/vuno/ST-MEM/blob/main/models/st_mem.py)의 `st_mem_vit_base_dec256d4b`와 같은 구조 |
| 파라미터 수 | 총 **88,534,753**, 학습 가능 **88,501,985** | 현재 공식 checkout을 직접 생성해 계산. 실행 시 다시 계산하여 기록 |
| Decoder | 4 blocks / 256 width / 4 heads, lead별 처리·가중치 공유 | 공식 Base 구성 |
| 입력 | 12 leads × 2,250 samples, 250 Hz의 9초 crop | 논문 Appendix A.1 |
| Patch / masking | P75 / 요청 비율 75%, 공식 per-lead random masking | 30 patches/lead 중 `int(30×0.25)=7`개를 남겨 실제 가림 비율은 76.67% |
| Loss / optimizer | 공식 patch-normalized masked loss / AdamW, betas=(0.9,0.95), weight decay=0.01 | 공식 구현·설정 파일 |
| Precision | 기본 AMP FP16; 필요 시 `--precision fp32`로 별도 실험 | 공식 pretraining engine도 autocast·gradient scaling 사용 |
| Batch / LR | 실제 GPU에 맞는 physical batch; accumulation 기본 1; LR=0.0012×effective batch/2048 | 논문 batch 2048·LR 0.0012 기준 선형 스케일링 |
| 논문 학습량 | 이번에는 단기 자원 측정 | **800 epochs, batch 2048**, Appendix Table 6 |
| 논문 GPU | 이번 배정 GPU는 실행 시 기록 | **NVIDIA GeForce RTX 3080 Ti**, Appendix A.1 |
| 논문 GPU 대수·총 학습 시간 | 확인 불가 | **논문에 미기재**. 임의의 GPU 대수나 시간을 쓰지 않는다. |

논문: [Learning Spatio-Temporal Representations of Multi-lead Electrocardiograms through Masked Signal Modeling, Appendix A.1·Table 6 (PDF p.14)](https://arxiv.org/pdf/2402.09450v3#page=14).
원문 p.14를 캡처하여 하드웨어·epoch·batch 근거로 첨부할 수 있다.
다만 **이 페이지에는 총 학습 시간이 없다.** 신청서가 논문에 명시된 시간을 반드시 요구한다면 이 자료만으로 해당 요건까지 충족했다고 볼 수 없다.
이번 실행의 실측 시간은 `resources.json`에 따로 남긴다. 단기 측정 시간을 논문 전체 재현 시간으로 쓰지 않는다.

**재현 범위:** 공식 Base 모델의 학습 연산·마스킹·복원 loss를 사용한다. 기존 데이터의 0.67–50 Hz / ECG별·lead별 Z-normalization을 재사용한다.
공식 예제의 0.67–40 Hz / 전체 lead·time 표준화와 다르고, 전체 사전학습 데이터·800 epochs·cosine schedule을 재현하지 않는다.
이번 결과로 논문 정확도 재현이나 downstream 성능을 주장하지 않는다. Test split·분류 라벨은 열지 않는다.

## Seraph에서 실행

Windows PowerShell에서 로그인한다.

```powershell
ssh seraph
```

아래는 **로그인한 Seraph의 Bash**에서 한 번에 실행한다.
기존 ECG Python 환경, 전처리 artifact, 공식 ST-MEM checkout이 있다고 가정한다.

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
git pull --ff-only origin main

export ECG_REPO="$PWD"
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_LOG_DIR=/data/gykwak03/capstone2/logs
export ECG_JOB_HOURS=2
export ECG_CPUS=4
export ECG_MEMORY=32G

STMEM=/data/gykwak03/capstone2/ST-MEM
if [ ! -f "$STMEM/models/st_mem.py" ]; then
  git clone https://github.com/vuno/ST-MEM.git "$STMEM"
fi

RUN="/data/gykwak03/capstone2/stmem_resource_$(date -u +%Y%m%dT%H%M%SZ)"
JOB=$(bash scripts/submit_stmem_resource_profile.sh \
  --output "$RUN" \
  --st-mem-repo "$STMEM" \
  --artifact-dir /data/gykwak03/capstone2/ptbxl_250hz_bp067_50 \
  --batch-size auto --precision amp \
  --measure-seconds 90 --min-updates 20) && {
  JOB=${JOB##*$'\n'}
  JOB=${JOB%%;*}
  printf 'JOB=%s\nRUN=%s\n' "$JOB" "$RUN"
  squeue -j "$JOB"
}
```

`JOB=`가 비어 있으면 제출 오류부터 확인한다. 빈 JOB으로 로그를 찾지 않는다.
이 작업은 기존 checkpoint나 결과를 수정하지 않고 새 폴더를 만든다. 같은 output 경로를 재사용하면 실패한다.
공식 source는 checkout 경로에서 읽으며, 파일 해시와 라이선스 해시를 `metadata.json`에 기록한다.
필요 패키지는 기존 환경의 torch/numpy/pandas/scipy/einops/matplotlib이다.

로그 확인:

```bash
tail -n 80 -F \
  "/data/gykwak03/capstone2/logs/stmem-resource-${JOB}.out" \
  "/data/gykwak03/capstone2/logs/stmem-resource-${JOB}.err"
```

`Ctrl+C`는 로그 보기만 종료한다. 취소할 때는 `scancel "$JOB"`을 사용한다.
데이터 로딩·검증 후 `Calibrating real training batch=...`가 나타난다.
자동 탐색 중 발생한 OOM은 batch를 줄여 재시도하며 `batch_calibration.csv`에 남긴다.
탐색이 끝나면 초기 모델·optimizer를 다시 만들고 워밍업 후 본 측정을 시작한다.
전체 작업 시간에는 데이터 검증·batch 탐색도 포함되므로 90초보다 길다.

완료 확인:

```bash
cat "$RUN/status.json"
cat "$RUN/resources.json"
cat "$RUN/gpu_summary.json"
head -n 18 "$RUN/gpu_raw.csv"
```

새로 로그인해 `RUN`이 사라졌다면:

```bash
find /data/gykwak03/capstone2 -maxdepth 2 -path '*/stmem_resource_*/status.json' -print
RUN=/data/gykwak03/capstone2/stmem_resource_위에서확인한실행태그
cat "$RUN/status.json"
```

## 캡처 파일과 원본 받기

| 파일 | 용도 |
| --- | --- |
| `resource_capture.png` | 모델·GPU·batch·precision, 이용률/VRAM 전체 시계열, 처음 15개 측정값을 포함한 이미지 |
| `gpu_raw.csv` | 신청서와 같은 필드의 `nvidia-smi -l 1` 원본. 워밍업도 포함 |
| `gpu_steady.csv` | 선언된 워밍업 이후 측정 구간의 **모든** 유효 표본 |
| `gpu_summary.json` | GPU 이용률·VRAM 비율의 평균/중앙값/최솟값/최댓값, 표본 수·기간 |
| `measurement_interval.json` | 측정 시작·끝(노드 현지 시각), warmup 수, 실제 측정 초 |
| `resources.json` | 성공한 update 수, update당 시간, 처리량, PyTorch peak allocated/reserved |
| `metadata.json` | 실제 parameter 수, 공식 소스 SHA, 데이터 ID, GPU, batch, LR 등 |
| `batch_calibration.csv` | 자동 batch 탐색 과정·OOM·peak memory (탐색 구간은 본 측정에서 제외) |
| `updates.csv`, `checkpoint.pt` | 실제 학습 loss·소요 시간·optimizer update 여부와 학습한 상태 |

**Windows PowerShell 새 창**에서 실행 태그를 실제 값으로 바꿔 받는다.

```powershell
Set-Location C:\Users\gykwa\kyunghee2026\capstonedesign2
New-Item -ItemType Directory -Force outputs\seraph_resource | Out-Null
scp -r seraph:/data/gykwak03/capstone2/stmem_resource_실제태그 outputs/seraph_resource/
```

이미지는 실측 CSV에서 만든 그래프다. 신청서가 터미널 화면 캡처도 요구하면 `head -n 18 "$RUN/gpu_steady.csv"` 결과를 별도로 캡처하고 원본 CSV를 함께 첨부한다.
`nvidia-smi`의 VRAM 사용량은 드라이버가 보고하는 GPU 전체 사용량이며 PyTorch의 live tensor 메모리와 다르다.
allocator cache도 포함될 수 있어 `resources.json`의 peak allocated/reserved를 함께 본다.
할당된 GPU의 UUID 하나만 추적하며, 같은 노드의 다른 GPU는 집계하지 않는다.

## 측정값이 낮거나 실행이 실패하면

- 기본 batch 탐색은 실제 forward/backward/AdamW step을 두 번 이상 성공시켜 optimizer state까지 포함한 메모리를 비교한다. 기본 여유분은 512 MiB다.
- batch가 상한 128에 도달했는데 메모리가 남으면 실제 연구에 쓸 큰 batch로 `--max-batch-size 256`을 지정해 새 실행을 할 수 있다. `--train-ecgs`보다 커질 수는 없다.
- GPU 모델·AMP·batch·데이터 공급에 따라 이용률은 달라진다. 남는 메모리를 가짜로 채우거나 좋은 구간만 선택하지 않는다.
- `--precision fp32`는 실제 연구에서 FP32를 검토할 때 별도 조건으로 측정한다. 논문 재현에 AMP가 사용된다는 사실과 구분해서 기록한다.
- batch 1도 안 들어가거나 telemetry가 10초 미만이면 실패 상태와 원본 오류를 남긴다. 이 경우 완료 자료로 제출하지 않는다.
- 이 코드는 Slurm GPU 1개를 요청한다. 단일 GPU 사용량만으로 다중 GPU 학습의 속도 향상까지 확인한 것은 아니다. 병렬 seed/조건 실험 수는 별도로 설명한다.

## 로컬 검증

`python -m pytest tests/test_stmem_resource_profile.py`는 batch 탐색·측정 구간 처리·할당 GPU 선택·로그인 노드 차단과 공식 클래스의 tiny 합성 CPU 학습을 검사한다.
`--smoke`는 작은 합성 모델이며 `real_gpu_measurement=false`로 기록하고 캡처 이미지·GPU 원본을 만들지 않는다.
실제 Base의 CUDA OOM 회복·성능·사용률은 Seraph 실행에서 확인해야 한다.
