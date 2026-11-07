# Plain과 ST-MEM의 lead별 decoder 비교

각 크기에서 두 모델이 같은 수의 파라미터를 사용한다. 모든 모델은 encoder에서 전체 lead의 visible patch를 함께 처리하고, decoder는 lead별로 실행하되 가중치를 공유한다. Plain은 기존 PyTorch Transformer 블록과 초기화 방식을 유지한다. 크기를 정확히 맞추기 위해 ST-MEM처럼 patch projection 전후 LayerNorm과 SEP 토큰을 사용하며 별도 decoder lead embedding은 두지 않는다. 비교 대상은 기존 global decoder Plain이 아닌 새 lead-wise Plain이다.

| 모델 | Encoder width / depth / heads | Decoder width / depth / heads | 전체 / 학습 가능 파라미터 |
| --- | --- | --- | ---: |
| Plain Small · ST-MEM Small | 384 / 12 / 6 | 256 / 4 / 4 | 24,624,278 / 24,590,998 |
| Plain Base · ST-MEM Base | 768 / 12 / 12 | 256 / 4 / 4 | 88,529,558 / 88,476,310 |

seed 42, 43, 44로 각각 초기화부터 사전학습한다. PTB-XL 전처리 데이터에서 같은 P50, lead마다 25/50 patch를 무작위로 가리는 r50, 가린 300개 patch 전체 MSE, FP32, batch 1 × accumulation 32, 3,000 optimizer updates를 쓴다. 같은 seed의 ECG와 마스크 순서는 네 모델에서 같으며 분석 단계에서 해시를 비교한다. GPU 종류와 연산 시간은 다를 수 있다.

0 / 1,000 / 3,000 update 시점에 동일한 validation mask bank에서 파형 복원 NRMSE 등을 측정한다. Downstream은 encoder를 동결하고 가리지 않은 12-lead ECG로 PTB-XL 5개 상위 클래스의 다중 라벨 linear probe를 학습한다. 두 pooling 순서와 probe 300 / 1,500 updates를 기록하며, 주 지표는 3,000 update encoder에 대한 `norm_then_mean` pooling, 1,500 update probe의 validation macro AUROC다. Fold 10 test는 사용하지 않는다.

## Seraph에서 실행

로컬 PowerShell에서 `ssh seraph`로 접속한다. 이어서 Moana 로그인 노드에서 실행한다.

```bash
show-assoc
show-qos
squeue -u "$USER"
sinfo -o '%P %a %l %G'

cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
git pull --ff-only origin main
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
export ECG_LOG_DIR=/data/gykwak03/capstone2/logs
export ECG_QOS=qos_gykwak03_2026_2
export ECG_JOB_HOURS=24
STUDY="/data/gykwak03/capstone2/results/leadwise_$(date -u +%Y%m%dT%H%M%SZ)"

"$ECG_PYTHON" experiments/run_leadwise_comparison.py plan --study-dir "$STUDY" \
  --artifact-dir /data/gykwak03/capstone2/ptbxl_250hz_bp067_50 \
  --st-mem-repo /data/gykwak03/capstone2/ST-MEM &&
JOB=$(bash scripts/submit_leadwise_comparison.sh "$STUDY") &&
JOB=${JOB%%;*} &&
printf 'JOB=%s\nSTUDY=%s\n' "$JOB" "$STUDY"
```

한 배열에 모델당 GPU 1장인 task 네 개를 제출한다. 각 task는 자기 모델의 seed 세 개를 차례로 처리한다. task 0은 Plain Small, 1은 Plain Base, 2는 ST-MEM Small, 3은 ST-MEM Base다. 체크포인트 공간은 전체 최소 약 17 GB에 임시 저장 여유분이 추가로 필요하다. 정상 종료 전 재제출할 때는 기존 task가 끝났는지 확인하고 같은 study 경로를 사용한다. 특정 모델만 재개하려면 예를 들어 `bash scripts/submit_leadwise_comparison.sh "$STUDY" 3`을 쓴다. 완료된 seed와 평가를 건너뛴다.

```bash
squeue -r -j "$JOB"
tail -n 50 -F "$ECG_LOG_DIR/leadwise-${JOB}_3."{out,err}
```

로그는 task가 시작된 뒤 생긴다. `Ctrl+C`는 tail만 종료한다. 학습이 끝나면 `sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed`로 확인하고 분석한다.

```bash
"$ECG_PYTHON" experiments/run_leadwise_comparison.py analyze --study-dir "$STUDY"
cat "$STUDY/analysis/audit.json"
cat "$STUDY/analysis/final_summary.csv"
```

`final_summary.csv`는 seed별 최종값을 평균·표준편차로 집계한다. `final_by_seed.csv`는 개별 성능, `paired_deltas_a_minus_b.csv`는 같은 seed의 모델 간 차이, `reconstruction.csv`와 `probes.csv`는 각 checkpoint의 복원과 클래스별 분류 지표를 담는다.
