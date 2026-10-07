# Mask 단위 확정 pilot

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

1. notebook/runner와 기존 `preprocessing/artifacts/ptbxl_250hz` 폴더 **전체**를 서버로 옮깁니다.
2. 서버에 맞는 CUDA PyTorch, numpy, pandas, matplotlib, scikit-learn, IPython을 준비합니다.
3. 먼저 `--mode smoke --train`으로 동작을 확인합니다.
4. 다음 명령의 예시 경로를 실제 읽기/쓰기 가능한 경로로 바꾸고 GPU pilot을 실행합니다.

```bash
python experiments/run_masking_pilot.py --mode pilot --train \
  --artifact-dir /your/path/ptbxl_250hz \
  --output-root /your/output/masking_pilot --seed 42
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

1-seed 단기 pilot은 최종 결론이 아닙니다. 상위 두 조건은 동일 모델의 더 긴 학습 및
seed 3개에서 재검증하세요. MSE는 조건별 과제 난이도가 달라 최적 설정 선택 기준이 아닙니다.
Easy/Medium/Hard 및 predictability-aware 실험은 이 pilot 이후 단계입니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_masking_pilot.py -q
```

로컬 전처리 데이터가 필요합니다. GPU 경로는 CPU 테스트로 검증할 수 없으며 GPU smoke를 먼저 실행하세요.

참고: [PyTorch 설치](https://pytorch.org/get-started/locally/),
[경희대 VLL의 Seraph 소개](https://vll.khu.ac.kr/joinus.html).
