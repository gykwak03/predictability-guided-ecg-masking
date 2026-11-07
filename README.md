# Predictability-guided ECG masking

현재 연구 단계는 **전처리 확정 → patch/mask pilot → Ridge·CNN·Transformer 난이도 추정기**입니다.
탐색 backbone은 **확대 Plain ECG-MAE + encoder/decoder 고정 sin/cos 시간 임베딩**으로 정했습니다.
ST-MEM은 이후 최종 검증 대상으로 남겨둡니다.

```text
preprocessing/                  현재 전처리 CLI와 원본 구현 노트북
experiments/                    재사용할 추정기·실행·시각화 코드
  phase1/                       새 Phase 1 config/data/mask/model/train/분석 모듈
  run_phase1.py                  새 실행 진입점
  artifacts/phase1/             새 결과 namespace (run ID별 분리)
configs/                        합성 CPU smoke / Seraph config
scripts/                        Seraph Slurm 제출·worker (Bash LF)
tests/                          전처리·추정기 검증
docs/PHASE1_HANDOFF.md           다음 구현에 사용할 범위와 새 참고 경로
archive/20261028_legacy/         과거 실험 코드·설명서·테스트 원문
```

- [현재 실행 코드 안내](experiments/README.md)
- [다음 Codex 구현 지침](docs/PHASE1_HANDOFF.md)
- [과거 코드 찾아보기 및 재실행 주의사항](archive/README.md)
- [이동 전후 경로와 SHA-256](archive/20261028_legacy/source_manifest.json)

새 Phase 1 파이프라인은 `experiments/run_phase1.py`입니다.
서버에서 단계별로 실행하려면 **[Seraph 튜토리얼](docs/SERAPH_TUTORIAL.md)**과
`scripts/seraph_phase1_tutorial.sh`를 사용하세요. 이 스크립트는 명시한 단계만 제출합니다.
**[전처리·모델·지표 및 Windows/Seraph 실행 명령](docs/PHASE1.md)**과
**[로컬 검증 기록](docs/PHASE1_VALIDATION.md)**을 먼저 읽으세요.
`plan` / `preflight` / `prepare` / `smoke` / `profile` / `pilot` /
`estimator-fit` / `estimator-score` / `analyze`를 분리했습니다.
합성 CPU smoke와 실제 PTB-XL 결과는 명시적으로 구분합니다. 실제 GPU 성능과 최적 patch/ratio는 아직 확정하지 않았습니다.

기존 `run_predictability.py`는 과거 40 Hz·global mask 결과 재현용입니다.
과거 Easy/Medium/Hard 출력은 확정된 학습 정책으로 취급하지 않습니다.

기존 `experiments/artifacts/`, `preprocessing/artifacts/`, `.venv/`와 저장소 밖 데이터·분석 결과는 이동하지 않았습니다.
과거 설명서의 당시 모델 선택과 명령어는 현재 실행 지침과 구분해서 읽으세요.

검증은 저장소 루트에서 `python -m pytest`로 실행합니다. Torch 등 의존성이 설치된 Python을 사용하세요.
이 명령은 `tests/`만 수집하고 과거 전체 실험을 실행하지 않습니다.

기존 Random 사전학습 checkpoint로 Within/Cross 추가 학습을 재검증하려면
[checkpoint 재사용·전체 hidden loss·optimizer 대조 실험](docs/DEPENDENCY_CONTINUATION.md)을 참고하세요.
`experiments/run_dependency_continuation.py`는 사전학습을 반복하지 않고 새 결과 폴더에서 D/E/F를 비교합니다.

GPU 상향 신청에 사용할 실제 ST-MEM Base 학습 자원 측정은
[ST-MEM Base 측정·로그·캡처 안내](docs/STMEM_RESOURCE_PROFILE.md)를 참고하세요.
`scripts/submit_stmem_resource_profile.sh`가 실제 ECG 학습과 1초 간격 GPU 측정을 제출합니다.

ST-MEM Base를 Small·Plain MAE와 같은 P50/r50·3000 updates로 학습하려면
[ST-MEM Base baseline 실행 안내](docs/STMEM_BASE_BASELINE.md)를 참고하세요.
`experiments/run_stmem_base_baseline.py`는 Base만 새로 학습하고 동일한 복원 평가·linear probe를 수행합니다.

Plain과 ST-MEM을 Small/Base 각각 같은 파라미터 수로 맞추고, lead별 decoder로
세 seed에서 복원·linear probe를 비교하려면 [네 모델 비교 실행 안내](docs/LEADWISE_COMPARISON.md)를 참고하세요.
