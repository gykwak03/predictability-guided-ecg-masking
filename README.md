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
