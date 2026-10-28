# 과거 실험·참고 코드 보관

`20261028_legacy/`는 정리 직전 소스를 보관한 묶음입니다. 파일 삭제 없이 50개를 이동했습니다.
현재도 쓰는 추정기 3개·테스트 2개·전처리 2개는 과거 동작 확인용 원문 사본도 함께 보관했습니다.
새 수정은 루트의 `experiments/`, `preprocessing/`, `tests/`에서 진행하세요.

| 참고 목적 | `20261028_legacy/experiments/` 아래 파일 |
| --- | --- |
| 선택한 Plain MAE + 고정 시간 임베딩 | `mae_cause_model.py`, `mae_cause_plan.py`의 `factor_010`, `Masking_pilot.ipynb` |
| MAE 원인 탐색 설계·실행·분석 | `mae_cause_*.py`, `run_mae_cause_study.py`, `analyze_mae_cause_study.py`, `MAE_CAUSE_STUDY.md` |
| 공식 ST-MEM 구현 연동 | `st_mem_adapter.py`, `run_st_mem_pilot.py`, `ST_MEM_pilot.ipynb` |
| 과거 Plain MAE pilot·진단 | `run_masking_pilot.py`, `masking_diagnostics.py`, `inspect_masking_pilot.py`, `plot_masking_pilot.py`, `pilot_comparison.py` |
| patch 난이도 확인 실험 | `patch_difficulty.py`, `run_patch_difficulty.py`, `PATCH_DIFFICULTY.md` |
| JEPA pilot | `*ecg_jepa*`, `ECG_JEPA_PILOT.md`, `requirements-ecg-jepa.txt` |
| SOTA 비교·환경 준비 | `*sota*`, `SOTA_COMPARISON.md` |
| 과거 Slurm 제출 | `submit_*.sh`, `prepare_and_run_mae_all.sbatch` |

과거 테스트는 `20261028_legacy/tests/`, 전처리 원문 사본은 `20261028_legacy/preprocessing/`에 있습니다.
옛 설명서도 원문을 유지했으므로 당시의 ‘현재 선택’ 문구는 현재 결정과 다를 수 있습니다.
전체 이전/현재 경로와 파일 해시는 [source_manifest.json](20261028_legacy/source_manifest.json)에 기록했습니다.
모델·학습·과거 분석 코드의 내용은 바꾸지 않았습니다. 특히 원인 탐색 runner의 `SOURCE_FILES` 묶음은 그대로입니다.

## 결과와 재실행 경로

기존 결과는 저장소 루트의 **`experiments/artifacts/`**에 그대로 있습니다.
전처리 배열은 **`preprocessing/artifacts/`** 또는 서버의 외부 데이터 경로에 그대로 있습니다.
archive 안에 결과를 복제하거나 기존 checkpoint를 수정하지 않았습니다.

과거 코드의 `__file__` 기준 기본 경로는 이동한 위치를 가리킵니다.
**옛 README의 `experiments/run_…` 경로나 Slurm 제출 스크립트를 그대로 실행하지 마세요.**
참고용 Python runner를 재실행하려면 새 파일 경로와 함께 데이터·결과·외부 모델 경로를 명시해야 합니다.
예를 들어 원인 탐색 분석은 저장소 루트에서 다음 경로를 사용합니다.

```bash
python archive/20261028_legacy/experiments/analyze_mae_cause_study.py \
  --root experiments/artifacts/mae_cause_all/study \
  --output experiments/artifacts/mae_cause_all/analysis_new
```

이 예시는 경로 안내이며, 과거 결과의 checkpoint 일치 검사 오류를 우회하지 않습니다.
원인 탐색 학습은 `--artifact-dir`(plan), `--raw-dir`(prepare), `--st-mem-repo`와
`--output-root` 등 해당 subcommand의 경로를 명시합니다. `--help`로 확인하세요.
옛 Slurm 스크립트는 원문 보존 대상으로, 이동된 배치에 맞춘 재제출 스크립트가 아닙니다.
이 정리 과정에서는 서버 학습을 제출하지 않았습니다.
