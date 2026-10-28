# Phase 1 로컬 검증 — 2026-10-28 (Asia/Seoul)

구현·CPU 테스트를 완료했다. Seraph 원격 접속, Slurm 실제 제출, 실제 PTB-XL 학습·평가는 하지 않았다.
자동 commit/push 및 환경 설치·downgrade도 하지 않았다. 작업 시작 전에 있던 archive 이동·기존 runner 수정은 보존했다.

## 실행 환경과 명령

로컬 `.venv/Scripts/python.exe`는 기존 Microsoft Store Python 경로가 끊겨 실행되지 않았다.
앱에 설치된 Python을 사용하고 기존 venv의 패키지 경로를 `PYTHONPATH`로 지정했다.

| 항목 | 실제 검증 값 |
|---|---|
| Python | 3.12.14, `%USERPROFILE%/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe` |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| SciPy | 1.18.1 |
| WFDB | 4.3.1 |
| CUDA | 없음 — 합성 CPU 검증 |

실행한 전체 테스트:

```powershell
$env:PYTHONPATH = Join-Path $repo '.venv\Lib\site-packages'
$phase1Temp = Join-Path $repo ('.pytest_cache\phase1-' + [guid]::NewGuid().ToString('N'))
& $env:ECG_PYTHON -m pytest -q --basetemp $phase1Temp
# 46 passed in 36.56s
```

초기 pytest 실행은 sandbox의 기본 TEMP에서 디렉터리를 만들지 못했다.
허용된 저장소 내부의 매번 새로운 `--basetemp`로 바꿔 전체 테스트를 통과했다.
환경 문제를 테스트 성공으로 계산하지 않았으며, 마지막 실행에는 skip/failure/error가 없었다.

## 확인한 계약

- P25/P50/P125 patchify/unpatchify, lead-major index 및 초 환산, lead별 정확한 hidden count. 불가능한 비율은 오류.
- Mask bank CSV의 64-bit seed·hash 저장/replay, 행 순서·모델 RNG 독립성, M/~M pair와 전체 patch 관측 수.
- Estimator-fit / held-out train score / validation 환자와 ECG 분리. Score·validation 진폭을 바꿔도 fit-only 분모는 불변.
- Ridge/CNN/Small Transformer hidden 입력 perturbation 후 예측 정확히 불변, visible 문맥 변화에는 반응. 실제 visible 0과 hidden indicator 구분.
- Hidden-only normalized loss 및 NRMSE, visible prediction의 점수 미반영, 미관측 patch NaN, baseline 열세·진폭 경고.
- 정규화 전 hidden 값 변경이 visible 정규화 값을 바꾸는 별도 검사. 모델 입력 차단과 전처리 통계 의존성을 혼동하지 않음.
- **전체 크기 factor_010**과 새 factory의 초기 state tensor·P50 CPU forward 출력이 정확히 일치.
- 고정 encoder/decoder 시간 임베딩은 gradient·변화 없음. 나머지 모든 학습 대상 파라미터는 유한한 gradient와 optimizer update 후 변화 확인.
- 동일 effective batch를 다른 microbatch로 나눠도 데이터·mask identity가 일치.
- 중단/재개 학습과 연속 학습의 최종 모델 tensor, update, 다음 데이터·mask, Torch RNG가 CPU에서 정확히 일치. Identity 변경 시 재개 거부.
- CNN/Transformer의 전체 학습 batch/mask sequence hash 일치.
- Train/validation만 여는 artifact loader. 의도적으로 손상시킨 test/label 파일을 두고 접근을 guard해도 검증 성공.
- 실제 노트북 전처리를 합성 500 Hz physical-mV WFDB에 실행하고, resample/filter/정규화 수식과 수치 일치 및 40/50 Hz 차이 확인.
- Test 원본 WFDB를 아예 만들지 않은 상태에서 train/validation 전처리와 artifact 재사용 성공. 다른 cwd의 상대 경로 및 `--raw-dir`/`--artifact-dir` alias 확인.
- 40 Hz·metadata 순서·환자 중복·데이터 누락·평가할 mask 없음은 명시적 오류. 실데이터 실패 시 합성 fallback 없음.
- 한 추정기 실패 시 다른 결과를 보존하고 partial 및 제외 사유 기록. 완료 scoring 덮어쓰기 거부.
- 세 patch의 축소 합성 pilot 및 별도 단일-patch profile 실행 확인.
- 경로 `CLI > 환경변수 > config > 기본값`, config-relative와 cwd-relative 기준, OS가 다른 절대 경로 거부, 절대 진입점의 다른 cwd 실행 확인.
- Slurm 스크립트 LF, export·argv 보존, log/output 생성 순서, partition/account/QOS/GPU/time 제한 확인.

`bash -n scripts/submit_phase1.sh`, `bash -n scripts/phase1_worker.sh`는 Git Bash로 통과했다.
Git Bash의 sandbox named-object 제한 때문에 구문 검사만 제한 밖에서 실행했다. 스크립트 본문 실행이나 제출은 하지 않았다.
`git diff --check`는 오류 없이 통과했다. 기존 `.gitattributes`·Git 설정에 따른 CRLF 안내는 남아 있으며 Bash 파일 바이트는 LF다.
`archive/20261028_legacy/source_manifest.json`에 있는 **57개 파일의 SHA-256은 모두 일치**했다.
Checkpoint와 `phase1/cache/*.json`가 Git ignore 대상인 것도 확인했다.

## 합성 smoke 산출물

실행 명령:

```powershell
& $env:ECG_PYTHON "$repo\experiments\run_phase1.py" smoke --config "$repo\configs\phase1_smoke.json" --run-id verified-v1
```

결과 root: `experiments/artifacts/phase1/smoke_verified-v1/`, `status=completed`, `synthetic=true`, `real_ecg_performance=false`, `test_opened=false`.
이 실행은 작은 MAE/CNN/Transformer, 각 신경망 **2 updates**, effective batch 4다.
총 약 22.75초는 로컬 합성 smoke의 기록이며 실제 ECG나 GPU 시간 추정에 사용하지 않는다.

| 합성 검사 모델 | 평가 pool | NRMSE | zero 대비 skill | 해석 |
|---|---|---:|---:|---|
| 축소 MAE | validation | 1.14657 | -0.32125 | zero보다 나쁨 — 코드 검사만 완료 |
| Ridge | held-out train score | 0.34930 | 0.87799 | 합성 파형의 복원 수치, 실데이터 주장 불가 |
| 축소 CNN | held-out train score | 1.02870 | -0.05822 | zero보다 나쁨으로 명시 |
| 축소 Transformer | held-out train score | 1.14137 | -0.30272 | zero보다 나쁨으로 명시 |

CNN/Transformer의 학습 sequence SHA-256은 둘 다
`6c134a18e70358ba4af3e2b1ec6366a330e9c7dcbc51b9b5e7e7f8194cbb25eb`이다.
Ridge fitting target은 총 388개이며 lead별 count·bank hash를 저장했다. 이 count는 subsampling으로 실제 사용한 target 수이며 고유 target 수라는 뜻은 아니다.

보고서:

- [합성 MAE 보고서](../experiments/artifacts/phase1/smoke_verified-v1/pilot_20261028T085413Z_0b50e899/analysis_5d9406aa/REPORT.html)
- [합성 추정기 보고서](../experiments/artifacts/phase1/smoke_verified-v1/estimators_20261028T085419Z_0ed9f1fb/analysis_a3e1f45a/REPORT.html)

학습 곡선, 집계 NRMSE/관측 수, 실제 합성 파형 위 hidden 영역·복원 그림을 직접 확인했다.
HTML·waveform 이미지·NPZ·checkpoint는 기본 Git 추적 대상이 아니므로 다른 clone에는 별도 전송해야 한다.
초기 구현 확인 smoke도 별도 run ID로 남겼으며 최종 검증 결과를 덮어쓰지 않았다.

## 사용자가 Seraph에서 확인할 항목

1. Commit/push할 기존 정리 변경과 새 코드 선택, 서버 checkout·job 상태 확인 후 pull.
2. 서버 Python import, 실제 50 Hz manifest/배열·환자 pool 및 출력 권한 preflight.
3. 합성 smoke → 할당 GPU/VRAM·버전 기록과 profile → pilot → estimator fit/score → 분석.
4. 실제 PTB-XL 복원 성능·수렴·진폭·학습 시간·VRAM을 확인한 뒤 patch/ratio 검토.

서버 PyTorch 2.5.1+cu121 환경에서의 실제 실행, 실데이터 성능, GPU 비용, 최적 설정, downstream 유용성은 아직 검증하지 않았다.
전체 복사 가능한 PowerShell/Bash 명령, job 재개와 결과 회수는 [Phase 1 실행 안내](PHASE1.md)에 있다.
