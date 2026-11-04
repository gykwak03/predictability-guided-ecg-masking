# Seraph Phase 1 따라 하기

현재 구현된 `run_phase1.py`와 `submit_phase1.sh`를 호출하는 보조 스크립트는
`scripts/seraph_phase1_tutorial.sh`다. 아래 블록은 **단계별로 실행**한다.
전체를 한꺼번에 붙여 넣지 않는다. 각 학습 단계는 job 종료·결과 확인 후 다음 단계로 넘어간다.
모델·지표 정의는 [PHASE1.md](PHASE1.md)를 참고한다.

## 0. 로컬 코드 전달과 접속

Windows PowerShell에서 실행한다. `git diff`로 변경을 검토한 뒤 필요한 코드·문서를 commit/push한다.
현재 저장소 remote는 `origin`, branch는 `main`이다. 기존 Phase 1 코드도 함께 전달되어 있어야 한다.
이 튜토리얼 작성 과정에서는 commit/push나 서버 접속·학습 제출을 하지 않았다.

```powershell
cd C:\Users\gykwa\kyunghee2026\capstonedesign2\predictability-guided-ecg-masking
git status --short
# 이 튜토리얼의 변경을 전달하는 예시. 다른 작업의 변경은 따로 검토한다.
git add scripts/seraph_phase1_tutorial.sh tests/test_seraph_tutorial.sh docs/SERAPH_TUTORIAL.md README.md
git diff --cached --stat
git commit -m "Add step-by-step Seraph phase 1 tutorial"
git push origin main
ssh seraph
```

이후 명령은 **Seraph Bash**에서 실행한다. 접속 alias는 PC의 기존 SSH 설정을 따른다.

```bash
cd /data/gykwak03/capstone2/predictability-guided-ecg-masking
squeue -u "$USER"
git status --short
# 이 checkout을 읽는 job이 없고 서버 변경사항을 정리한 뒤에만 실행한다.
git pull --ff-only origin main
```

Git pull은 서버 데이터·가상환경·checkpoint를 옮기지 않는다. 기존 데이터를 그대로 읽는다.

## 1. 실행 세션 만들기

```bash
bash scripts/seraph_phase1_tutorial.sh init
bash scripts/seraph_phase1_tutorial.sh env
```

`init`은 학습 없이 TAG를 만들고 `experiments/artifacts/phase1/cache/tutorial/current-tag`에 저장한다.
같은 경로로 재접속하면 TAG를 자동 재사용한다. 별도의 새 실험은 `init 내가정한영문TAG`로 구분한다.
기존 실험을 찾을 때는 `init 기존TAG`를 사용한다. TAG에는 영문·숫자·`.`·`_`·`-`만 쓴다.

기본 경로는 다음과 같다.

| 용도 | 경로 |
| --- | --- |
| Python | `/data/gykwak03/envs/ecg-mask-pilot/bin/python` |
| 50 Hz 전처리 데이터 | `/data/gykwak03/capstone2/ptbxl_250hz_bp067_50` |
| 원본 WFDB | `/data/gykwak03/capstone2/ptbxl_raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3` |
| 새 결과 | 현재 저장소의 `experiments/artifacts/phase1/` |
| 로그 | `/data/gykwak03/capstone2/logs/` |

데이터가 다른 위치라면 `export ECG_ARTIFACT_DIR=/실제/경로`로 지정한 뒤 계속 유지한다.
결과/cache를 바꾸면 TAG 저장 위치도 달라지므로 재접속할 때 같은 환경변수를 다시 지정한다.
이 보조 스크립트는 위 Seraph 기본 경로를 `ECG_*` 변수로 전달한다. JSON config만 바꿔도 이 변수들이 우선한다.
학습량 등 실험 config는 기본 `configs/phase1_seraph.json`이고, 다른 파일은 `ECG_PHASE1_CONFIG`로 지정한다.
실험 도중 config·소스·seed·microbatch를 바꾸면 재개 검사가 거부할 수 있다.

## 2. 계획과 데이터 확인

```bash
bash scripts/seraph_phase1_tutorial.sh plan
bash scripts/seraph_phase1_tutorial.sh check
```

`plan`은 학습 없이 설정을 저장한다. `check`는 개인 Python의 import를 확인한 뒤,
**srun으로 할당받은 노드에서** train/validation 데이터·manifest·환자 분리·쓰기 권한을 검증한다.
파일 전체 검증/hash 때문에 시간이 들 수 있고 할당 대기 중일 수도 있다. 실제 학습은 하지 않는다.
기존 50 Hz artifact가 맞으면 원본 ZIP 추출·전처리를 반복하지 않는다.
의존성이 빠졌다면 오류를 확인하고 해당 개인 환경만 수정한다. base에 설치하거나 전체 환경을 downgrade하지 않는다.

50 Hz 데이터가 없거나 잘못된 경우에만 새 경로로 전처리한다.

```bash
export ECG_ARTIFACT_DIR=/data/gykwak03/capstone2/ptbxl_250hz_bp067_50_phase1_new
export ECG_PYTHON=/data/gykwak03/envs/ecg-mask-pilot/bin/python
bash scripts/submit_phase1.sh prepare --config configs/phase1_seraph.json
# 위 job 완료를 squeue/sacct와 phase1-ecg-phase1-prepare-JOBID.out/.err로 확인한 뒤:
bash scripts/seraph_phase1_tutorial.sh check
```

이 경우 이후에도 바꾼 `ECG_ARTIFACT_DIR`를 계속 지정한다. 기존 디렉토리를 삭제하지 않는다.

## 3. 합성 smoke → P25 profile

```bash
bash scripts/seraph_phase1_tutorial.sh smoke
bash scripts/seraph_phase1_tutorial.sh logs smoke
```

제출 시 job 번호와 결과 경로를 출력하고 세션에 저장한다. 로그 화면은 `Ctrl+C`로 빠져나온다.
이는 로그 보기만 멈추며 학습 job을 취소하지 않는다.

```bash
bash scripts/seraph_phase1_tutorial.sh status
# smoke status가 completed이고 해당 job이 큐에서 사라진 뒤:
bash scripts/seraph_phase1_tutorial.sh profile
bash scripts/seraph_phase1_tutorial.sh logs profile
```

Smoke는 합성 ECG·작은 모델·CPU 실행 검사다. Profile은 실제 데이터와 기본 MAE의 **P25, 3 updates** 실행이다.
P25는 1200 tokens로 가장 크다. `env`에 나온 결과 root 아래
`profile_TAG-profile/p25_r0.5/runtime.json`, `resources.json`, `history.csv`를 확인한다.
실제 GPU·전체 메모리·peak VRAM·시간·오류를 확인하고 pilot 진행 여부를 결정한다.
3 updates 시간에는 초기화 영향이 있고 전체 3000 updates 시간이나 수렴을 보장하지 않는다.
OOM이면 다른 조건도 같은 effective batch가 되도록 config를 수정한 **새 TAG**로 check/profile부터 다시 진행한다.
예: microbatch 2 × accumulation 16 = effective batch 32. 기존 run을 설정만 바꿔 재개하지 않는다.

## 4. 세 patch pilot

다음 세 블록을 **각각 완료 확인 후** 실행한다. 모두 50%·3000 updates·seed 42이며 조건별 seed 실험을 모두 돌리는 작업은 아니다.

```bash
bash scripts/seraph_phase1_tutorial.sh pilot 25
bash scripts/seraph_phase1_tutorial.sh logs p25
bash scripts/seraph_phase1_tutorial.sh status
```

```bash
bash scripts/seraph_phase1_tutorial.sh pilot 50
bash scripts/seraph_phase1_tutorial.sh logs p50
bash scripts/seraph_phase1_tutorial.sh status
```

```bash
bash scripts/seraph_phase1_tutorial.sh pilot 125
bash scripts/seraph_phase1_tutorial.sh logs p125
bash scripts/seraph_phase1_tutorial.sh status
```

각 결과는 `pilot_TAG-p25/`, `pilot_TAG-p50/`, `pilot_TAG-p125/`에 있다.
진행 중인 같은 세션 job이 있으면 다음 제출을 거부한다. GPU 총량 1장·동시 제출 제한 안에서 한 job씩 진행한다.
Smoke/profile은 기본 1시간, pilot/fit/score는 기본 24시간 한도로 제출한다.
이는 요청한 시간 제한이지 예상 실행 시간이 아니다. 기본 partition은 `batch_ugrad_advisor_x`다.
다른 시간은 `ECG_JOB_HOURS`, partition은 `ECG_PARTITION`으로 지정한다. Debug는 최대4시간이다.

```bash
# 끝난/부분 완료 결과도 보고서를 만들 수 있다.
bash scripts/seraph_phase1_tutorial.sh analyze all
```

보고서는 각 run의 `analysis_XXXXXXXX/REPORT.html`에 생성된다.
NRMSE, zero 대비 skill, correlation, prediction/target RMS, 학습 곡선과 비용을 함께 검토한다.
스크립트가 최적 patch를 자동 선정하지 않는다.

## 5. 선택한 patch로 추정기 학습 → score

세 pilot을 모두 완료한 뒤 patch를 명시한다. 아래 50은 실행 예시이며 선정 결과가 아니다.

```bash
bash scripts/seraph_phase1_tutorial.sh fit 50
bash scripts/seraph_phase1_tutorial.sh logs fit
bash scripts/seraph_phase1_tutorial.sh status
```

`fit_completed`가 되면 Ridge/CNN/Transformer 학습이 끝난 것이다. Score는 다음에 별도로 실행한다.
선택한 patch는 세션에 저장하므로 score·resume에 다시 적을 필요가 없다.

```bash
bash scripts/seraph_phase1_tutorial.sh score
bash scripts/seraph_phase1_tutorial.sh logs score
bash scripts/seraph_phase1_tutorial.sh status
# 최종 completed 확인 후:
bash scripts/seraph_phase1_tutorial.sh analyze fit
```

최종 결과는 `estimators_TAG-fit/estimator_summary.csv`, 모델별 `score/` 아래 오차·map·관측 수에 있다.
`fit_completed`와 score까지 끝난 `completed`를 구분한다.

## 6. TIMEOUT·오류·부분 결과

먼저 실제 job 번호로 종료 이유와 stdout/stderr를 확인한다.

```bash
sacct -j 실제JOB번호 --format=JobID,State,Elapsed,ExitCode
bash scripts/seraph_phase1_tutorial.sh logs p25
bash scripts/seraph_phase1_tutorial.sh status
```

TIMEOUT 등으로 끝났다면 **동일 TAG·동일 설정**에서 다음처럼 마지막 checkpoint부터 재개한다.

```bash
bash scripts/seraph_phase1_tutorial.sh resume p25
# 추정기 학습 재개는 다음 명령:
bash scripts/seraph_phase1_tutorial.sh resume fit
```

위 두 명령도 필요한 것만 하나씩 실행한다. 이미 완료한 run은 재개하지 않는다.
계산 오류는 원인을 확인한 뒤 재개한다. 실패한 score는 세 모델 fit이 완료되어 있을 때 `score`를 다시 실행할 수 있다.
완료 score를 덮어쓰지는 않는다. 소스·실험 설정 불일치는 정상 결과로 무시하지 않는다.
72회 원인 탐색을 재실행하는 명령은 이 튜토리얼에 없다.

## 7. 선택 사항: 60%·70%

선택한 patch에서만 필요할 때 각각 실행한다. 기본 튜토리얼에 자동 포함되지 않는다.

```bash
bash scripts/seraph_phase1_tutorial.sh sensitivity 50 0.6
# 완료·검토 후에만:
bash scripts/seraph_phase1_tutorial.sh sensitivity 50 0.7
```

해당 로그는 `logs p50r60`, `logs p50r70`, 재개는 `resume p50r60`, `resume p50r70`다.

## 8. 보고서 다운로드

먼저 `analyze all`을 실행하고 서버에서 묶음을 만든다.

```bash
bash scripts/seraph_phase1_tutorial.sh pack
```

출력의 `REPORT_BUNDLE=...tar.gz` 경로를 복사한다. 아래는 **로컬 Windows PowerShell**에서 실행한다.

```powershell
$remoteBundle = '/서버가_출력한_실제_경로.tar.gz'
$dest = 'C:\Users\gykwa\kyunghee2026\capstonedesign2\outputs\phase1-download'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
scp "seraph:$remoteBundle" $dest
$bundleName = Split-Path $remoteBundle -Leaf
tar -xzf (Join-Path $dest $bundleName) -C $dest
```

묶음에는 CSV·JSON·생성된 HTML/그림을 넣고 `.pt`, `.npy`, `.npz`는 제외한다.
수치 map으로 추가 분석할 때는 `ecg_maps.npz`·`patch_errors.npz`, 재학습에는 checkpoint를 별도로 받는다.
`pack`에는 선택 사항 sensitivity run은 포함되지 않으므로 그 결과는 별도로 전송한다.

## 제출 전에 명령만 확인

```bash
bash scripts/seraph_phase1_tutorial.sh --dry-run pilot 25
bash scripts/seraph_phase1_tutorial.sh --dry-run fit 50
bash scripts/seraph_phase1_tutorial.sh help
```

Dry-run은 제출·srun·학습·세션 파일 변경 없이 실제 호출할 명령을 출력한다.
일반 실행은 이전 단계 상태와 같은 세션의 job을 확인한다. `status`는 queue와 run 상태를 함께 보며,
job 종료 코드가 0이어도 `status.json`이 partial이면 전체 완료로 취급하지 않는다.

이 보조 스크립트는 Bash 구문 검사와 `tests/test_seraph_tutorial.sh`의 로컬 모의 테스트로 검증했다.
테스트는 scheduler·제출기·Python을 가짜 명령으로 대체하므로 실제 job을 제출하지 않는다.
실제 Seraph 환경·할당 GPU에서의 실행 결과는 사용자가 위 절차로 확인해야 한다.
