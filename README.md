# WCB waveform-group masking experiments

현재 실행 경로는 `experiments/run_wcb.py`와 `experiments/wcb/`입니다.
40ms 고정 patch를 검출 파형 전체로 묶어 마스킹하며, 기존 Plain MAE와 전체 hidden MSE를 유지합니다.
W/C/B 세 비율을 독립적으로 조절하고 Coordinate-wise, REINFORCE, SPSA를 초기화부터 비교합니다.
Random masking warmup과 과거 사전학습 checkpoint 출발은 없습니다.

- [Seraph 전송·환경·제출·재개·평가 명령](docs/WCB_SERAPH.md)
- [학습 없는 100ms/40ms 파형 묶음 분석](docs/PATCH_AUDIT.md)
- [메인 설정](configs/wcb_seraph.json)
- [의존성](configs/wcb_requirements.txt)
- [기존 GitHub 소스와 로컬 변경분 보관](archive/20261112_pre_wcb/README.md)

```text
experiments/wcb/          새 마스크·데이터·MAE·학습·평가 모듈
experiments/run_wcb.py    plan / prepare / profile / run / replay / analyze / test
configs/wcb_seraph.json   8방법 × 3seed, 40ms, 3000 committed updates
scripts/submit_wcb.sh     CPU prepare와 GPU profile/run/test Slurm 제출
scripts/package_wcb.py    데이터·가중치 없이 검증 가능한 전송 묶음 생성
archive/20261112_pre_wcb/ GitHub main 전체 zip + 로컬 변경 overlay + SHA256
```

Random과 Balanced Control Probe는 개별 평가합니다. 메인 정책 보상은 Random LP이며
Balanced는 기본 5구간마다 및 최종 상태에서 측정합니다. Balanced 보상은 별도 설정으로 실행할 수 있습니다.
최종 표현 품질은 동결 encoder의 validation linear probe로 측정하며 Test는 별도 명령에서만 엽니다.

기존 `phase1`, backbone, dependency 및 lead-wise 코드와 결과는 보존했습니다.
새 실험은 해당 실행 경로에 의존하지 않습니다. 보관 zip에서 기존 소스 전체를 재구성할 수 있습니다.
실제 성능과 GPU 비용은 Seraph 실행 후 판단합니다. 합성 CPU smoke를 실제 ECG 결과로 취급하지 않습니다.

관련 검사: `python -m pytest tests/test_wcb.py tests/test_patch_audit.py -q`.
