"""Verify expensive full-study submission and dispatch without requesting a GPU."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="Bash required")
SCRIPT = ROOT / "experiments/submit_mae_cause_all.sh"


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@pytest.fixture
def shell(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "args.txt"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n', newline="\n")
    sbatch.chmod(0o755)
    env = dict(os.environ, CAPTURE_ARGS=shell_path(capture), MAE_LOG_DIR=shell_path(tmp_path / "logs"))
    # Do not inherit the host's actual allocation or study overrides.
    for name in ("SLURM_JOB_ID", "MAE_PARTITION", "MAE_QOS", "MAE_REPO_DIR", "MAE_WORK_DIR"):
        env.pop(name, None)

    def run(*args, updates=None):
        return subprocess.run([BASH, "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"',
                               "mae-submit-test", shell_path(bin_dir), shell_path(SCRIPT), *args],
                              env=dict(env, **(updates or {})), text=True, capture_output=True, timeout=30)
    return run, capture, env, bin_dir


def test_requires_explicit_partition_and_never_submits_on_missing_choice(shell):
    run, capture, _, _ = shell
    result = run()
    assert result.returncode != 0 and not capture.exists()
    assert "MAE_PARTITION" in result.stderr


def test_dry_run_is_read_only_and_describes_all_runs(shell):
    run, capture, env, _ = shell
    result = run("--dry-run", updates={"MAE_PARTITION": "allowed_batch"})
    assert result.returncode == 0, result.stderr
    assert not capture.exists() and not Path(env["MAE_LOG_DIR"]).exists()
    assert "24 conditions" in result.stdout and "42,43,44" in result.stdout
    assert "--worker" in result.stdout


def test_one_gpu_explicit_partition_qos_and_spool_safe_repo(shell):
    run, capture, _, _ = shell
    result = run(updates={"MAE_PARTITION": "allowed_batch", "MAE_ACCOUNT": "my-account", "MAE_QOS": "my-qos"})
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert "--partition=allowed_batch" in args and "--account=my-account" in args
    assert "--qos=my-qos" in args and "--gres=gpu:normal:1" in args
    assert "--export=ALL" in args and "--chdir=" + shell_path(ROOT) in args
    assert args[-2:] == [shell_path(SCRIPT), "--worker"]
    assert not any("nodelist" in a or "dependency" in a or "array=" in a for a in args)


def test_filters_and_unallocated_worker_are_rejected(shell):
    run, capture, _, _ = shell
    assert run("--conditions", "legacy_short").returncode == 2
    assert run("--worker").returncode == 2
    assert not capture.exists()


@pytest.mark.parametrize("run_status", [0, 7])
def test_worker_runs_full_plan_expanded_probe_and_analyzes_partial_results(shell, tmp_path, run_status):
    run, _, _, bin_dir = shell
    repo, work = tmp_path / "repo with spaces", tmp_path / "full study"
    repo.mkdir()
    artifact, raw, upstream = [tmp_path / p for p in ("artifact", "raw", "stmem")]
    artifact.mkdir()
    (artifact / "manifest.json").write_text("{}")
    (raw / "records500").mkdir(parents=True)
    (upstream / "models").mkdir(parents=True)
    (upstream / "models/st_mem.py").write_text("")
    for name, content in {
        "srun": '#!/usr/bin/env bash\nexec "$@"\n',
        "flock": '#!/usr/bin/env bash\nexit 0\n',
        "fake-python": '''#!/usr/bin/env bash
printf '%q ' "$@" >> "$CAPTURE_COMMANDS"
printf '\\n' >> "$CAPTURE_COMMANDS"
if [[ "$1" == "-" ]]; then cat >/dev/null; exit 0; fi
mode=$3
shift 3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache) cache=$2; shift 2 ;;
    --output-root) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
if [[ "$mode" == "prepare" ]]; then mkdir -p "$cache"; printf '{}' > "$cache/manifest.json"; fi
if [[ "$mode" == "run" ]]; then mkdir -p "$output"; printf '{}' > "$output/plan.json"; exit "$FAKE_RUN_STATUS"; fi
''',
    }.items():
        file = bin_dir / name
        file.write_text(content, newline="\n")
        file.chmod(0o755)
    commands = tmp_path / "commands.txt"
    updates = dict(SLURM_JOB_ID="987", MAE_REPO_DIR=shell_path(repo), MAE_PYTHON=shell_path(bin_dir / "fake-python"),
                   MAE_WORK_DIR=shell_path(work), MAE_ARTIFACT_DIR=shell_path(artifact), MAE_RAW_DIR=shell_path(raw),
                   MAE_STMEM_REPO=shell_path(upstream), CAPTURE_COMMANDS=shell_path(commands), FAKE_RUN_STATUS=str(run_status))
    result = run("--worker", updates=updates)
    assert result.returncode == run_status, result.stderr
    calls = [shlex.split(line) for line in commands.read_text().splitlines()]
    train = next(c for c in calls if len(c) > 2 and c[2] == "run")
    assert "--probe-audit" in train and "--resume" in train
    assert not {"--conditions", "--seeds", "--no-probe", "--smoke", "--tiny-overfit"} & set(train)
    assert train[train.index("--output-root") + 1] == shell_path(work / "study")
    assert train[train.index("--st-mem-repo") + 1] == shell_path(upstream)
    assert any(c[1].endswith("analyze_mae_cause_study.py") for c in calls if len(c) > 1)
    # Re-submission must reuse a complete cache rather than rebuild it.
    result = run("--worker", updates=updates)
    assert result.returncode == run_status, result.stderr
    calls = [shlex.split(line) for line in commands.read_text().splitlines()]
    assert sum(len(c) > 2 and c[2] == "prepare" for c in calls) == 1
