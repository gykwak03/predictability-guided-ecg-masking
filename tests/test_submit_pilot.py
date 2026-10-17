"""Validate submission scripts with a fake sbatch; no real jobs are submitted."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="Bash required for submission-script validation")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    if os.name == "nt":
        return "/" + value[0].lower() + value[2:]
    return value


@pytest.fixture
def submission_environment(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "sbatch"
    fake.write_text('#!/usr/bin/env bash\ncat > "$PILOT_CAPTURE_SCRIPT"\n'
                    'printf "%s\\n" "$@" > "$PILOT_CAPTURE_ARGS"\n'
                    'printf "Submitted batch job 12345\\n"\n', encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    template = tmp_path / "working-template.sbatch"
    template.write_text("#!/bin/bash\n#SBATCH --account=working-account\n"
                        "#SBATCH --partition=working-partition\n#SBATCH --gres=gpu:normal:1\n"
                        "exit 99\n", encoding="utf-8", newline="\n")
    capture, arguments = tmp_path / "submitted.sh", tmp_path / "arguments.txt"
    env = dict(os.environ, PILOT_SBATCH_TEMPLATE=shell_path(template),
               PILOT_PYTHON="/example/env/python", PILOT_CAPTURE_SCRIPT=shell_path(capture),
               PILOT_CAPTURE_ARGS=shell_path(arguments))

    def run(*args):
        # Set PATH inside Bash so MSYS does not reinterpret a Windows PATH list.
        command = 'export PATH="$1:$PATH"; shift; exec bash "$@"'
        return subprocess.run([BASH, "-c", command, "submission-test", shell_path(fake_bin),
                               shell_path(ROOT / "experiments/submit_pilot.sh"), *args],
                              env=env, capture_output=True, text=True, timeout=20)

    return run, capture, arguments, template


@pytest.mark.parametrize("backend, options, family, runner", [
    ("mae", [], "masking_pilot", "run_masking_pilot.py"),
    ("mae", ["--model-size", "stmem-small"], "mae_stmem_size_pilot", "run_masking_pilot.py"),
    ("mae", ["--model-size=stmem-small"], "mae_stmem_size_pilot", "run_masking_pilot.py"),
    ("stmem", ["--model-size", "small"], "st_mem_pilot", "run_st_mem_pilot.py"),
    ("predictability", [], "predictability", "run_predictability.py"),
    ("patch-difficulty", ["--source-run", "/example/source"], "patch_difficulty", "run_patch_difficulty.py"),
])
def test_submits_correct_backend_and_repository_output(submission_environment, backend, options, family, runner):
    run, capture, arguments, _ = submission_environment
    result = run(backend, *options, "--steps", "1000", "--artifact-dir", "/example/ECG data")
    assert result.returncode == 0, result.stderr
    script = capture.read_text(encoding="utf-8")
    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --account=working-account" in script
    assert "#SBATCH --gres=gpu:normal:1" in script
    assert "exit 99" not in script  # only allocation directives are copied
    command = shlex.split(next(line for line in script.splitlines() if line.startswith("srun ")))
    assert command[:3] == ["srun", "/example/env/python", "-u"]
    assert command[3] == shell_path(ROOT / "experiments" / runner)
    assert command[-2:] == ["--output-root", shell_path(ROOT / "experiments/artifacts" / family)]
    assert command.count("/example/ECG data") == 1
    assert ("--train" in command) == (backend != "patch-difficulty")
    assert "--chdir=" + shell_path(ROOT / "experiments") in arguments.read_text(encoding="utf-8")
    assert "Submitted batch job 12345" in result.stdout


@pytest.mark.parametrize("options", [["--output-root", "/outside/results"], ["--output-root=/outside/results"]])
def test_external_output_rejected_before_submission(submission_environment, options):
    run, capture, _, _ = submission_environment
    result = run("mae", *options)
    assert result.returncode == 2 and "omit --output-root" in result.stderr
    assert not capture.exists()


def test_non_gpu_template_rejected_before_submission(submission_environment):
    run, capture, _, template = submission_environment
    template.write_text("#!/bin/bash\n#SBATCH --cpus-per-task=4\n", encoding="utf-8")
    result = run("mae")
    assert result.returncode == 2 and "request a GPU" in result.stderr
    assert not capture.exists()
