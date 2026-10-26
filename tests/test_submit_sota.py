"""Capture generated Slurm scripts with fake sbatch; never submit real jobs."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH,reason="Bash required")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/"+value[0].lower()+value[2:] if os.name == "nt" else value


@pytest.fixture
def fake_submit(tmp_path):
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    fake = bin_dir/"sbatch"
    fake.write_text('#!/usr/bin/env bash\ncat > "$CAPTURE_SCRIPT"\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nprintf "Submitted batch job 12345\\n"\n',encoding="utf-8",newline="\n")
    fake.chmod(0o755)
    data = tmp_path/"new 50hz data"
    data.mkdir()
    (data/"manifest.json").write_text('{}')
    script,args = tmp_path/"submitted.sh",tmp_path/"args.txt"
    env = dict(os.environ,SOTA_PYTHON="/bin/bash",SOTA_ARTIFACT_DIR=shell_path(data),
               SOTA_EXTERNAL_ROOT="/example/external",SOTA_ACCOUNT="test-account",SOTA_PARTITION="test-partition",
               CAPTURE_SCRIPT=shell_path(script),CAPTURE_ARGS=shell_path(args))
    def run(*options):
        return subprocess.run([BASH,"-c",'export PATH="$1:$PATH"; shift; exec bash "$@"',
            "sota-test",shell_path(bin_dir),shell_path(ROOT/"experiments/submit_sota.sh"),*options],
            env=env,text=True,capture_output=True,timeout=20)
    return run,script,args,data


@pytest.mark.parametrize("model",["stmem","patchecg","ecgcpc","all"])
def test_one_gpu_with_correct_new_data_and_no_stale_node(fake_submit,model):
    run,script,args,data = fake_submit
    result = run(model,"--mode","profile","--steps","20")
    assert result.returncode == 0,result.stderr
    commands = [shlex.split(line) for line in script.read_text().splitlines() if line.startswith("srun ")]
    names = [c[c.index("--model")+1] for c in commands]
    assert names == (["stmem","patchecg","ecgcpc"] if model == "all" else [model])
    for command in commands:
        assert command[command.index("--artifact-dir")+1] == shell_path(data)
        assert command[command.index("--external-root")+1] == "/example/external"
    flags = args.read_text()
    assert "--gres=gpu:normal:1" in flags
    assert "--account=test-account" in flags
    assert "--nodelist" not in flags
    assert "--chdir="+shell_path(ROOT/"experiments") in flags


def test_reject_conflicting_paths_without_submission(fake_submit):
    run,script,_,_ = fake_submit
    result = run("all","--artifact-dir","/wrong/data")
    assert result.returncode == 2 and not script.exists()
