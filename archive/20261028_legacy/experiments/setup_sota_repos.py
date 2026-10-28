"""Clone pinned official sources into a sibling external/ directory.

Existing checkouts are inspected, never reset, pulled, or overwritten.
No package installation or training is performed by this command.
"""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def setup(root, only=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = json.loads((HERE/"sota_sources.json").read_text(encoding="utf-8"))
    if only:
        unknown = set(only)-set(lock)
        if unknown:
            raise ValueError(f"Unknown source repositories: {sorted(unknown)}")
        lock = {name: source for name, source in lock.items() if name in only}
    result = {}
    for name, source in lock.items():
        folder = root/name
        if folder.exists():
            if not (folder/".git").exists():
                raise ValueError(f"Existing non-Git directory: {folder}; choose another --external-root")
            head = git("-C", folder, "rev-parse", "HEAD")
            if head != source["commit"]:
                raise ValueError(f"Existing {name} is at {head}, expected {source['commit']}. "
                                 "It was left unchanged. Choose a fresh external root for pinned sources.")
        else:
            subprocess.run(["git", "clone", "--depth", "1", source["url"], str(folder)], check=True)
            if git("-C", folder, "rev-parse", "HEAD") != source["commit"]:
                subprocess.run(["git", "-C", str(folder), "fetch", "--depth", "1", "origin", source["commit"]], check=True)
                subprocess.run(["git", "-C", str(folder), "checkout", "--detach", source["commit"]], check=True)
        dirty = git("-C", folder, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise ValueError(f"Tracked edits in {folder}; sources were left unchanged")
        result[name] = dict(path=str(folder), **source)
        print(f"READY {name}: {source['commit']} at {folder}", flush=True)
    record_path = root/"sota_sources_record.json"
    records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    records.update(result)
    record_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=HERE.parents[1]/"external")
    parser.add_argument("--only", nargs="+", help="Set up only named repositories, e.g. ECG_JEPA")
    args = parser.parse_args()
    setup(args.external_root, args.only)
