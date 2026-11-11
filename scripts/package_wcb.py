"""Create/verify a source-only portable tarball. Never includes ECGs or weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--verify',type=Path,help='Extracted bundle root')
    args=p.parse_args()
    if args.verify:
        m=json.loads((args.verify/'bundle_manifest.json').read_text(encoding='utf-8'))
        for name,expected in m.items():
            path=(args.verify/name).resolve()
            if not path.is_relative_to(args.verify.resolve()) or sha(path)!=expected: raise ValueError(f'Hash mismatch: {name}')
        print(f'Verified {len(m)} bundled files'); return
    if args.output is None or args.output.exists(): p.error('Supply a new --output tar.gz path')
    names=['README.md','pytest.ini','experiments/run_wcb.py','experiments/patch_audit.py','experiments/run_patch_audit.py',
        'scripts/submit_wcb.sh','scripts/wcb_worker.sh','scripts/package_wcb.py',
        'docs/WCB_SERAPH.md','docs/PATCH_AUDIT.md','tests/test_wcb.py','tests/test_patch_audit.py']
    files=[ROOT/n for n in names]+list((ROOT/'experiments/wcb').glob('*.py'))+list((ROOT/'configs').glob('wcb*'))
    files+=list((ROOT/'archive/20261112_pre_wcb').glob('*'))
    files=sorted(p for p in files if p.is_file())
    hashes={f.relative_to(ROOT).as_posix():sha(f) for f in files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as raw,tarfile.open(fileobj=raw,mode='w:gz') as tar:
        for f in files: tar.add(f,arcname=f.relative_to(ROOT).as_posix(),recursive=False)
        content=json.dumps(hashes,indent=2).encode()
        info=tarfile.TarInfo('bundle_manifest.json'); info.size=len(content); tar.addfile(info,io.BytesIO(content))
    print(f'{args.output.resolve()} ({args.output.stat().st_size} bytes; {len(files)} files)')

if __name__=='__main__': main()
