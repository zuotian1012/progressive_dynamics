"""Preserve the validated flag code and binaries without changing active solvers."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/validated_flag_baseline_20261123'
SUITE = ROOT/'output/flag_to_L9_20261122'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert json.loads((SUITE/'summary.json').read_text())['complete']
    assert not (OUT/'baseline_lock.json').exists(), 'A completed baseline is never overwritten.'
    OUT.mkdir(exist_ok=True)
    files=[]
    def save(source, relative):
        target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        digest=sha(source)
        assert sha(target)==digest
        files.append(dict(source=str(source),snapshot=str(relative),sha256=digest))
    for folder in ['progressive','native/libuipc','native/flag_l3_repair']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.cu','.cuh','.cpp','.h','.inl'):
                save(path,Path('source')/path.relative_to(ROOT))
    for name in ['run_flag_parallel_wind.py','build_flag_l3_fix.py','bunny_runtime.py',
                 'libuipc_native_runtime.py','report_flag_parallel_wind.py',
                 'validate_flag_psq.py','freeze_flag_baseline.py']:
        path=ROOT/'scripts'/name
        assert path.is_file(),path
        save(path,Path('source/scripts')/name)
    index=json.loads((SUITE/'index.json').read_text())
    entries=[index['L0']]+[e for level in index['levels'].values() for e in level.values()]
    packages={}
    for e in entries:
        run=Path(e['run']);meta=json.loads((run/'run.json').read_text())
        package=Path(meta['native_package']);dll=package/'uipc/_native/uipc_backend_cuda.dll'
        assert sha(dll)==meta['runtime_sha256']
        packages[str(package)]=package
        prefix=Path('runs')/run.name
        for name in ['run.json','source_snapshot.py']:
            source=run/name
            if name=='source_snapshot.py' and not source.exists():source=Path(e['root'])/name
            save(source,prefix/name)
    for package in packages.values():
        for path in sorted(package.parent.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                save(path,Path('runtime')/package.parent.name/path.relative_to(package.parent))
        build=package.parent.parent/package.parent.name.replace('_runtime','_build')
        for path in sorted(build.iterdir()):
            if path.is_file() and path.suffix in ('.json','.cu','.inl'):
                save(path,Path('build_provenance')/build.name/path.name)
    for path in [SUITE/'index.json',SUITE/'summary.json',SUITE/'report/manifest.json']:
        save(path,Path('evidence')/path.name)
    manifest=dict(baseline='validated_flag_20261123',files=files,
        scope='Code and runtime copies plus configuration and evidence references; existing trajectories and mesh assets remain at their recorded paths.',
        status='Passed existing flag audits; not a proof of complete PSQ/PD/PD++ accuracy.',
        policy='Retain current ordered assembly and CPU LDLT. New validation scenes use separate output directories; no solver replacement in this baseline.',
        reference_suite=str(SUITE),validated_runs=27,shared_L0_runs=1)
    (OUT/'baseline_lock.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    assert all(sha(OUT/f['snapshot'])==f['sha256'] for f in files)
    print(json.dumps(dict(snapshot=str(OUT),verified_files=len(files),runtime_packages=len(packages))))


if __name__=='__main__':main()
