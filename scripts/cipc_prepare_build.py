"""WSL build preparation; verify upstream sources, never alter solver code."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

REVISION = '9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a'

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)

def prepare(project, base):
    base.mkdir(parents=True, exist_ok=True)
    src = base / 'source'
    original = project / 'external/Codim-IPC-main'
    tree = get_json(f'https://api.github.com/repos/ipc-sim/Codim-IPC/git/trees/{REVISION}?recursive=1')
    verified = {}
    for entry in tree['tree']:
        name = entry['path']
        if entry['type'] != 'blob' or not name.startswith(('Library/', 'Python/', 'Externals/flat_hash_map/', 'Externals/meta/')):
            continue
        data = (original / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob != entry['sha']:
            raise RuntimeError(f'Local source differs from upstream: {name}')
        dest = src / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError(f'Build source already differs: {name}')
        dest.write_bytes(data)
        verified[name] = hashlib.sha256(data).hexdigest()
    manifest = {'repository': 'https://github.com/ipc-sim/Codim-IPC', 'commit': REVISION,
                'verified_source_files': verified, 'solver_source_changes': [],
                'build_changes': ['Standalone CMake build', 'Ubuntu pybind11 2.9.1 for Python 3.10',
                                  'System CHOLMOD 5.10 / OpenBLAS instead of downloaded CHOLMOD / MKL',
                                  'Omit unused Kokkos-kernels target; retain Kokkos 3.1.01 and upstream Cabana fork']}
    deps = base / 'deps'
    deps.mkdir(exist_ok=True)
    def dep(spec):
        name, repo, ref = spec
        commit = get_json(f'https://api.github.com/repos/{repo}/commits/{ref}')['sha']
        archive = deps / (name + '.tar.gz')
        url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
        if not archive.exists():
            subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '300', url, '-o', str(archive)+'.part'], check=True)
            Path(str(archive)+'.part').rename(archive)
        destination = deps / name
        if not (destination/'_extracted_complete').exists():
            destination.mkdir(exist_ok=True)
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    parts = Path(member.name).parts[1:]
                    if not parts:
                        continue
                    member.name = str(Path(*parts))
                    target = (destination / member.name).resolve()
                    if destination.resolve() not in target.parents:
                        raise RuntimeError(f'Unsafe archive member {member.name}')
                    if member.issym():
                        link_target = (target.parent/member.linkname).resolve()
                        if destination.resolve() not in link_target.parents:
                            raise RuntimeError(f'Unsafe archive symlink {member.name}')
                    if member.islnk():
                        raise RuntimeError(f'Unsupported archive hard link {member.name}')
                    tar.extract(member, destination)
            (destination/'_extracted_complete').write_text(commit)
        print(f'Dependency ready: {name} {commit}', flush=True)
        return name, {'repository': repo, 'commit': commit, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        manifest['dependencies'] = dict(pool.map(dep, [('kokkos','kokkos/kokkos','3.1.01'),
                         ('cabana','Liby99/Cabana','master'), ('amgcl','ddemidov/amgcl','master')]))
    (base/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(base/'source_manifest.json', project/'native/cipc/source_manifest.json')
    print(f'Verified {len(verified)} upstream source files; no solver edits.', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('/opt/cipc-baseline'))
    a = p.parse_args()
    prepare(a.project, a.base)
