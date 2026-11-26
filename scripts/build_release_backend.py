"""Build the complete event-fallback backend from shipped source (Windows).

No historical output directory, object file, runtime or compiler command cache
is read. Xmake discovers the local compiler and resolves its own dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party/libuipc'
MANIFEST = ROOT / 'native/release/backend_sources.json'
SOURCE = ROOT / 'build/uipc'
RUNTIME = ROOT / 'build/runtime/src'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for entry in manifest['files']:
        relative = Path(entry['path'])
        origin = (VENDOR / relative).resolve()
        target = (SOURCE / relative).resolve()
        if not origin.is_relative_to(VENDOR.resolve()) or not target.is_relative_to(SOURCE.resolve()):
            raise ValueError('Source manifest path escapes its root')
        if digest(origin) != entry['sha256']:
            raise ValueError(f'Source checksum mismatch: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != entry['sha256']:
            shutil.copy2(origin, target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--configure-only', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--cuda', type=Path, help='CUDA Toolkit directory; defaults to CUDA_PATH or nvcc discovery')
    parser.add_argument('--cuda-arch', default='sm_86', help='Target architecture; default RTX 3060 Ti / Ampere sm_86')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if sys.version_info[:2] != (3, 11):
        parser.error('This release is validated with Python 3.11; use that interpreter to build the bindings')
    manifest = prepare()
    print(f'Prepared and verified {len(manifest["files"])} source files in {SOURCE}', flush=True)
    if args.prepare_only:
        return
    if os.name != 'nt':
        parser.error('The event-fallback backend currently supports Windows/MSVC; Linux is not validated')
    xmake = shutil.which('xmake')
    if not xmake:
        parser.error('Install xmake >= 3.0.5 and add it to PATH')
    cuda = args.cuda or (Path(os.environ['CUDA_PATH']) if os.environ.get('CUDA_PATH') else None)
    if cuda is None:
        nvcc = shutil.which('nvcc')
        if nvcc:
            cuda = Path(nvcc).parent.parent
    if cuda is None or not (cuda / 'bin/nvcc.exe').is_file():
        parser.error('CUDA Toolkit not found; pass --cuda PATH')
    env = os.environ.copy()
    env['UIPC_BUILD_PYTHON'] = sys.executable
    record = {'source_manifest_sha256': digest(MANIFEST), 'commands': [],
              'python_version': sys.version, 'cuda_arch': args.cuda_arch,
              'status': 'configuring', 'started_at_unix': time.time()}
    status_file = ROOT / 'build/build_status.json'

    def save():
        status_file.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')

    def run(command, log_name):
        record['commands'].append(command)
        save()
        print('Running: ' + subprocess.list2cmdline(command), flush=True)
        with (ROOT / 'build' / log_name).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        if result.returncode:
            record.update(status='failed', failed_log=log_name, exit_code=result.returncode)
            save()
            raise RuntimeError(f'Command failed; see build/{log_name}')

    run([xmake, 'f', '-y', '-m', 'release', '--pybind=true', '--python_system=true',
         '--python_version=3.11.x', '--tests=false', '--examples=false', '--benchmarks=false',
         '--dev=true', '--cuda=' + str(cuda), '--pd_cuda_arch=' + args.cuda_arch], 'configure.log')
    record['status'] = 'configured'
    save()
    if args.configure_only:
        return
    record['status'] = 'building'
    run([xmake, 'build', '-y', '-j', str(args.jobs), 'pyuipc'], 'compile.log')
    package = SOURCE / 'build/python/src'
    if not (package / 'uipc/_native/uipc_backend_cuda.dll').is_file():
        raise RuntimeError('Build did not produce the expected CUDA runtime')
    # Resolve dependency DLL locations through this build's xmake package graph.
    # Never read or copy a DLL from the old experiment workspace or pip wheel.
    lua = '''import("core.project.config")
config.load()
import("core.project.project")
import("core.base.json")
local packages = {}
for _, target in pairs(project.targets()) do
    for name, pkg in pairs(target:pkgs()) do
        packages[name] = {root=pkg:installdir(), version=pkg:get("version"),
                          linkdirs=pkg:get("linkdirs"), libfiles=pkg:get("libfiles")}
    end
end
io.writefile("dependency_packages.json", json.encode(packages))
'''
    run([xmake, 'lua', '-c', lua], 'dependencies.log')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, RUNTIME, dirs_exist_ok=True)
    native = RUNTIME / 'uipc/_native'
    packages = json.loads((SOURCE / 'dependency_packages.json').read_text(encoding='utf-8'))
    candidates = set()
    def items(value):
        return value if isinstance(value, list) else ([value] if value else [])
    for package_info in packages.values():
        if package_info.get('root'):
            candidates.update(Path(package_info['root']).rglob('*.dll'))
        for filename in items(package_info.get('libfiles')):
            if Path(filename).suffix.lower() == '.dll':
                candidates.add(Path(filename))
        # Follow actual link directories, including system oneAPI installations.
        # Do not use PATH entries which may refer to a different TBB version.
        for directory in items(package_info.get('linkdirs')):
            directory = Path(directory)
            searches = {directory, directory.parent/'bin'}
            for i, part in enumerate(directory.parts):
                if part.lower() == 'lib':
                    # Preserve intel64/vc14 rather than mixing UWP, MT or x86
                    # DLLs from the same SDK's other redistribution directories.
                    for name in ('bin', 'redist'):
                        searches.add(Path(*directory.parts[:i], name, *directory.parts[i+1:]))
            for search in searches:
                if search.is_dir():
                    candidates.update(search.glob('*.dll'))
    for dll in sorted(candidates):
        if any(part.lower() in {'ia32', 'x86', 'win32', 'arm64', 'x86_32'} for part in dll.parts):
            continue
        target = native / dll.name
        if target.exists() and digest(target) != digest(dll):
            raise ValueError(f'Conflicting dependency DLL: {dll.name}')
        shutil.copy2(dll, target)
    (RUNTIME.parent/'dependency_versions.json').write_text(json.dumps(
        {name: data.get('version') for name, data in packages.items()}, indent=2)+'\n', encoding='utf-8')
    provenance = {
        'schema_version': 1, 'release_source_build': True,
        'source_manifest_sha256': digest(MANIFEST),
        'source_hashes': {str(VENDOR / r['path']): r['sha256'] for r in manifest['files']},
        'dll_sha256': digest(native / 'uipc_backend_cuda.dll'),
        'runtime': str(RUNTIME), 'event_fallback_control': True,
        'bunny_linear_solver': 'native_pcg', 'ordered_fem_assembly': True,
        'ordered_psq_gradient': True, 'bunny_newton_norm': 'free_vertex_rms_3',
        'velocity_only_preserves_friction': True, 'energy_checked_convergence': True,
        'checked_search_exhaustion': True, 'exhausted_search_acceptance': 'strict_decrease',
        'frozen_direction_derivative': True, 'representable_preview': True,
        'validation': 'built_from_source; numerical CUDA validation is separate',
    }
    (RUNTIME.parent / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    record.update(status='built', runtime=str(RUNTIME), dll_sha256=provenance['dll_sha256'], finished_at_unix=time.time())
    save()
    print(f'Built runtime: {RUNTIME}', flush=True)


if __name__ == '__main__':
    main()
