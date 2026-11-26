"""Check a freshly built runtime's hashes, Python ABI and DLL loading.

Does not create an Engine, allocate a scene or advance a GPU simulation.
"""
import argparse
import ctypes
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bunny_runtime import runtime_provenance
from scripts.libuipc_native_runtime import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-package', type=Path, default=ROOT/'build/runtime/src')
    parser.add_argument('--output', type=Path, default=ROOT/'build/runtime_check.json')
    args = parser.parse_args()
    package = args.runtime_package.resolve()
    record = runtime_provenance(package)
    uipc = load(package)
    backend = ctypes.WinDLL(str(package/'uipc/_native/uipc_backend_cuda.dll'))
    result = dict(passed=True, python_package=str(Path(uipc.__file__).resolve()),
                  dll_sha256=record['dll_sha256'], backend_loaded=bool(backend._handle),
                  event_fallback_source_build=bool(record.get('event_fallback_control')),
                  gpu_simulation_run=False,
                  scope='DLL/ABI/source-provenance check; not numerical validation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
