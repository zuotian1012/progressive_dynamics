"""Apply the reviewed MAS/PCG patch to the isolated source; does not promote a runtime."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[1] / 'external/libuipc-psq-src/src/backends/cuda'


def main():
    package = HERE / 'mas_fix'
    manifest = json.loads((package/'manifest.json').read_text())
    # Check every file before modifying any file.
    for relative, record in manifest.items():
        current = hashlib.sha256((TARGET/relative).read_bytes()).hexdigest()
        source = hashlib.sha256((package/record['source']).read_bytes()).hexdigest()
        if source != record['after_sha256'] or current not in (record['before_sha256'], source):
            raise RuntimeError(f'Unexpected source at {relative}; inspect before applying MAS patch')
    for relative, record in manifest.items():
        shutil.copy2(package/record['source'], TARGET/relative)
    shutil.copy2(HERE/'mas_factor_kernels.inl', TARGET/'finite_element/mas_factor_kernels.inl')
    print(f'Installed MAS factor/PCG guard source into {TARGET}; runtime unchanged')


if __name__ == '__main__':
    main()
