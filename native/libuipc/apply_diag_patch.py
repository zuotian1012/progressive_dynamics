"""Install the audited FEM diag fix into the isolated checkout; does not build or promote a runtime."""
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[1] / 'external/libuipc-psq-src/src/backends/cuda/finite_element'
BEFORE = '003747b50c6cbfb2ff75eb2bd00d358f3416be9365eb22909ba655552c3be0e2'


def main():
    name = 'fem_diag_preconditioner.cu'
    current = hashlib.sha256((TARGET / name).read_bytes()).hexdigest()
    patched = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    if current not in (BEFORE, patched):
        raise RuntimeError('Unexpected FEM diag source; inspect changes before applying this patch')
    for name in ('diag_cholesky.h', 'fem_diag_preconditioner.cu'):
        shutil.copy2(HERE / name, TARGET / name)
    print(f'Installed FEM diag scaled Cholesky source into {TARGET}; runtime unchanged')


if __name__ == '__main__':
    main()
