"""Record child exit and accepted prefix even when the native engine exits Python."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def finalize(output, returncode):
    path = output / 'run.json'
    if not path.exists():
        return
    meta = json.loads(path.read_text())
    if returncode and meta.get('status') in ('running', 'initializing'):
        (output / 'run_before_native_exit.json').write_text(path.read_text())
        meta.update(status='failed', completed=False, child_exit_code=returncode,
                    failed_step=meta['completed_steps'] + 1,
                    error='Native child exited before completing the next step; consult sibling .log')
        path.write_text(json.dumps(meta, indent=2))
        (output / 'failed_step.json').write_text(json.dumps(dict(
            step=meta['failed_step'], accepted=False, child_exit_code=returncode,
            last_saved_step=meta['completed_steps'], reason=meta['error']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--finalize-existing', type=int)
    p.add_argument('child_args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    child_args = args.child_args
    if child_args and child_args[0] == '--':
        child_args = child_args[1:]
    output = Path(child_args[child_args.index('--output') + 1]).resolve()
    if args.finalize_existing is not None:
        finalize(output, args.finalize_existing)
        sys.exit(0)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(Path(__file__).with_name('run_shirt_psq.py')), *child_args]
    start = time.time()
    with output.with_suffix('.log').open('wb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    finalize(output, result.returncode)
    record = dict(command=command, child_exit_code=result.returncode,
                  wall_seconds=time.time()-start, output=str(output))
    output.with_suffix('.process.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)
    sys.exit(result.returncode)
