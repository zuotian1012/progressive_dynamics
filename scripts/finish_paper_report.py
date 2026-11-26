"""Monitor existing runs and render/audit their terminal results; never changes simulation settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            time.sleep(.2)
    raise ValueError(f'Cannot read {path}')


def run(*args):
    print('REPORT', *map(str, args), flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / args[0]), *map(str, args[1:])],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    pending = {'flag_budget1000', 'bunny_event_window'}
    last = {}
    while pending:
        for branch in sorted(pending.copy()):
            folder = root / branch
            chain = read(folder / 'chain.json')
            stamp = (chain['status'], len(chain['chain']))
            if last.get(branch) != stamp:
                print('PROGRESS', branch, stamp, flush=True)
                last[branch] = stamp
            if chain['status'] == 'running':
                continue
            paths = [Path(item['run']) for item in chain['chain']]
            if 'failed_level' in chain:
                paths.append(folder / f'L{chain["failed_level"]}')
            levels = []
            for path in paths:
                meta = read(path / 'run.json')
                if meta['completed_steps']:
                    levels.append(meta['level'])
                    if not (path / 'audit.json').exists():
                        run('audit_paper_run.py', path)
            run('compare_paper_levels.py', folder)
            flag = branch.startswith('flag')
            chosen = sorted(set(k for k in ([0, 3, 6, max(levels)] if flag else [0, 3, 5, max(levels)]) if k in levels))
            run('render_paper_levels.py', folder, '--end-seconds', 20 if flag else 1,
                '--playback-rate', 1 if flag else .25, '--levels', *chosen)
            if flag:
                run('render_flag_paper_frames.py', folder)
            else:
                run('render_paper_levels.py', folder, '--start-seconds', .1, '--end-seconds', .65,
                    '--playback-rate', .1, '--azim', -135, '--tag', 'contact_closeup', '--levels', *chosen)
            pending.remove(branch)
            print('SCENE_REPORT_COMPLETE', branch, flush=True)
        if pending:
            time.sleep(15)
    run('build_paper_catalog.py', root)
    run('build_paper_gallery.py', root)
    print('ALL_REPORTS_COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
