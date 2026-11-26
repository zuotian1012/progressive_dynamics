"""Compare the two integrators against their own L0, with and without friction."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from bunny_runtime import audit_bunny_contact
from compare_bunny_consistency import movie


def read_chain(folder):
    chain = json.loads((folder / 'chain.json').read_text())
    paths = {}
    current = Path(chain['levels'][-1]['run']) if chain['levels'] else Path(chain['seed'])
    while True:
        meta = json.loads((current / 'run.json').read_text())
        if not meta['completed'] or not json.loads((current / 'continuation_audit.json').read_text())['passed']:
            raise ValueError(f'Unverified trajectory at {current}')
        paths[meta['level']] = (current, meta)
        if not meta['level']:
            break
        current = Path(meta['lower_run'])
    errors = json.loads((folder / 'level_errors.json').read_text())
    return paths, {row['level']: row for row in errors}


def matched_settings(meta):
    cfg = json.loads(json.dumps(meta['config']))
    del cfg['contact']['friction']['enable']
    return (cfg, meta['level'], meta['dt'], meta['steps'], meta['assets'], meta['bundle_sha256'],
            meta['material'], meta['ground'], meta['library_sha256'], meta['exact_offset_gradient'],
            meta['initial_non_descent_rescue'], meta['adaptive_line_search'],
            meta['non_descent_probe_refinement'], meta.get('bunny_newton_norm', 'max_coordinate'),
            meta.get('velocity_map', 'intrinsic'), meta.get('requested_contact_stiffness', 1e9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zero', type=Path, nargs=2, metavar=('SEMI_CHAIN', 'VELPRO_CHAIN'), required=True)
    parser.add_argument('--baseline', type=Path, nargs=2, metavar=('SEMI_CHAIN', 'VELPRO_CHAIN'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--level', type=int, help='Level to render. Defaults to the highest common verified level.')
    args = parser.parse_args()
    groups = [[read_chain(folder) for folder in pair] for pair in (args.zero, args.baseline)]
    common = min(max(errors) for group in groups for _, errors in group)
    level = common if args.level is None else args.level
    if not 1 <= level <= common:
        raise ValueError(f'Expected a shared completed level between 1 and {common}')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, panels, settings = [], [], {}
    for mu, group in zip((0., .3), groups):
        if group[0][0][0][0].resolve() != group[1][0][0][0].resolve():
            raise ValueError('The two integrators must share exactly the same saved L0 for each friction setting')
        for method, (paths, errors) in zip(('semi', 'velpro'), group):
            for k in range(level+1):
                run, meta = paths[k]
                contact = audit_bunny_contact(run, meta)
                if contact['cloth_bunny'] != mu or meta.get('consistency_weight', 0.) != 0:
                    raise ValueError(f'Unexpected coefficient at {run}')
                if k and meta['method'] != method:
                    raise ValueError(f'Expected {method} at {run}')
                expected = settings.setdefault(k, matched_settings(meta))
                if matched_settings(meta) != expected:
                    raise ValueError(f'Non-friction physical or solver settings differ at {run}')
                if k:
                    e = errors[k]
                    rows.append(dict(mu=mu, method=method, level=k, run=str(run),
                                     peak_parent_mm=e['max_parent_mm'],
                                     peak_common_finest_L0_mm=e['max_common_finest_L0_mm'],
                                     peak_time_s=e['max_common_finest_L0_frame']*meta['dt'],
                                     final_common_finest_L0_mm=e['final_common_finest_L0_mm']))
                if (k == 0 and method == 'semi') or k == level:
                    label = f'mu={mu:g}  L{k}  {"preview" if k == 0 else method}'
                    with np.load(Path(meta['assets']) / f'mesh_{k}.npz') as mesh:
                        faces = mesh['faces']
                    track = np.load(run / 'positions.npy', mmap_mode='r')
                    panels.append((label, track, faces))
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with np.load(Path(meta['assets']) / 'bunny.npz') as bunny:
        triangles = bunny['vertices'][bunny['faces']]
    movie(args.output, panels, triangles, meta['dt'], columns=3)
    # Use an actual synchronous movie frame nearest the largest measured
    # zero-friction discrepancy at this resolution.
    zero_peak = max((row for row in rows if row['mu'] == 0 and row['level'] == level),
                    key=lambda row: row['peak_common_finest_L0_mm'])
    last = len(panels[0][1]) - 1
    samples = np.unique(np.r_[np.arange(0, last+1, 4), last])
    index = int(np.argmin(np.abs(samples * meta['dt'] - zero_peak['peak_time_s'])))
    capture = cv2.VideoCapture(str(args.output / 'comparison.mp4'))
    try:
        if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(samples):
            raise ValueError('Rendered video has a different frame count from the simulation samples')
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or not cv2.imwrite(str(args.output / 'peak.png'), frame):
            raise OSError('Could not save the synchronized comparison frame')
    finally:
        capture.release()


if __name__ == '__main__':
    main()
