"""Render the two zero-friction bunny trajectories with one camera and one L0."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from bunny_runtime import audit_bunny_contact
from compare_bunny_consistency import movie
from report_bunny_friction_control import matched_settings, read_chain


def render(semi, velpro, output):
    chains = [read_chain(folder) for folder in (semi, velpro)]
    first, second = (runs for runs, _ in chains)
    if set(first) != set(second):
        raise ValueError('Compare two branches with the same verified levels')
    level = max(first)
    if first[0][0].resolve() != second[0][0].resolve():
        raise ValueError('The integrators must use the same saved L0')
    steps = first[level][1]['steps']
    if steps != second[level][1]['steps']:
        raise ValueError('Compare two branches over the same time window')
    dt = first[0][1]['dt']
    output.mkdir(parents=True, exist_ok=True)
    panels, rows = [], []
    for method, (runs, errors) in zip(('semi', 'velpro'), chains):
        for k in range(level + 1):
            run, meta = runs[k]
            if matched_settings(meta) != matched_settings(first[k][1]):
                raise ValueError(f'Unmatched experiment settings at {run}')
            contact = audit_bunny_contact(run, meta)
            if contact['friction_systems_active'] or contact['soft_position_constraint_active']:
                raise ValueError(f'Expected zero friction and zero consistency weight at {run}')
            if meta['steps'] < steps or meta['dt'] != dt or (k and meta['method'] != method):
                raise ValueError(f'Unexpected trajectory or time window at {run}')
            with np.load(Path(meta['assets']) / f'mesh_{k}.npz') as mesh:
                faces = mesh['faces']
            states = np.load(run / 'positions.npy', mmap_mode='r')[:steps + 1]
            name = 'PD semi' if method == 'semi' else 'PD++ VelPro'
            panels.append((f'{name}   L{k}', states, faces))
            rows.append(dict(method=method, level=k, vertices=meta['vertices'],
                             peak_rms_mm=errors[k]['max_common_finest_L0_mm'],
                             final_rms_mm=errors[k]['final_common_finest_L0_mm']))
    with np.load(Path(first[0][1]['assets']) / 'bunny.npz') as bunny:
        movie(output, panels, bunny['vertices'][bunny['faces']], dt,
              measure_silhouettes=True, columns=min(level + 1, 4))
    with (output / 'silhouettes.csv').open() as stream:
        overlaps = list(csv.DictReader(stream))
    for panel, row in enumerate(rows):
        samples = [x for x in overlaps if int(x['level']) == panel]
        lowest = min(samples, key=lambda x: float(x['iou_from_L0']))
        row.update(min_iou=float(lowest['iou_from_L0']), min_iou_time_s=float(lowest['time_s']),
                   final_iou=float(samples[-1]['iou_from_L0']))
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    video = cv2.VideoCapture(str(output / 'comparison.mp4'))
    try:
        samples = np.unique(np.r_[np.arange(0, steps + 1, 4), steps])
        if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(samples):
            raise RuntimeError('Incomplete comparison movie')
        lowest = min(overlaps, key=lambda x: float(x['iou_from_L0']))
        video.set(cv2.CAP_PROP_POS_FRAMES, int(np.where(samples == int(lowest['frame']))[0][0]))
        ok, frame = video.read()
        if not ok or not cv2.imwrite(str(output / 'minimum_overlap.png'), frame):
            raise RuntimeError('Could not export the minimum-overlap frame')
        if dt == .01 and steps >= 40:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(np.where(samples == 40)[0][0]))
            ok, frame = video.read()
            if not ok or not cv2.imwrite(str(output / 'contact_0_40s.png'), frame):
                raise RuntimeError('Could not export the contact frame')
    finally:
        video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('semi', 'velpro', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    render(args.semi, args.velpro, args.output)
