"""Verify the switching protocol and compare two L3 runs with one fixed L2."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from report_bunny_preview_pilot import weights, rms, summary, plt, ShellProlongation
from run_bunny_event_fallback_pilot import OUT, PARENT, ASSETS, ROOT


def event_audit(run, meta, rows):
    modes = {}; events = {}; counts = {}
    for line in run.with_suffix('.log').open(encoding='utf-8', errors='replace'):
        if 'BUNNY_EVENT_MODE ' in line:
            d = dict(re.findall(r'(\w+)=([^\s]+)', line))
            key = (int(d['frame']), int(d['newton']))
            assert key not in modes
            modes[key] = d['exact'] == 'true'
        if 'BUNNY_EVENT_TRIGGER ' in line:
            d = dict(re.findall(r'(\w+)=([^\s]+)', line))
            frame, iteration = int(d['frame']), int(d['newton'])
            assert meta['event_fallback_enabled'] and frame not in events
            assert not modes[frame, iteration]
            e0, retained = float(d['E0']), float(d['retained'])
            assert np.isfinite([e0, retained]).all()
            assert abs(e0-retained) <= 64*np.finfo(float).eps*max(abs(e0), 1e-30)
            events[frame] = dict(newton=iteration, reason=d['reason'])
            counts[d['reason']] = counts.get(d['reason'], 0) + 1
    for row in rows:
        frame = row['step']; n = row['native']['newton_iterations']
        switch = events.get(frame, {}).get('newton', n)
        assert not modes[frame, 0]
        for i in range(n):
            assert modes[frame, i] == (i > switch), (frame, i)
        if frame in events:
            assert row['native']['converged'] and not row['native']['preview_non_descent']
            assert n - switch - 1 >= 3
    assert len(modes) == sum(r['native']['newton_iterations'] for r in rows)
    result = dict(passed=True, event_frames=events, event_reason_counts=counts,
        exact_gradient_directions=sum(modes.values()), frozen_gradient_directions=len(modes)-sum(modes.values()),
        per_step_reset_and_post_event_switch_verified=True,
        limitation='Switching and retained-energy checks use native logs; trigger decisions use the unchanged original preview condition in the isolated source overlay.')
    (run / 'event_audit.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    coarse = np.load(ASSETS / 'mesh_2.npz'); fine = np.load(ASSETS / 'mesh_3.npz')
    w = weights(fine)
    transfer = ShellProlongation(coarse['vertices'], fine['vertices'], coarse['faces'],
        load_npz(ASSETS / 'map_2_3.npz'), np.load(ASSETS / 'anchors_2_3.npy'))
    parent = np.load(PARENT / 'positions.npy')
    prolonged = np.array([transfer.position(x) for x in parent])
    parent_hash = hashlib.sha256((PARENT / 'positions.npy').read_bytes()).hexdigest()
    data = {}; result = dict(runs={}, fixed_parent=str(PARENT), parent_sha256=parent_hash)
    series = {}; metas = {}
    for mode in ('baseline', 'fallback'):
        run = OUT / mode
        meta = json.loads((run / 'run.json').read_text())
        assert meta['completed'] and meta['completed_steps'] == 30 and meta['level'] == 3
        assert meta['dt'] == .01 and meta['method'] == 'velpro'
        assert not meta['ground']['enabled'] and meta['contact_friction']['cloth_bunny'] == 0
        assert meta['lower_sha256'] == parent_hash and Path(meta['lower_run']).resolve() == PARENT.resolve()
        assert meta['event_fallback_enabled'] == (mode == 'fallback')
        for filename in ('no_ground_audit.json', 'line_search_audit.json'):
            assert json.loads((run / filename).read_text())['passed']
        assert json.loads((run / 'audit.json').read_text())['accepted_prefix_checks_passed']
        audit = json.loads((run / 'no_ground_audit.json').read_text())
        assert audit['saved_states_checked'] == 31
        assert audit['positions_sha256'] == hashlib.sha256((run / 'positions.npy').read_bytes()).hexdigest()
        rows = json.loads((run / 'steps.json').read_text())
        x = np.load(run / 'positions.npy'); data[mode] = x; metas[mode] = meta
        delta = x - prolonged
        series[mode + '_parent_mm'] = rms(delta, w)
        metrics = summary(series[mode + '_parent_mm'])
        metrics['max_vertex_mm'] = float(1000*np.linalg.norm(delta, axis=2).max())
        result['runs'][mode] = dict(parent_rms=metrics, converged=meta['native_steps'],
            preview=meta['preview_steps'], wall_seconds=meta['wall_seconds'],
            newton_total=sum(r['native']['newton_iterations'] for r in rows),
            protocol=event_audit(run, meta, rows))
    for field in ('config', 'material', 'ground', 'contact_friction', 'library_sha256', 'bundle_sha256'):
        assert metas['baseline'][field] == metas['fallback'][field], field
    np.testing.assert_array_equal(data['baseline'][0], data['fallback'][0])
    series['fallback_minus_baseline_mm'] = rms(data['fallback'] - data['baseline'], w)
    old_run = ROOT / 'output/bunny_preview_pilot_20261125/L3_frozen'
    old_meta = json.loads((old_run / 'run.json').read_text())
    assert old_meta['lower_sha256'] == parent_hash
    for field in ('config', 'material', 'ground', 'contact_friction', 'bundle_sha256'):
        assert old_meta[field] == metas['baseline'][field], field
    old = np.load(old_run / 'positions.npy')
    series['baseline_repeat_change_mm'] = rms(data['baseline'] - old, w)
    result['same_level_change'] = summary(series['fallback_minus_baseline_mm'])
    result['baseline_vs_previous_frozen_run'] = summary(series['baseline_repeat_change_mm'])
    result['previous_frozen_parent_rms'] = summary(rms(old-prolonged, w))
    a, b = (result['runs'][k]['parent_rms'] for k in ('baseline', 'fallback'))
    result['parent_rms_reduction_percent'] = {k: 100*(a[k]-b[k])/a[k] for k in ('mean_mm', 'peak_mm', 'final_mm')}
    result['scope'] = 'L3 VelPro, zero friction, no ground, dt=.01, 30 steps; both use exactly the same frozen L2 parent; event-triggered full gradient only for the remainder of an affected step.'
    result['limitations'] = ['Single A/B pair; no repeated-run confidence interval.',
        'Parent trajectory is a consistency reference, not physical ground truth.',
        'Wall time includes uncontrolled GPU contention; not an isolated speed benchmark.',
        'Full gradient here includes the PSQ offset derivative; the Hessian approximation and GPU PCG are unchanged.']
    time = np.arange(31)*.01
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for mode, color in [('baseline', '#1684b8'), ('fallback', '#d86b29')]:
        axes[0].plot(time, series[mode + '_parent_mm'], label=mode, color=color)
    for frame in result['runs']['fallback']['protocol']['event_frames']:
        axes[0].axvline(frame*.01, color='#d86b29', alpha=.12, linewidth=.7)
    axes[0].set_title('L3 versus the SAME prolonged L2')
    axes[1].plot(time, series['fallback_minus_baseline_mm'], label='fallback versus baseline', color='#70429b')
    axes[1].plot(time, series['baseline_repeat_change_mm'], label='baseline versus previous run', color='#888888', linestyle='--')
    axes[1].set_title('L3 trajectory changes')
    for ax in axes:
        ax.set_xlabel('Physical time (s)'); ax.set_ylabel('Area-weighted RMS (mm)')
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(OUT / 'comparison.png', dpi=180)
    np.savetxt(OUT / 'comparison.csv', np.column_stack([time, *series.values()]), delimiter=',',
        header=','.join(['time_s', *series]), comments='')
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
