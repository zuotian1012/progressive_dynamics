"""Compare the short frozen/exact pilot without interpreting L0 as ground truth."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from progressive.prolongation import ShellProlongation, frames

OUT = ROOT / 'output/bunny_preview_pilot_20261125'
ASSETS = ROOT / 'output/bunny_paper_assets'


def weights(mesh):
    a = frames(mesh['vertices'], mesh['faces'])[3]
    return np.bincount(mesh['faces'].ravel(), weights=np.repeat(a / 3, 3),
                       minlength=len(mesh['vertices'])) / a.sum()


def rms(delta, w):
    return 1000 * np.sqrt(np.einsum('tvi,tvi,v->t', delta, delta, w))


def summary(v):
    return dict(mean_mm=float(v[1:].mean()), peak_mm=float(v.max()), final_mm=float(v[-1]))


def main():
    data = {}; results = {}
    for name in ('frozen', 'exact', 'L1_frozen', 'L1_exact'):
        p = OUT / name
        m = json.loads((p / 'run.json').read_text())
        assert m['completed'] and m['completed_steps'] == 30
        assert json.loads((p / 'no_ground_audit.json').read_text())['passed']
        x = np.load(p / 'positions.npy')
        rows = json.loads((p / 'steps.json').read_text())
        data[name] = (m, x)
        results[name] = dict(converged=m['native_steps'], preview=m['preview_steps'],
                             wall_seconds=m['wall_seconds'],
                             newton_total=sum(r['native']['newton_iterations'] for r in rows),
                             preview_frames=[r['step'] for r in rows if not r['native']['converged']])
    for a, b in [('frozen', 'exact'), ('L1_frozen', 'L1_exact')]:
        ma, xa = data[a]; mb, xb = data[b]
        for field in ('config', 'material', 'ground', 'contact_friction', 'library_sha256', 'bundle_sha256'):
            assert ma[field] == mb[field], field
        np.testing.assert_array_equal(xa[0], xb[0])
    coarse = np.load(ASSETS / 'mesh_0.npz'); fine = np.load(ASSETS / 'mesh_1.npz')
    wc, wf = weights(coarse), weights(fine)
    transfer = ShellProlongation(coarse['vertices'], fine['vertices'], coarse['faces'],
                                load_npz(ASSETS / 'map_0_1.npz'), np.load(ASSETS / 'anchors_0_1.npy'))
    series = {}
    for mode in ('frozen', 'exact'):
        mapped = np.array([transfer.position(x) for x in data[mode][1]])
        delta = data['L1_' + mode][1] - mapped
        v = rms(delta, wf)
        series[mode + '_parent'] = v
        results[mode + '_L1_parent'] = summary(v)
        results[mode + '_L1_parent']['max_vertex_mm'] = float(np.linalg.norm(delta, axis=2).max() * 1000)
    for level, prefix, w in [(0, '', wc), (1, 'L1_', wf)]:
        v = rms(data[prefix + 'exact'][1] - data[prefix + 'frozen'][1], w)
        series[f'L{level}_change'] = v
        results[f'L{level}_exact_vs_frozen'] = summary(v)
    original = np.load(ROOT / 'output/bunny_pcg_no_ground_20261124/mu0/positions.npy', mmap_mode='r')[:31]
    results['frozen_repeat_vs_original'] = summary(rms(data['frozen'][1] - original, wc))
    results['scope'] = 'Zero friction, no ground, 0.3s; exact gradient at every iteration, not event-triggered rescue.'
    results['timing_limitation'] = 'Other GPU simulations were running; wall times are not isolated benchmarks.'
    (OUT / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    time = np.arange(31) * .01
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for mode, color in [('frozen', '#1684b8'), ('exact', '#dd7430')]:
        axes[0].plot(time, series[mode + '_parent'], label=mode, color=color)
    for level in (0, 1):
        axes[1].plot(time, series[f'L{level}_change'], label=f'L{level}')
    axes[0].set_title('L1 versus its own prolonged L0')
    axes[1].set_title('Exact versus frozen: same-level change')
    for ax in axes:
        ax.set_xlabel('Physical time (s)'); ax.set_ylabel('Area-weighted RMS (mm)')
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(OUT / 'comparison.png', dpi=170)
    np.savetxt(OUT / 'comparison.csv', np.column_stack([time, *series.values()]),
               delimiter=',', header=','.join(['time_s', *series]), comments='')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
