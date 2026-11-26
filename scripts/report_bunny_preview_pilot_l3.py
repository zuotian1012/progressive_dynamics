"""Compare L0-L3 short controls on adjacent levels and a common L7 surface."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from report_bunny_preview_pilot import OUT, ASSETS, weights, rms, summary, plt, ShellProlongation


def name(level, mode):
    return mode if level == 0 else f'L{level}_{mode}'


def main():
    data = {}; result = {'runs': {}, 'metrics': {}, 'same_level_changes': {}}
    for level in range(4):
        mesh = np.load(ASSETS / f'mesh_{level}.npz')
        for mode in ('frozen', 'exact'):
            key = name(level, mode); run = OUT / key
            m = json.loads((run / 'run.json').read_text())
            assert m['completed'] and m['completed_steps'] == 30 and m['dt'] == .01
            assert m['exact_offset_gradient'] == (mode == 'exact')
            assert m['level'] == level and (level == 0 or m['method'] == 'velpro')
            a = json.loads((run / 'no_ground_audit.json').read_text())
            assert a['passed'] and a['saved_states_checked'] == 31
            assert a['positions_sha256'] == hashlib.sha256((run / 'positions.npy').read_bytes()).hexdigest()
            if level:
                parent = OUT / name(level - 1, mode)
                assert Path(m['lower_run']).resolve() == parent.resolve()
                assert m['lower_sha256'] == hashlib.sha256((parent / 'positions.npy').read_bytes()).hexdigest()
            x = np.load(run / 'positions.npy')
            with np.load(run / 'operator/operator.npz') as op:
                proxy = ShellProlongation(mesh['vertices'], op['fine_rest'], mesh['faces'],
                    csr_matrix((op['U_data'], op['U_indices'], op['U_indptr']), shape=tuple(op['U_shape'])), op['anchors'])
                fw = weights(dict(vertices=op['fine_rest'], faces=op['fine_faces']))
            projected = np.array([proxy.position(state) for state in x])
            data[key] = dict(meta=m, x=x, mesh=mesh, projected=projected, weights=weights(mesh))
            rows = json.loads((run / 'steps.json').read_text())
            result['runs'][key] = dict(converged=m['native_steps'], preview=m['preview_steps'],
                wall_seconds=m['wall_seconds'], newton_total=sum(r['native']['newton_iterations'] for r in rows))
        a, b = data[name(level, 'frozen')], data[name(level, 'exact')]
        for field in ('config', 'material', 'ground', 'contact_friction', 'library_sha256', 'bundle_sha256'):
            assert a['meta'][field] == b['meta'][field], field
        np.testing.assert_array_equal(a['x'][0], b['x'][0])
        result['same_level_changes'][str(level)] = summary(rms(b['x'] - a['x'], a['weights']))
        print('Verified pair', level, flush=True)

    series = {}; time = np.arange(31) * .01
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for level in range(1, 4):
        for mode, color in [('frozen', '#1684b8'), ('exact', '#dd7430')]:
            child, parent = data[name(level, mode)], data[name(level - 1, mode)]
            transfer = ShellProlongation(parent['mesh']['vertices'], child['mesh']['vertices'], parent['mesh']['faces'],
                load_npz(ASSETS / f'map_{level-1}_{level}.npz'), np.load(ASSETS / f'anchors_{level-1}_{level}.npy'))
            delta = child['x'] - np.array([transfer.position(x) for x in parent['x']])
            comparisons = dict(parent=rms(delta, child['weights']),
                own_l0=rms(child['projected'] - data[mode]['projected'], fw),
                fixed_frozen_l0=rms(child['projected'] - data['frozen']['projected'], fw))
            key = name(level, mode)
            result['metrics'][key] = {basis: summary(v) for basis, v in comparisons.items()}
            result['metrics'][key]['parent']['max_vertex_mm'] = float(1000*np.linalg.norm(delta, axis=2).max())
            for basis, v in comparisons.items(): series[f'{key}_{basis}_mm'] = v
            for row, basis in enumerate(('parent', 'own_l0')):
                axes[row, level-1].plot(time, comparisons[basis], label=mode, color=color)
        axes[0, level-1].set_title(f'L{level} versus its own prolonged L{level-1}')
        axes[1, level-1].set_title(f'L{level} versus own L0 on common L7 surface')
    for ax in axes.flat:
        ax.set_xlabel('Physical time (s)'); ax.set_ylabel('Area-weighted RMS (mm)')
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(OUT / 'comparison_L3.png', dpi=170)
    result['scope'] = 'Zero friction, no ground, 30 steps at dt=.01; L1-L3 VelPro; full PSQ gradient at every iteration versus frozen-offset gradient.'
    result['limitations'] = ['Single realization; no repeated-run uncertainty estimate.',
        'Other GPU jobs were running; times are not isolated benchmarks.',
        'Each hierarchy uses its own L0. fixed_frozen_l0 is also reported as a fixed reference, not physical ground truth.',
        'This is not an event-triggered rescue experiment.']
    (OUT / 'comparison_L3.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savetxt(OUT / 'comparison_L3.csv', np.column_stack([time, *series.values()]), delimiter=',',
        header=','.join(['time_s', *series]), comments='')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
