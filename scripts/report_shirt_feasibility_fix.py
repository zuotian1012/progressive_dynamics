"""Compare recorded L2 attempts; missing baseline states are left blank."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import load_run, draw, plt, Image
from report_shirt_pickup_hierarchy import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    original = Path('output/shirt_four_level_full_fold_20261118')
    attempts = [('Original LS8', original / 'L2_ls8_failed/run'),
                ('Original LS32', original / 'L2/run'),
                ('Coordinate guard + diag LS8', root / 'L2_guard_v4_diag/run'),
                ('Coordinate guard + MAS LS8', root / 'L2/run')]
    data = [load_run(path) for _, path in attempts]
    rows = []
    for (label, path), d in zip(attempts, data):
        m = d['meta']; n = m['completed_steps']
        assert m['status'] != 'running'
        audit = json.loads((path / 'report/audit.json').read_text())
        records = json.loads((path / 'steps.json').read_text())
        assert len(records) == n
        assert audit['velocity_transfer_audit']['all_injected_velocities_verified']
        same_tail = 0
        for k in range(n, 0, -1):
            if not np.array_equal(d['x'][k], d['x'][k - 1]): break
            same_tail += 1
        preview_tail = 0
        for record in reversed(records):
            if not record['native']['preview_non_descent']: break
            preview_tail += 1
        rows.append(dict(label=label, status=m['status'], completed_steps=n,
            requested_steps=m['steps'], physics_seconds=m['physics_seconds'],
            native_steps=m['native_converged_steps'], preview_steps=m['preview_stopped_steps'],
            newton_limit_steps=m['newton_limit_steps'], line_search_limit_steps=m['line_search_limit_steps'],
            trailing_identical_cloth_steps=same_tail,
            trailing_preview_steps=preview_tail, max_stretch=audit['maximum_principal_stretch'],
            minimum_singular_value=audit['minimum_singular_value'],
            maximum_robot_surface_error_mm=1000 * audit['maximum_body_surface_tracking_error_m'],
            screening=audit['screening']))
        d['plot_bounds'] = [(.15, 1.25), (-.55, .55), (.77, 1.26)]
    for key in ['material_parameters', 'mesh_sha256', 'robot_q_sha256',
                'trajectory_sha256', 'psq_bundle_sha256', 'lower_sha256', 'map_sha256']:
        assert data[0]['meta'][key] == data[-1]['meta'][key], key
    expected_config=json.loads(json.dumps(data[0]['meta']['config']))
    preconditioner=data[-1]['meta']['parameters'].get('fem_preconditioner')
    if preconditioner is not None:
        assert data[-1]['meta']['solver_overrides']['linear_system.fem_preconditioner']==preconditioner
        expected_config['linear_system']['fem_preconditioner']=preconditioner
    assert expected_config==data[-1]['meta']['config']
    latest = data[-1]; n = latest['meta']['completed_steps']
    result = dict(attempts=rows, pickup=measure(latest, min(1076, n)),
        physical_inputs_and_ls8_settings_match=True,linear_preconditioner_override=preconditioner,
        completed_full_L2=bool(latest['meta']['completed'] and n == 3476),
        fold_quality_certified=False,
        limitations=['Preview exits are reported separately from Newton convergence.',
                     'An identical cloth tail does not establish physical equilibrium.',
                     'Missing original states are blank; no prolonged or invented states are shown.',
                     'Local numerical regression uses saved triangles with synthetic directions.',
                     'See build_provenance.json for isolated runtime build ancestry.'])
    out = root / 'comparison'; out.mkdir(exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))

    def panel(ax, d, step, title):
        if step > d['meta']['completed_steps']:
            ax.set_axis_off()
            ax.text2D(.5, .6, title + '\nSTOPPED', transform=ax.transAxes, ha='center', color='#b04435')
            ax.text2D(.5, .4, 'No simulated state at this time', transform=ax.transAxes, ha='center')
            return
        draw(ax, d, step, title, False)
        ax.view_init(elev=25, azim=55)

    fig = plt.figure(figsize=(12, 9))
    for i, (name, step) in enumerate([('Settled', 60), ('Lift and hold', 1076),
                                      ('Fold arc', 2216), ('Final', 3476)]):
        panel(fig.add_subplot(2, 2, i + 1, projection='3d'), latest, step,
              f'L2 {name} | t={step / 120:.2f}s')
    fig.tight_layout(h_pad=2); fig.savefig(out / 'stages.png', dpi=125); plt.close(fig)
    images = []
    for i, step in enumerate(np.unique(np.linspace(0, 3476, 120).astype(int))):
        fig = plt.figure(figsize=(12, 5), dpi=85)
        for j, index in enumerate([0, len(data)-1]):
            panel(fig.add_subplot(1, 2, j + 1, projection='3d'), data[index], step, attempts[index][0])
        fig.suptitle(f'Actual L2 states | t={step / 120:.2f}s')
        fig.tight_layout(); fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)
        if i % 30 == 0: print(f'Rendered {i + 1}/120', flush=True)
    images[-1].save(out / 'final_comparison.png')
    images[0].save(out / 'L2_comparison.gif', save_all=True, append_images=images[1:], duration=240, loop=0)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
