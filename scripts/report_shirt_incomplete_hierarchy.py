"""Render available states of an incomplete hierarchy without inventing missing frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import ipctk
from shirt_demo.report import load_run, draw, plt, Image
from report_shirt_pickup_hierarchy import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'comparison_incomplete'
    out.mkdir(exist_ok=True)
    selection = json.loads((root / 'selection.json').read_text())
    count = len(json.loads((Path(selection['assets']) / 'manifest.json').read_text())['levels'])
    plan = json.loads((root / 'L0/plan.json').read_text())
    requested = plan['stages_physical_steps']['observe_end']
    data = []
    rows = []
    for k in range(count):
        path = root / f'L{k}/run'
        if not (path / 'run.json').exists():
            data.append(None)
            rows.append(dict(level=k, status='not_started', completed_steps=0,
                             reason='No complete immediate-lower velocity trajectory'))
            continue
        d = load_run(path); data.append(d)
        m = d['meta']; n = m['completed_steps']
        assert m['status'] != 'running' and n <= requested and m['level'] == k
        audit = json.loads((path / 'report/audit.json').read_text())
        if k:
            assert audit['velocity_transfer_audit']['all_injected_velocities_verified']
        records = json.loads((path / 'steps.json').read_text())
        assert len(records) == n and np.isfinite(d['x'][:n + 1]).all()
        unchanged = 0
        for step in range(n, 0, -1):
            if not np.array_equal(d['x'][step], d['x'][step - 1]): break
            unchanged += 1
        tail_preview = 0
        for record in reversed(records):
            if not record['native']['preview_non_descent']: break
            tail_preview += 1
        f = d['f']
        edges = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0).astype(np.int32)
        mesh = ipctk.CollisionMesh(d['x'][0], edges, f.astype(np.int32))
        samples = [dict(step=step, self_intersection=bool(ipctk.has_intersections(mesh, d['x'][step])))
                   for step in sorted({min(s, n) for s in [0, 60, 596, 1076, 1616, 1916, 2216, 2576, 2936, n]})]
        row = {key: m.get(key) for key in ['level', 'status', 'vertices', 'faces', 'completed_steps',
                'physics_seconds', 'native_converged_steps', 'preview_stopped_steps', 'newton_limit_steps']}
        row.update(physical_seconds=n * m['dt'], line_search_budget=m['config']['line_search']['max_iter'],
                   failed_step=m.get('failed_step'), trailing_identical_cloth_steps=unchanged,
                   trailing_preview_steps=tail_preview, sampled_self_intersections=samples,
                   maximum_robot_surface_tracking_error_mm=audit['maximum_body_surface_tracking_error_m'] * 1000,
                   maximum_principal_stretch=audit['maximum_principal_stretch'],
                   pickup=measure(d, min(1076, n)),
                   table=json.loads((path / 'report/table_extent_audit.json').read_text()),
                   audit_screening=audit['screening'])
        rows.append(row)
    actual = [d for d in data if d is not None]
    low = np.min([d['x'][:d['meta']['completed_steps'] + 1].min(axis=(0, 1)) for d in actual], axis=0)
    high = np.max([d['x'][:d['meta']['completed_steps'] + 1].max(axis=(0, 1)) for d in actual], axis=0)
    bounds = [(min(.15, float(low[0]) - .03), max(1.18, float(high[0]) + .03)),
              (min(-.55, float(low[1]) - .03), max(.55, float(high[1]) + .03)),
              (min(.78, float(low[2]) - .03), max(1.26, float(high[2]) + .03))]
    for d in actual: d['plot_bounds'] = bounds
    attempts = []
    for label in ['L0', 'L1_attempt1_failed', 'L1', 'L2_ls8_failed', 'L2']:
        m = json.loads((root / label / 'run/run.json').read_text())
        attempts.append(dict(label=label, level=m['level'], status=m['status'],
                             steps=m['completed_steps'], physics_seconds=m['physics_seconds'],
                             line_search_budget=m['config']['line_search']['max_iter']))
    provenance = json.loads((root / 'provenance.json').read_text())
    preserved = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                    for p, h in provenance['files_sha256'].items())
    assert preserved
    result = dict(full_four_level_fold_completed=False, requested_steps=requested,
                  physical_duration_requested_s=requested * actual[0]['meta']['dt'],
                  levels=rows, attempts=attempts, core_bundle_trajectory_hashes_unchanged=preserved,
                  limitations=['L2 uses an explicitly different 32-trial budget; original 8-trial failure is retained.',
                               'No L3 state exists for this full-fold experiment.',
                               'Missing animation states are blank, never frozen or prolonged from another level.',
                               'L0/L1 have motionless preview tails; this does not demonstrate equilibrium.',
                               'L2 has 8 capped Newton steps and a failed feasibility screening.',
                               'Only sampled intersection checks, not a continuous guarantee.'])
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))

    def panel(ax, level, step):
        d = data[level]
        if d is None or step > d['meta']['completed_steps']:
            ax.set_axis_off()
            title = f'L{level} NOT RUN' if d is None else f'L{level} STOPPED'
            note = 'No complete L2 input' if d is None else f'Last saved: {d["meta"]["completed_steps"] / 120:.2f}s\nNo simulated state at this time'
            ax.text2D(.5, .62, title, transform=ax.transAxes, ha='center', fontsize=15, color='#b04435')
            ax.text2D(.5, .42, note, transform=ax.transAxes, ha='center', fontsize=11)
        else:
            budget = d['meta']['config']['line_search']['max_iter']
            draw(ax, d, step, f'L{level} | {d["meta"]["faces"]:,} faces | LS budget {budget}', False)
            ax.view_init(elev=25, azim=55)

    fig = plt.figure(figsize=(12, 9))
    for k, d in enumerate(data):
        step = d['meta']['completed_steps'] if d is not None else requested
        ax = fig.add_subplot(2, 2, k + 1, projection='3d'); panel(ax, k, step)
        if d is not None: ax.set_title(f'L{k} last saved: {step / 120:.2f}s | {d["meta"]["status"]}')
    fig.suptitle('Last available states: L2 is incomplete, L3 was not run; times differ')
    fig.tight_layout(h_pad=2); fig.savefig(out / 'last_available_states.png', dpi=130); plt.close(fig)
    images = []
    frames = np.unique(np.linspace(0, requested, 120).astype(int))
    for i, step in enumerate(frames):
        fig = plt.figure(figsize=(12, 9), dpi=85)
        for k in range(count): panel(fig.add_subplot(2, 2, k + 1, projection='3d'), k, step)
        fig.suptitle(f'Incomplete four-level fold attempt | {step / 120:.2f}s | missing states left blank')
        fig.tight_layout(h_pad=2); fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()); images.append(frame)
        if i in (0, 60, 90, 119): frame.save(out / f'animation_{i:03d}.png')
        plt.close(fig)
        if i % 20 == 0: print(f'Rendered {i + 1}/{len(frames)}', flush=True)
    images[0].save(out / 'attempt_full_timeline.gif', save_all=True, append_images=images[1:], duration=240, loop=0)
    print('Incomplete experiment report and actual-state animation completed', flush=True)


if __name__ == '__main__': main()
