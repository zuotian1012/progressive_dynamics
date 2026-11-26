"""Compare completed first-fold runs using their actual saved physical states."""
import argparse
import copy
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
    parser.add_argument('--skip-animation', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    selection = json.loads((root / 'selection.json').read_text())
    count = len(json.loads((Path(selection['assets']) / 'manifest.json').read_text())['levels'])
    paths = [root / f'L{k}/run' for k in range(count)]
    data = [load_run(path) for path in paths]
    plan = json.loads((root / 'L0/plan.json').read_text())
    stages = plan['stages_physical_steps']
    end = stages['observe_end']
    out = root / 'comparison'
    out.mkdir(exist_ok=True)
    reference = data[0]['meta']
    rows = []
    for k, (path, d) in enumerate(zip(paths, data)):
        m = d['meta']
        assert m['level'] == k and m['completed'] and m['completed_steps'] == end
        assert m['finest_level'] == count - 1
        for key in ['trajectory_sha256', 'robot_q_sha256', 'dt', 'material_parameters',
                    'runtime_hashes', 'robot_drive_strength']:
            assert m[key] == reference[key], key
        expected_config = copy.deepcopy(reference['config'])
        budget = m['parameters'].get('line_search_max_iterations')
        if budget is not None:
            assert 8 <= budget <= 128 and m['solver_overrides']['line_search.max_iter'] == budget
            expected_config['line_search']['max_iter'] = budget
        assert m['config'] == expected_config
        audit = json.loads((path / 'report/audit.json').read_text())
        fold = json.loads((path / 'report/custom_fold_audit.json').read_text())
        records = json.loads((path / 'steps.json').read_text())
        unchanged_tail = 0
        for step in range(end, 0, -1):
            if not np.array_equal(d['x'][step], d['x'][step - 1]):
                break
            unchanged_tail += 1
        preview_tail = 0
        for record in reversed(records):
            if not record['native']['preview_non_descent']:
                break
            preview_tail += 1
        if k:
            assert Path(m['lower_run']).resolve() == paths[k - 1]
            assert audit['velocity_transfer_audit']['all_injected_velocities_verified']
        if k == count - 1:
            assert m['finest_requires_convergence'] and m['preview_stopped_steps'] == 0
            assert all(r['native']['converged'] and not r['native']['preview_non_descent']
                       and not r['native']['hit_line_search_limit']
                       and not r['native']['hit_newton_limit'] for r in records)
        rows.append(dict(level=k, vertices=m['vertices'], faces=m['faces'],
                         completed_steps=end, physics_seconds=m['physics_seconds'],
                         native_converged_steps=m['native_converged_steps'],
                         preview_stopped_steps=m['preview_stopped_steps'],
                         newton_limit_steps=m['newton_limit_steps'],
                         line_search_budget=m['config']['line_search']['max_iter'],
                         trailing_identical_cloth_steps=unchanged_tail,
                         trailing_preview_steps=preview_tail,
                         pickup=measure(d, stages['lift_hold_end']), fold=fold))
    low = np.min([d['x'][:end + 1].min(axis=(0, 1)) for d in data], axis=0)
    high = np.max([d['x'][:end + 1].max(axis=(0, 1)) for d in data], axis=0)
    bounds = [(min(.15, float(low[0]) - .03), max(1.18, float(high[0]) + .03)),
              (min(-.55, float(low[1]) - .03), max(.55, float(high[1]) + .03)),
              (min(.78, float(low[2]) - .03), max(1.26, float(high[2]) + .03))]
    for d in data:
        d['plot_bounds'] = bounds
    result = dict(full_first_fold_trajectory_completed=True, common_steps=end,
                  common_physical_seconds=end * reference['dt'], levels=rows,
                  total_physics_seconds=sum(row['physics_seconds'] for row in rows),
                  paths=[str(path) for path in paths], plot_bounds=bounds,
                  limitations=['Motion completion is not a successful fold verdict.',
                               'Preview exits are not Newton convergence.',
                               'A motionless tail of preview exits is not proof of physical equilibrium.',
                               'Only actual saved cloth positions are displayed.',
                               'Material-point tracking is not contact-force evidence.',
                               'Intersection checks sample times, not continuous motion.'])
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))
    fig = plt.figure(figsize=(4.6 * count, 12))
    for row, (view, elev, azim) in enumerate([('Overview', 48, -125),
                                            ('Overhead', 90, -90), ('Table edge', 22, 55)]):
        for k, d in enumerate(data):
            ax = fig.add_subplot(3, count, row * count + k + 1, projection='3d')
            draw(ax, d, end, f'L{k} | {view} | {end * reference["dt"]:.2f}s', False)
            ax.view_init(elev=elev, azim=azim)
            if view == 'Overhead':
                ax.set_proj_type('ortho'); ax.set_zticks([]); ax.set_zlabel('')
    fig.suptitle('Completed first-fold trajectory: actual final cloth states')
    fig.tight_layout(h_pad=3)
    fig.savefig(out / 'four_levels_final.png', dpi=120)
    plt.close(fig)
    chosen = [('Settled', 60), ('Lift hold', stages['lift_hold_end']),
              ('Before fold', stages['transfer_hold_end']),
              ('Fold arc end', stages['fold_arc_end']), ('Opened', stages['opened']), ('Final', end)]
    fig = plt.figure(figsize=(4 * len(chosen), 3.8 * count))
    for k, d in enumerate(data):
        for col, (name, step) in enumerate(chosen):
            ax = fig.add_subplot(count, len(chosen), k * len(chosen) + col + 1, projection='3d')
            draw(ax, d, step, f'L{k} | {name} | {step * reference["dt"]:.2f}s', False)
    fig.tight_layout(h_pad=3)
    fig.savefig(out / 'four_levels_stages.png', dpi=110)
    plt.close(fig)
    if not args.skip_animation:
        images = []
        frames = np.unique(np.linspace(0, end, 120).astype(int))
        columns = 2; grid_rows = (count + 1) // 2
        for index, step in enumerate(frames):
            fig = plt.figure(figsize=(12, 4.5 * grid_rows), dpi=85)
            for k, d in enumerate(data):
                ax = fig.add_subplot(grid_rows, columns, k + 1, projection='3d')
                draw(ax, d, step, f'L{k} | {d["meta"]["faces"]:,} triangles', False)
                ax.view_init(elev=25, azim=55)
            fig.suptitle(f'Actual four-level full first-fold motion | {step * reference["dt"]:.2f}s')
            fig.tight_layout(h_pad=2)
            fig.canvas.draw()
            frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            images.append(frame)
            if index in (0, len(frames) // 2, len(frames) - 1):
                frame.save(out / f'animation_{index:03d}.png')
            plt.close(fig)
            if index % 20 == 0:
                print(f'Rendered {index + 1}/{len(frames)} frames', flush=True)
        duration = max(20, round(end * reference['dt'] * 100 / len(images)) * 10)
        images[0].save(out / 'four_levels_full_fold.gif', save_all=True,
                       append_images=images[1:], duration=duration, loop=0)
    print(json.dumps(dict(completed=True, output=str(out), levels=count, steps=end)), flush=True)


if __name__ == '__main__':
    main()
