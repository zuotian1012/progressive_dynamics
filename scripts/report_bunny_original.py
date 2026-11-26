"""Audit and export the three independent stock-libuipc bunnycloth runs."""
import json
from pathlib import Path
import subprocess
import sys
import re
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root = Path(sys.argv[1]).resolve()
assets = Path('output/bunny_paper_assets')
bunny = np.load(assets / 'bunny.npz')
rows, trajectories, meshes = [], [], []
for level in range(3):
    folder = root / f'L{level}'
    meta = json.loads((folder / 'run.json').read_text())
    records = json.loads((folder / 'steps.json').read_text())
    if meta['status'] == 'running':
        log = folder.with_suffix('.log').read_text(encoding='utf-8-sig')
        failures = [s for s in log.splitlines() if 'Exits with Max Iteration' in s]
        if not failures:
            raise RuntimeError('Run is still active or termination reason is unknown')
        timestamps = re.findall(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]', log)
        elapsed = (datetime.fromisoformat(timestamps[-1]) - datetime.fromisoformat(timestamps[0])).total_seconds()
        meta.update(status='native_aborted', completed=False, failed_step=meta['completed_steps']+1,
                    error=failures[-1], physics_seconds=sum(r['seconds'] for r in records),
                    wall_seconds=elapsed, wall_seconds_source='estimated from native log timestamps')
        (folder / 'run.json').write_text(json.dumps(meta, indent=2))
    n = meta['completed_steps']
    assert not meta['psq_enabled'] and not meta.get('isolated_source')
    assert not meta.get('force_updates') and not meta.get('coarse_preview')
    x = np.load(folder / 'accepted_positions.npy')
    mesh = np.load(assets / f'mesh_{level}.npz')
    np.testing.assert_array_equal(x[0], mesh['vertices'])
    assert x.shape == (n+1, len(mesh['vertices']), 3) and np.isfinite(x).all()
    assert len(records) == n
    assert all(r['native']['converged'] and r['native']['completed'] and r['fixed_error'] == 0 for r in records)
    subprocess.run([sys.executable, 'scripts/report_libuipc_probe.py', str(folder)],
                   stdout=(folder / 'report.log').open('w'), check=True)
    audit = json.loads((folder / 'audit.json').read_text())
    with (folder / 'final_cloth.obj').open('w') as f:
        for v in x[-1]: f.write('v ' + ' '.join(map(str, v)) + '\n')
        for face in mesh['faces']: f.write('f ' + ' '.join(map(str, face + 1)) + '\n')
    np.savez_compressed(folder / 'trajectory.npz', positions=x, faces=mesh['faces'],
                        times=np.arange(n+1) * meta['dt'],
                        bunny_vertices=bunny['vertices'], bunny_faces=bunny['faces'])
    rows.append(dict(level=level, vertices=len(mesh['vertices']), steps=n,
                     all_saved_frames_converged=True,
                     completed=meta['completed'], failed_step=meta.get('failed_step'), error=meta.get('error'),
                     physical_seconds=n*meta['dt'], wall_seconds=meta['wall_seconds'],
                     physics_seconds=meta['physics_seconds'],
                     max_newton_iterations=audit['max_newton_iterations'],
                     max_principal_stretch=audit['max_principal_stretch'],
                     all_frames_strictly_converged=audit['all_frames_strictly_converged']))
    trajectories.append(x)
    meshes.append(mesh)
    print(f'L{level}: audited and exported', flush=True)

# Common view at first draping and at the end, using actual saved positions.
fig = plt.figure(figsize=(15, 9))
for row in range(2):
    frames = [min(50,len(x)-1) if row==0 else len(x)-1 for x in trajectories]
    pts = np.concatenate([bunny['vertices']] + [x[f] for x,f in zip(trajectories,frames)])
    center = (pts.min(0) + pts.max(0)) / 2
    radius = np.ptp(pts, axis=0).max() * .55
    for level, (x, mesh) in enumerate(zip(trajectories, meshes)):
        frame = frames[level]
        ax = fig.add_subplot(2, 3, row * 3 + level + 1, projection='3d')
        triangles, colors = [], []
        for v, faces, color in [(bunny['vertices'], bunny['faces'], [.6,.62,.65]),
                                 (x[frame], mesh['faces'], [.29,.62,.85])]:
            tri = v[faces]
            normal = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
            normal /= np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-20)
            triangles.append(tri)
            colors.append((.4+.6*np.abs(normal @ np.array([-.3,-.4,.866])))[:,None] * color)
        ax.add_collection3d(Poly3DCollection(np.concatenate(triangles), facecolors=np.concatenate(colors), edgecolors='none'))
        ax.set(xlim=(center[0]-radius,center[0]+radius), ylim=(center[1]-radius,center[1]+radius), zlim=(center[2]-radius,center[2]+radius))
        ax.view_init(18,-65)
        ax.set_box_aspect((1,1,1))
        ax.set_axis_off()
        ax.set_title(f'L{level} | {len(mesh["vertices"])} vertices | t={frame*.01:.2f} s' + (' | last accepted' if row else ''))
fig.suptitle('Stock libuipc 0.0.28 | Independent direct simulations | PSQ OFF')
fig.tight_layout()
fig.savefig(root / 'comparison.png', dpi=140)
plt.close(fig)
(root / 'summary.json').write_text(json.dumps(rows, indent=2))
lines = ['Original libuipc bunnycloth direct simulations',
         'pyuipc 0.0.28, installed wheel; package RECORD verification in runtime_verification.json.',
         'Each level starts independently from its own original rest mesh, with zero initial velocity.',
         'Target: 400 steps x dt 0.01 s = 4.0 s. See per-level completion below. All SAVED frames converged; bunny remained fixed.',
         'NeoHookeanShell + DiscreteShellBending; Young=30000, Poisson=0.3, density=200, full thickness=0.00007.',
         'Gravity=(0,0,-9.81), friction=0.3, d_hat=0.001; native adaptive contact stiffness.',
         'Newton velocity tolerance=0.001 (native max-axis), relative/adaptive tolerance disabled, cap=1000.',
         'No PSQ, no coarse trajectory injection, no custom hard strain barrier, no forced acceptance.',
         'These are native model baselines, not identical constitutive/solver settings to C-IPC or the paper.',
         'The cap100 L0 attempt stopped at frame 71 and is retained separately; final three runs use cap1000.',
         '', 'level,vertices,steps,physical_s,wall_s,advance_retrieve_s,max_newton,max_stretch']
for r in rows:
    lines.append(f'L{r["level"]},{r["vertices"]},{r["steps"]},{r["physical_seconds"]:.2f},{r["wall_seconds"]:.6f},{r["physics_seconds"]:.6f},{r["max_newton_iterations"]},{r["max_principal_stretch"]:.8f}')
    lines.append(f'Completed target: {r["completed"]}; failed step: {r["failed_step"]}; error: {r["error"]}')
lines += ['', 'Per level: accepted_positions.npy, trajectory.npz, final_cloth.obj, run.json, steps.json, audit.json, contact.gif, final.png, metrics.png.',
          'Reproduce from repository root with a fresh output directory for each level:',
          '.venv/Scripts/python.exe scripts/probe_libuipc_bunny.py --output <new-output>/L0 --level 0 --steps 400 --step-budget 400 --wall-budget 3600 --max-newton 1000',
          'Repeat with --level 1 and --level 2. Preserve stdout/stderr as sibling L0.log, L1.log, L2.log.',
          '.venv/Scripts/python.exe scripts/report_bunny_original.py <new-output>']
(root / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
print(json.dumps(rows, indent=2))
