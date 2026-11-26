"""Audit physical trajectories and render a synchronized four-level comparison."""
import argparse
import csv
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import ipctk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.sparse import load_npz

from run_hanging_square_demo import DT, STEPS, ROOT, run_path, write_json


def main(root):
    out = root/'report'
    out.mkdir(exist_ok=True)
    data, audits, rows = {}, {}, []
    for method in ('semi', 'velpro'):
        for level in range(4):
            path = run_path(root, method, level)
            meta = json.loads((path/'run.json').read_text())
            steps = json.loads((path/'steps.json').read_text())
            x = np.load(path/'positions.npy')
            if meta['status'] != 'completed' or len(x) != STEPS+1 or len(steps) != STEPS:
                raise ValueError(f'Incomplete trajectory: {path}')
            mesh = dict(np.load(root/'assets'/f'mesh_{level}.npz'))
            rest, f, pins = mesh['vertices'], mesh['faces'], mesh['pin_indices']
            edges = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
            if path.name not in audits:
                cm = ipctk.CollisionMesh(rest, edges.astype(np.int32), f.astype(np.int32))
                intersections = [i for i, y in enumerate(x) if ipctk.has_intersections(cm, y)]
                pin_error = float(np.abs(x[:, pins]-rest[pins]).max())
                if not np.isfinite(x).all() or pin_error > 1e-10:
                    raise ValueError(f'Invalid saved geometry: {path}')
                audits[path.name] = dict(accepted_frames=len(x)-1, self_intersection_frames=intersections,
                    max_pin_error_m=pin_error, max_stretch=max(s['stretch'] for s in steps),
                    newton_converged_frames=sum(bool(s['native']['converged']) for s in steps),
                    coarse_preview_frames=sum(bool(s['native']['preview_non_descent']) for s in steps),
                    elapsed_s=meta['elapsed_s'])
                print(f'Audited {path.name}: {len(intersections)} self-intersecting frames', flush=True)
            tip = np.flatnonzero(np.isclose(mesh['uv'][:, 0], .5) & np.isclose(mesh['uv'][:, 1], 0))[0]
            angle = np.rad2deg(np.arctan2(x[:, tip, 1], 1.5-x[:, tip, 2]))
            tri = rest[f]
            mass = np.zeros(len(rest))
            area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
            np.add.at(mass, f.ravel(), np.repeat(area/3, 3))
            if level:
                U = load_npz(root/'assets'/f'map_{level-1}_{level}.npz')
                parent = data[method, level-1]['x']
                diff = x-np.asarray([U @ y for y in parent])
                rms = np.sqrt(np.einsum('i,tij,tij->t', mass/mass.sum(), diff, diff))*1000
            else:
                rms = np.zeros(len(x))
            data[method, level] = dict(x=x, mesh=mesh, angle=angle, rms=rms)
            for i in range(len(x)):
                rows.append((method, level, i, i*DT, angle[i], x[i, tip, 1], x[i, tip, 2], rms[i]))
    write_json(out/'audit.json', audits)
    with (out/'per_frame.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('method', 'level', 'frame', 'time_s', 'bottom_center_swing_deg', 'bottom_center_y_m', 'bottom_center_z_m', 'rms_vs_prolongated_parent_mm'))
        writer.writerows(rows)
    differences = {}
    for level in range(4):
        distance = np.linalg.norm(data['semi', level]['x']-data['velpro', level]['x'], axis=2)
        differences[f'L{level}'] = dict(max_vertex_distance_mm=float(distance.max()*1000),
            final_vertex_rms_mm=float(np.sqrt(np.mean(distance[-1]**2))*1000))
    write_json(out/'method_differences.json', differences)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = ['#586578', '#ca7045', '#15888e', '#695daf']
    for row, method in enumerate(('semi', 'velpro')):
        for level in range(4):
            d = data[method, level]
            axes[row, 0].plot(np.arange(STEPS+1)*DT, d['angle'], color=colors[level], label=f'L{level}')
            if level:
                axes[row, 1].plot(np.arange(STEPS+1)*DT, d['rms'], color=colors[level], label=f'L{level}')
        axes[row, 0].set_title(('PD / Semi-diagonal' if method == 'semi' else 'PD++ / VelPro')+' : swing')
        axes[row, 1].set_title('Coarse/fine position difference\n(not reference-solution error)', fontsize=11)
        axes[row, 0].set_ylabel('Bottom-center angle (deg)')
        axes[row, 1].set_ylabel('Mass-weighted RMS (mm)')
        for ax in axes[row]:
            ax.set_xlabel('Physical time (s)')
            ax.legend(ncol=4)
            ax.grid(alpha=.2)
    fig.savefig(out/'metrics.png', dpi=150)
    plt.close(fig)

    # One orthographic camera and one scale for every frame and every panel.
    direction = np.array([1.4, -3., .85]); direction /= np.linalg.norm(direction)
    right = np.cross([0, 0, 1], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    basis = np.column_stack((right, up))
    center = np.array([0., 0., 1.])
    project = lambda p: (p-center) @ basis
    light = np.array([-.3, -1., .8]); light /= np.linalg.norm(light)
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), dpi=100, facecolor='#f4f6f9')
    fig.subplots_adjust(left=.025, right=.985, top=.82, bottom=.10, hspace=.30, wspace=.08)
    fig.text(.04, .945, 'TWO-CORNER CLOTH SWING', fontsize=24, weight='bold', color='#213147')
    fig.text(.04, .905, '1 m square  |  35-degree release  |  gravity only  |  identical camera and physical time', fontsize=12, color='#526078')
    time_label = fig.text(.96, .945, '', ha='right', fontsize=17, color='#213147')
    fig.text(.04, .855, 'PD  /  SEMI-DIAGONAL', fontsize=14, weight='bold', color='#213147')
    fig.text(.04, .465, 'PD++  /  VELPRO', fontsize=14, weight='bold', color='#213147')
    fig.text(.04, .055, 'Orange: two fixed corners   |   Shared L0   |   PSQ elastic proxy: L3   |   dt = 0.01 s', fontsize=11, color='#526078')
    total_intersections = sum(len(a['self_intersection_frames']) for a in audits.values())
    note = 'All 2,100 steps accepted; no self-intersections in saved frames.' if total_intersections == 0 else f'Self-contact disabled; {total_intersections} saved frames contain self-intersections.'
    fig.text(.04, .025, note, fontsize=10, color='#526078')
    polygons = []
    for row, method in enumerate(('semi', 'velpro')):
        for level in range(4):
            ax, d = axes[row, level], data[method, level]
            mesh = d['mesh']; faces = mesh['faces']; rest = mesh['vertices']
            ax.set_facecolor('#f4f6f9')
            ax.set(xlim=(-.78, .78), ylim=(-.68, .80), aspect='equal')
            ax.axis('off')
            pinpos = rest[mesh['pin_indices']]
            for p in pinpos:
                line = project(np.array([p, p+[0, 0, .15]]))
                ax.plot(*line.T, color='#e7794b', lw=2, zorder=5)
            ax.scatter(*project(pinpos).T, s=35, color='#e7794b', edgecolor='white', linewidth=1, zorder=6)
            poly = PolyCollection([], linewidths=.28 if level < 2 else .12, edgecolors=(.03, .17, .22, .24))
            ax.add_collection(poly)
            uv = mesh['uv'][faces].mean(axis=1)
            checker = (np.floor(uv[:, 0]*4)+np.floor(uv[:, 1]*4)).astype(int) % 2
            base = np.where(checker[:, None] == 0, [.16, .57, .65], [.33, .72, .76])
            ax.set_title(f'L{level}  |  {len(rest):,} vertices\n{len(faces):,} triangles'+('  /  shared' if level == 0 else ''), fontsize=11, color='#34465c', pad=0)
            polygons.append((poly, d, base))
    ids = np.arange(0, STEPS+1, 4)
    gifs = []
    with imageio.get_writer(out/'comparison.mp4', fps=25, codec='libx264', macro_block_size=1, quality=8) as writer:
        for frame in ids:
            for poly, d, base in polygons:
                tri = d['x'][frame][d['mesh']['faces']]
                normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
                color = base * (.62+.38*np.abs(normals @ light))[:, None]
                order = np.argsort(tri.mean(axis=1) @ direction)
                poly.set_verts(project(tri)[order])
                poly.set_facecolors(color[order])
            time_label.set_text(f't = {frame*DT:.2f} s')
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(image)
            # GIF is an explicitly labelled half-speed preview.
            gifs.append(image[::2, ::2])
            if frame == 80:
                fig.savefig(out/'poster.png', dpi=150)
            if frame % 100 == 0:
                print(f'Rendered frame {frame}/{STEPS}', flush=True)
    imageio.mimsave(out/'comparison_half_speed.gif', gifs, duration=80, loop=0)
    plt.close(fig)
    write_json(out/'render.json', dict(frames=ids.tolist(), video_fps=25, video_playback_rate=1., gif_playback_rate=.5,
        geometry='accepted simulated vertices, no smoothing or interpolated geometry', camera_direction=direction.tolist(),
        coarse_preview_frames=sum(a['coarse_preview_frames'] for a in audits.values())))
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=ROOT/'output/hanging_square_demo')
    main(parser.parse_args().root.resolve())
