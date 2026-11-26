"""Actual intrinsic simplification and PSQ transport on a multiscale cape.

All motion is prescribed. No dynamics, collision response, fine relaxation,
vertex smoothing, or replacement of the original fine topology is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_psq_intrinsic_pleats import load, collision_frames
from demo_psq_shallow_pleats import evaluate, rotation, write_json
from report_square_sphere_demo import rasterize

COUNT = 81
BG = '#f4f6f9'
INK = '#213147'
MUTED = '#526078'


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    # Open annular sector: curved neckline and hem, large radial pleats,
    # smaller radial ripples, and an oscillating hem. Units are metres.
    a, b = np.meshgrid(np.linspace(0, 1, 129), np.linspace(0, 1, 65))
    uv = np.column_stack((a.ravel(), b.ravel()))
    theta = np.deg2rad(232) * (a - .5)
    radius = .16 + .43*b + .009*b**3*np.sin(22*np.pi*a)
    large = .031*b**1.1*np.sin(16*np.pi*a + .45*b)
    small = .010*b**2*np.sin(40*np.pi*a + 1.5*b)
    z = .12*(1-b)**2 + large + small + .006*np.sin(6*np.pi*b)*np.sin(np.pi*a)
    vertices = np.column_stack(((radius*np.sin(theta)).ravel(),
                                (radius*np.cos(theta)-.15).ravel(), z.ravel()))
    faces = []
    for j in range(64):
        for i in range(128):
            v = j*129+i
            faces.extend(((v, v+1, v+130), (v, v+130, v+129)))
    faces = np.asarray(faces, dtype=np.int32)
    np.savez(root/'original_fine_mesh.npz', vertices=vertices, faces=faces, uv=uv)
    with (root/'original_fine.obj').open('w') as fp:
        for v in vertices:
            fp.write('v '+' '.join(format(float(x), '.17g') for x in v)+'\n')
        for f in faces:
            fp.write('f '+' '.join(str(int(x)+1) for x in f)+'\n')
    subprocess.run([sys.executable, '-m', 'progressive.prepare_intrinsic',
                    '--input', str(root/'original_fine.obj'), '--targets',
                    '4096', '1024', '256', '96', '--output', str(root/'assets'),
                    '--quadric', 'probabilistic', '--contact-clearance', '.0002'],
                   check=True, cwd=ROOT)


def deform(rest, s):
    """Smooth spatial bend, differential twist, and rigid tilt; identity at s=0."""
    u, v, z = rest.T
    beta = 1.7*s
    if beta > 1e-10:
        y = np.sin(beta*v)/beta - z*np.sin(beta*v)
        h = (1-np.cos(beta*v))/beta + z*np.cos(beta*v)
    else:
        y, h = v.copy(), z.copy()
    twist = 1.05*s*v + .32*s*u
    x = np.column_stack((u*np.cos(twist)+h*np.sin(twist), y,
                         -u*np.sin(twist)+h*np.cos(twist)))
    # Smoothly raise both lateral wings, with asymmetric lift.
    x[:, 2] += s*(.65*u*u + .12*u)
    return x @ rotation(np.deg2rad(58)*s).T


class Camera:
    def __init__(self, states):
        view = np.array([1.05, -2.5, 2.25]); view /= np.linalg.norm(view)
        right = np.cross([0, 0, 1], view); right /= np.linalg.norm(right)
        self.view = view
        self.basis = np.column_stack((right, np.cross(view, right)))
        lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
        for state in states:
            projected = np.asarray(state).reshape(-1, 3) @ self.basis
            lower = np.minimum(lower, projected.min(axis=0))
            upper = np.maximum(upper, projected.max(axis=0))
        self.center = (lower+upper)/2
        self.scale = 1.30/np.max(upper-lower)

    def draw(self, ax, vertices, faces, wire=False, size=650):
        tri = vertices[faces]
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
        light = np.array([-.8, -.5, 1.1]); light /= np.linalg.norm(light)
        # Uniform material, geometric normals: all visible wrinkles are mesh geometry.
        colors = np.array([.15, .65, .71])*(.34+.66*np.abs(normal@light))[:, None]
        projected = (tri@self.basis-self.center)*self.scale + np.array([0, .045])
        img = rasterize(projected, tri@self.view, colors, 0 if wire else len(faces), size)
        ax.imshow(img, extent=(-.79, .79, -.70, .79), origin='upper')
        ax.set(xlim=(-.79, .79), ylim=(-.70, .79), aspect='equal')
        ax.axis('off')


def static_figures(root, h, original, metrics):
    cam = Camera(h.rests)
    fig, axes = plt.subplots(1, 5, figsize=(19, 5), facecolor=BG)
    fig.subplots_adjust(left=.015, right=.985, bottom=.16, top=.73, wspace=.02)
    fig.text(.025, .93, 'RUFFLED CAPE / ACTUAL SIMPLIFICATION', color=INK, fontsize=24, weight='bold')
    fig.text(.025, .855, 'Curved neckline + radial pleats + small ripples + scalloped hem', color=MUTED, fontsize=13)
    for ax, k in zip(axes, reversed(range(5))):
        cam.draw(ax, h.rests[k], h.faces[k], wire=k<3)
        ax.set_title(f'L{k}'+('  ORIGINAL' if k==4 else '')+
                     f'\n{len(h.rests[k]):,} vertices / {len(h.faces[k]):,} triangles', color=INK, fontsize=12)
    fig.text(.025, .06, 'Original L4 -> native edge collapses -> L0. Correspondence uses successive self-parameterization.', color=MUTED, fontsize=12)
    fig.savefig(root/'hierarchy.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=BG)
    k = 1; p = h.to_finest[k]
    for ax, x, title in zip(axes, (h.rests[4], p.U@h.rests[k], p.position(h.rests[k])),
                           ('ORIGINAL FINE', 'LINEAR FROM L1', 'PSQ FROM L1')):
        cam.draw(ax, x, h.faces[4], size=900)
        ax.set_title(title, color=INK, fontsize=15, weight='bold')
    fig.suptitle('Which rest-shape details survive coarse-to-fine reconstruction?', fontsize=19, color=INK, y=.94)
    fig.text(.03, .06, 'Same original topology in every panel. PSQ transports stored normal and tangential residuals.', color=MUTED, fontsize=12)
    fig.subplots_adjust(left=.02, right=.98, bottom=.13, top=.83, wspace=.02)
    fig.savefig(root/'rest_comparison.png', dpi=140); plt.close(fig)

    # Follow one material-space ring, not a geometric nearest-point section.
    ids = np.flatnonzero(np.isclose(original['uv'][:, 1], .8125))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor=BG, sharex=True, sharey=True)
    for ax, k in zip(axes.flat, range(4)):
        p = h.to_finest[k]
        ax.plot(original['uv'][ids, 0], original['vertices'][ids, 2]*1000, c='#526078', lw=2.5, label='Original')
        ax.plot(original['uv'][ids, 0], (p.U@h.rests[k])[ids, 2]*1000, c='#cf824b', label='Linear', lw=1.7)
        ax.plot(original['uv'][ids, 0], p.position(h.rests[k])[ids, 2]*1000, '--', c='#118994', label='PSQ', lw=1.4)
        ax.set_title(f'L{k} -> L4 / {len(h.rests[k]):,} driver vertices')
        ax.set(xlabel='Material coordinate along the cape', ylabel='Height (mm)')
        ax.grid(alpha=.18)
    axes[0, 0].legend(ncol=3, fontsize=9)
    fig.suptitle('Actual detail loss and PSQ rest-shape recovery', fontsize=18, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(root/'detail_profiles.png', dpi=140); plt.close(fig)


def render(root, name, h, k, poses, outputs, errors):
    cam = Camera([poses, *outputs.values()])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=.025, right=.975, top=.845, bottom=.10, hspace=.19, wspace=.05)
    fig.text(.04, .96, 'PSQ / RUFFLED CAPE', color=INK, fontsize=25, weight='bold')
    fig.text(.04, .913, f'Actual L{k} driver: {len(h.rests[k]):,} vertices  ->  original L4: {len(h.rests[4]):,} vertices', color=MUTED, fontsize=13)
    status = fig.text(.96, .96, '', ha='right', color=INK, fontsize=12)
    labels = ('COARSE DRIVER', 'LINEAR  Ux', 'FIXED REST OFFSET', 'PSQ  /  ROTATING DETAIL')
    notes = ('Actual simplified mesh', 'Residual detail omitted', 'Residual stays in world coordinates', 'Normal + tangential residual transport')
    footer = fig.text(.04, .035, 'Prescribed motion | No physics solve or fine relaxation | Same camera and scale | No geometry smoothing', fontsize=10, color=MUTED)
    frames = []
    writer = imageio.get_writer(root/f'{name}.mp4', fps=20, codec='libx264', quality=8, macro_block_size=1)
    for i in range(COUNT):
        states = (poses[i], outputs['linear'][i], outputs['fixed'][i], outputs['psq'][i])
        for j, (ax, state) in enumerate(zip(axes.flat, states)):
            ax.clear()
            cam.draw(ax, state, h.faces[k] if j==0 else h.faces[4], wire=j==0)
            ax.set_title(labels[j], fontsize=14, weight='bold', color=INK, pad=7)
            note = notes[j]
            if errors and j:
                method = ('linear', 'fixed', 'psq')[j-1]
                note = f'Rigid-motion RMS error: {errors[method][i]:.3f} mm'
            ax.text(.5, -.035, note, transform=ax.transAxes, ha='center', color=MUTED, fontsize=10)
        status.set_text(('RIGID ROTATION' if name=='rigid_rotation' else 'BEND + TWIST + WING LIFT')+f'\nframe {i+1:02d} / {COUNT}')
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(frame)
        if i%2==0: frames.append(frame)
        if i==COUNT//2: imageio.imwrite(root/f'{name}_peak.png', frame)
    writer.close(); plt.close(fig)
    imageio.mimsave(root/f'{name}.gif', frames, duration=100, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/psq_ruffled_cape_demo')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); root = args.output.resolve()
    if args.prepare: prepare(root)
    shutil.copy2(__file__, root/'source_snapshot.py')
    h, original, manifest, weights = load(root)
    metrics = []
    for k, p in enumerate(h.to_finest):
        delta = p.U@h.rests[k]-h.rests[4]
        error = float(np.abs(p.position(h.rests[k])-h.rests[4]).max())
        assert error < 1e-12
        metrics.append(dict(level=k, vertices=len(h.rests[k]), triangles=len(h.faces[k]),
                            linear_rest_material_rms_mm=float(np.sqrt(np.einsum('v,vc,vc', weights, delta, delta))*1000),
                            psq_rest_max_error_m=error))
    static_figures(root, h, original, metrics)
    phase = np.sin(np.linspace(0, np.pi, COUNT))**2
    k = 1; audits = {}
    for name in ('bend_twist', 'rigid_rotation'):
        if name=='bend_twist':
            poses = np.array([deform(h.rests[k], s) for s in phase]); truth = None
        else:
            poses = np.array([h.rests[k]@rotation(s*np.pi/2).T for s in phase])
            truth = np.array([h.rests[4]@rotation(s*np.pi/2).T for s in phase])
        outputs, errors, audit = evaluate(h.to_finest[k], poses, truth, weights)
        audit['coarse_intersecting_sampled_frames'] = collision_frames(h.rests[k], h.faces[k], poses)
        audit['psq_intersecting_sampled_frames'] = collision_frames(h.rests[4], h.faces[4], outputs['psq'])
        if errors: audit['peak_rigid_area_weighted_rms_mm'] = {m:float(e[COUNT//2]) for m,e in errors.items()}
        audits[name] = audit
        print(name, audit, flush=True)
        np.savez_compressed(root/f'{name}_geometry.npz', coarse=poses, phase=phase, **outputs)
        render(root, name, h, k, poses, outputs, errors)
    peak = [h.to_finest[level].position(deform(h.rests[level], 1)) for level in range(5)]
    np.save(root/'deformed_all_levels.npy', peak)
    cam = Camera(peak)
    fig, axes = plt.subplots(1, 5, figsize=(19, 5.2), facecolor=BG)
    fig.subplots_adjust(left=.015, right=.985, bottom=.17, top=.75, wspace=.02)
    for level, (ax, x) in enumerate(zip(axes, peak)):
        cam.draw(ax, x, h.faces[4])
        ax.set_title(f'L{level} -> ORIGINAL L4\n{len(h.rests[level]):,} driver vertices', fontsize=12, color=INK)
    fig.text(.025, .92, 'SAME SPATIAL DEFORMATION / DIFFERENT DRIVER LEVELS', fontsize=22, color=INK, weight='bold')
    fig.text(.025, .055, 'L4: prescribed deformation directly on the fine mesh, not a physics reference. Coarse transport can introduce local seams.', fontsize=12, color=MUTED)
    fig.savefig(root/'deformed_all_levels.png', dpi=130); plt.close(fig)
    result = dict(scope='Prescribed geometric PSQ demonstration, not a physical simulation or safe-expansion solve',
                  original_geometry_exact=True, original_topology_exact=True, nearest_point_mapping=False,
                  levels=metrics, driver_level=k, frames_per_clip=COUNT, clips=audits,
                  collapse_counts=[x['collapses'] for x in manifest['expansion_histories']],
                  simplification_parameters=manifest['quadric_parameters'],
                  metric_note='Material RMS includes tangential correspondence drift; it is not pure wrinkle amplitude.',
                  intersection_note='Checks are discrete sampled-frame triangle tests, not continuous collision detection.',
                  limitations='PSQ transport alone does not guarantee smoothness, strain bounds or contact safety. No smoothing applied.',
                  source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    (Path(__file__), ROOT/'progressive/prolongation.py', ROOT/'progressive/prepare_intrinsic.py',
                     root/'original_fine_mesh.npz', root/'assets/manifest.json')})
    write_json(root/'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__':
    main()
