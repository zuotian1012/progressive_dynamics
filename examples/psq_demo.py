"""Exercise nonlinear PSQ on the included intrinsic fixture; no GPU/assets needed.

The animation is a prescribed deformation, not a physical simulation.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation
from progressive.prolongation import ShellProlongation


def mesh(path):
    vertices, faces = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if fields and fields[0] == 'v':
            vertices.append([float(v) for v in fields[1:4]])
        elif fields and fields[0] == 'f':
            faces.append([int(v.split('/')[0])-1 for v in fields[1:4]])
    return np.array(vertices), np.array(faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/psq_demo')
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    fixture = ROOT/'tests/fixtures/intrinsic_expansion'
    coarse, coarse_faces = mesh(fixture/'mesh_1.obj')
    fine, fine_faces = mesh(fixture/'mesh_0.obj')
    values = np.loadtxt(fixture/'map_1.txt', skiprows=1)
    U = coo_matrix((values[:, 2], (values[:, 0].astype(int), values[:, 1].astype(int))),
                   shape=(len(fine), len(coarse))).tocsr()
    P = ShellProlongation(coarse, fine, coarse_faces, U,
                         np.loadtxt(fixture/'anchors_1.txt', dtype=int))
    if P.zero_offset:
        raise ValueError('Demo must exercise nonzero nonlinear offsets')
    np.testing.assert_allclose(P.position(coarse), fine, atol=1e-12)
    rotation = Rotation.from_rotvec([.3, -.2, .4]).as_matrix()
    np.testing.assert_allclose(P.position(coarse@rotation.T+[.1, .2, .3]),
                               fine@rotation.T+[.1, .2, .3], atol=1e-12)
    positions, coarse_tracks, linear_tracks = [], [], []
    for t in np.linspace(0, 2*np.pi, 49):
        x = coarse.copy()
        x[:, 2] += .15*np.sin(t)*(x[:, 0]**2 + .3*x[:, 1])
        positions.append(P.position(x))
        coarse_tracks.append(x)
        linear_tracks.append(U@x)
    velocity = np.random.default_rng(7).normal(size=coarse.shape)
    x = coarse_tracks[8]
    numerical = (P.position(x+1e-6*velocity)-P.position(x-1e-6*velocity))/2e-6
    analytic = P.velocity(x, velocity)
    np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-7)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/'trajectories.npz', coarse=coarse_tracks, psq=positions,
             linear=linear_tracks, coarse_faces=coarse_faces, fine_faces=fine_faces)
    summary = dict(kind='prescribed deformation, not dynamics', coarse_vertices=len(coarse),
                   fine_vertices=len(fine), nonlinear_offset=True,
                   rest_max_error=float(np.abs(P.position(coarse)-fine).max()),
                   velocity_derivative_max_error=float(np.abs(analytic-numerical).max()),
                   max_nonlinear_offset=float(np.linalg.norm(np.array(positions)-linear_tracks, axis=2).max()))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    if not args.no_render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from PIL import Image
        frames = []
        bounds = np.concatenate([np.array(positions).reshape(-1, 3), np.array(coarse_tracks).reshape(-1, 3)])
        lo, hi = bounds.min(0), bounds.max(0)
        center, radius = (lo+hi)/2, max(hi-lo)*.6
        for c, p, linear in zip(coarse_tracks, positions, linear_tracks):
            fig = plt.figure(figsize=(10, 3.5), dpi=100)
            for k, (vertices, faces, title) in enumerate([(c, coarse_faces, 'Coarse input'),
                    (linear, fine_faces, 'Linear Ux'), (p, fine_faces, 'Nonlinear PSQ')]):
                ax = fig.add_subplot(1, 3, k+1, projection='3d')
                ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor='#80b9d8',
                                                    edgecolor='#30465b', linewidth=.5))
                ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                       zlim=(center[2]-radius, center[2]+radius), title=title)
                ax.set_box_aspect((1, 1, 1)); ax.view_init(25, -65); ax.set_axis_off()
            fig.suptitle('PSQ operator demonstration: prescribed deformation (not dynamics)')
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
            plt.close(fig)
        frames[12].save(args.output/'comparison.png')
        frames[0].save(args.output/'comparison.gif', save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
