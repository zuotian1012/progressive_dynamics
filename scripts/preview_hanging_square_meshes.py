"""Export and plot the proposed nested square-cloth hierarchy (no simulation)."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloth.mesh import grid, prolongation
from progressive.prolongation import nested_anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/hanging_square_mesh_preview")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meshes = [grid(n) for n in (4, 8, 16, 32)]
    rests = [np.column_stack((m.uv[:, 0] - .5, np.zeros(len(m.uv)), m.uv[:, 1] + .5)) for m in meshes]
    manifest = {"status": "mesh_preview_only", "size_m": [1, 1], "plane": "XZ", "gravity_direction": "-Z",
                "fixed_vertices": "only the upper left and upper right corners; upper edge interior is free",
                "prolongation": "nested barycentric; flat rest shape has zero PSQ residual", "levels": []}
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), facecolor="#f5f7fa")
    fig.suptitle("SQUARE CLOTH / NESTED MESH HIERARCHY", fontsize=20, fontweight="bold", y=.97, color="#202b3c")
    fig.text(.5, .927, "1 m x 1 m  |  two fixed upper corners  |  rest meshes, before simulation", ha="center", fontsize=12, color="#526078")
    for level, (mesh, rest, ax) in enumerate(zip(meshes, rests, axes.flat)):
        pins = np.array([mesh.segments * (mesh.segments + 1), len(rest) - 1])
        f = mesh.faces
        areas = np.linalg.norm(np.cross(rest[f[:, 1]]-rest[f[:, 0]], rest[f[:, 2]]-rest[f[:, 0]]), axis=1)/2
        assert np.all(areas > 0) and np.isclose(areas.sum(), 1)
        np.testing.assert_allclose(rest[pins], [[-.5, 0, 1.5], [.5, 0, 1.5]])
        np.savez(args.output / f"mesh_{level}.npz", vertices=rest, faces=f, uv=mesh.uv, pin_indices=pins)
        with (args.output / f"mesh_{level}.obj").open("w", encoding="utf-8") as out:
            out.write("# Square cloth rest mesh; units: metres\n")
            for x, y, z in rest:
                out.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for a, b, c in f + 1:
                out.write(f"f {a} {b} {c}\n")
        if level:
            U = prolongation(meshes[level-1], mesh)
            U.eliminate_zeros()
            np.testing.assert_allclose(U @ rests[level-1], rest, atol=1e-14)
            np.testing.assert_allclose(np.asarray(U.sum(axis=1)).ravel(), 1)
            save_npz(args.output / f"map_{level-1}_{level}.npz", U)
            np.save(args.output / f"anchors_{level-1}_{level}.npy", nested_anchors(U, meshes[level-1].faces))
        manifest["levels"].append({"level": level, "segments_per_side": mesh.segments, "vertices": len(rest), "triangles": len(f), "pin_indices": pins.tolist(), "area_m2": float(areas.sum())})
        xy = rest[:, [0, 2]]
        edges = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
        ax.set_facecolor("#f5f7fa")
        ax.fill([-.5, .5, .5, -.5], [.5, .5, 1.5, 1.5], color="#e4eff4")
        ax.add_collection(LineCollection(xy[edges], colors="#437a92", linewidths=.85 if level < 2 else .5))
        ax.plot([-.5, -.5, .5, .5, -.5], [.5, 1.5, 1.5, .5, .5], color="#264f66", lw=1.4)
        if level < 2:
            ax.scatter(*xy.T, s=8, color="#264f66", zorder=3)
        ax.scatter(*xy[pins].T, s=110, color="#e6693f", edgecolors="white", linewidths=1.7, zorder=5)
        for pin in pins:
            x, z = xy[pin]
            ax.plot([x, x], [z+.025, z+.12], color="#e6693f", lw=2)
        ax.set_title(f"L{level}   /   {mesh.segments} x {mesh.segments} cells\n{len(rest):,} vertices   |   {len(f):,} triangles", fontsize=13, color="#202b3c", pad=12, linespacing=1.6)
        ax.set(xlim=(-.63, .63), ylim=(.39, 1.65), aspect="equal")
        ax.axis("off")
    fig.text(.5, .035, "Orange = fixed vertices (2 per level)     |     All other vertices are free     |     Gravity: downward", ha="center", fontsize=11, color="#526078")
    fig.subplots_adjust(left=.035, right=.965, top=.85, bottom=.075, hspace=.32, wspace=.12)
    fig.savefig(args.output / "mesh_levels.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(args.output / "mesh_levels.svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
