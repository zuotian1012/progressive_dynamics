"""Render saved physical states and report measured, separately timed runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image


def load_run(path):
    path = Path(path)
    meta = json.loads((path / "run.json").read_text())
    data = {"meta": meta, "x": np.load(path / "positions.npy", mmap_mode="r"),
            "f": np.load(path / "faces.npy"), "robot": np.load(path / "robot.npy", mmap_mode="r"),
            "colliders": []}
    for i in range(8):
        with np.load(path / f"collider_{i}.npz") as mesh:
            v, f = mesh["vertices"], mesh["faces"]
        # Reduce only the drawing mesh. Physics always uses the actual collision mesh.
        if len(f) > 500:
            import fast_simplification
            v, f = fast_simplification.simplify(v, f, target_count=500)
        data["colliders"].append((v, f))
    return data


def draw(ax, data, index, title="", wire=False):
    x, f = data["x"][index], data["f"]
    style = {"edgecolors": "#15476a", "linewidth": .10} if wire else {}
    cloth = Poly3DCollection(x[f], facecolors="#398ac0", alpha=1, shade=True, **style)
    ax.add_collection3d(cloth)
    for (v, f), t in zip(data["colliders"], data["robot"][index]):
        p = v @ t[:3, :3].T + t[:3, 3]
        ax.add_collection3d(Poly3DCollection(p[f], facecolors="#9aa4ad", shade=True))
    table = np.array([[.15,-.7,.8],[1.15,-.7,.8],[1.15,.7,.8],[.15,.7,.8]])
    # An outline avoids Matplotlib's whole-collection depth sorting overlaying cloth.
    edge = np.vstack((table, table[0]))
    ax.plot(edge[:,0], edge[:,1], edge[:,2], color="#a6b1bb", linewidth=1)
    ax.set(xlim=(.15,1.18), ylim=(-.55,.55), zlim=(.78,1.26), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_box_aspect((1.03,1.1,.48))
    if 'plot_bounds' in data:
        bounds = data['plot_bounds']
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
        ax.set_box_aspect([b[1]-b[0] for b in bounds])
    ax.view_init(elev=48, azim=-125)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)


def snapshots(path, output):
    data = load_run(path)
    end = data["meta"]["completed_steps"]
    fig = plt.figure(figsize=(12, 9))
    for i, k in enumerate(np.linspace(0, end, 4, dtype=int)):
        draw(fig.add_subplot(2, 2, i+1, projection="3d"), data, k, f"step {k}, t={k*data['meta']['dt']:.2f}s", True)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def compare(paths, output, animate=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = [load_run(p) for p in paths]
    end = min(d["meta"]["completed_steps"] for d in data)
    labels = [f"{d['meta']['method']} L{d['meta']['level']} | {len(d['f']):,} faces" for d in data]
    fig = plt.figure(figsize=(5*len(data), 5))
    for i, d in enumerate(data):
        draw(fig.add_subplot(1,len(data),i+1,projection="3d"), d, end, labels[i], True)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)
    if animate:
        images = []
        for k in np.unique(np.linspace(0, end, 100, dtype=int)):
            fig = plt.figure(figsize=(4.5*len(data),4.5), dpi=85)
            for i,d in enumerate(data):
                draw(fig.add_subplot(1,len(data),i+1,projection="3d"),d,k,labels[i])
            fig.suptitle(f"Simulation time: {k*data[0]['meta']['dt']:.2f} s | synchronized playback", fontsize=10)
            fig.tight_layout()
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
            plt.close(fig)
        duration = max(10, round(end*data[0]['meta']['dt']*1000/len(images)/10)*10)
        images[0].save(output / "comparison.gif", save_all=True, append_images=images[1:], duration=duration, loop=0)
    rows = [{k:d["meta"].get(k) for k in ("method","level","vertices","faces","completed","completed_steps","physics_seconds","physics_steps_per_second","simulated_seconds_per_wall_second","build_seconds")} for d in data]
    (output / "timing_summary.json").write_text(json.dumps(rows,indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    if args.snapshots:
        snapshots(args.runs[0], args.output)
    else:
        compare(args.runs, args.output, args.animate)
