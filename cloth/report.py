import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .simulation import ClothSystem, METHODS, predictor


COLORS = {"independent": "#777777", "semi": "#ce7b6e", "full": "#8da548", "velpro": "#438eac"}
LABELS = {"independent": "Independent (proxy energy)", "semi": "Semi-diagonal (2024)",
          "full": "Full-diagonal (2025)", "velpro": "VelPro (2025)"}


def weighted_rms(v, mass, free=None):
    if free is not None:
        v, mass = v[free], mass[free]
    return float(np.sqrt(np.sum(mass[:, None] * v*v) / mass.sum()))


def analyze(system, data, methods):
    h, steps = system.config["dt"], system.config["steps"]
    result = {}
    for method in methods:
        tracks = [data[f"{method}_{k}"] for k in range(len(system.levels))]
        levels = []
        for k, l in enumerate(system.levels):
            x = tracks[k]
            free = ~l.pins
            continuity, solve_error, ratio = [], [], []
            for t in range(1, steps):
                _, g = system.potential(k, x[t+1], (t+1)*h)
                mid_residual = 0.5 * (x[t+1] - 2*x[t] + x[t-1] + h*h*g/l.mass[:, None])
                xhat = predictor(method, k, t, tracks, system.maps, h, system.velocities)
                grad_solve = g.copy()
                w = system.config.get("consistency_weight", 0)
                if k and method != "independent" and w:
                    target = system.maps[k-1] @ tracks[k-1][t+1]
                    grad_solve += 2*w*l.mass[:, None]*(x[t+1]-target)
                solver_residual = x[t+1] - xhat + h*h*grad_solve/l.mass[:, None]
                e = weighted_rms(mid_residual, l.mass, free)
                es = weighted_rms(solver_residual, l.mass, free)
                continuity.append(e)
                solve_error.append(es)
                # Literal squared ratio of the printed formulas: direct baseline is 1/4.
                ratio.append(e*e/(es*es) if es > 1e-10 else None)
            velocity = np.diff(x, axis=0) / h
            # Singular stretches of the current-level physical mesh, independent of optimizer status.
            uv = l.mesh.uv[l.mesh.faces]
            inv_dm = np.linalg.inv(np.stack((uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]), axis=2))
            stretches = []
            for frame in x:
                tri = frame[l.mesh.faces]
                f = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2) @ inv_dm
                stretches.append(float(np.linalg.svd(f, compute_uv=False).max()))
            levels.append({"continuity_rms_m": continuity, "solve_residual_rms_m": solve_error,
                           "max_principal_stretch": stretches,
                           "max_speed_m_s": np.linalg.norm(velocity, axis=2).max(axis=1).tolist(),
                           "literal_squared_ratio": ratio,
                           "kinetic_energy_j": (0.5*np.sum(l.mass[None, :, None]*velocity**2, axis=(1, 2))).tolist()})
        p = system.levels[0].to_fine.toarray()
        # Use an SVD-backed least squares projection rather than explicitly inverting P^T P.
        projection = np.linalg.pinv(p)
        restricted = np.einsum("ij,tjk->tik", projection, tracks[-1])
        remainder = tracks[-1] - np.einsum("ij,tjk->tik", p, restricted)
        result[method] = {
            "levels": levels,
            "coarse_fine_consistency_rms_m": [weighted_rms(a-b, system.levels[0].mass) for a, b in zip(restricted, tracks[0])],
            "fine_detail_rms_m": [weighted_rms(v, system.levels[-1].mass) for v in remainder],
            "minimum_z": float(min(np.min(x[:, :, 2]) for x in tracks)),
            "max_pin_error_m": float(max((np.max(np.abs(x[:, l.pins] - l.initial[l.pins])) if l.pins.any() else 0)
                                         for x, l in zip(tracks, system.levels))),
        }
    return result


def plot_metrics(metrics, config, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    time = np.arange(config["steps"]+1) * config["dt"]
    for method, m in metrics.items():
        style = dict(color=COLORS[method], label=LABELS[method], lw=1.8)
        axes[0, 0].semilogy(time[1:-1], np.maximum(m["levels"][-1]["continuity_rms_m"], 1e-12), **style)
        axes[0, 1].plot(time, m["coarse_fine_consistency_rms_m"], **style)
        axes[1, 0].plot(time, m["fine_detail_rms_m"], **style)
        axes[1, 1].plot(time[1:], m["levels"][-1]["kinetic_energy_j"], **style)
    for ax, title, ylabel in zip(axes.flat,
            ["Temporal continuity residual (fine level)", "Coarse / fine consistency", "Fine detail outside coarse subspace", "Kinetic energy (fine level)"],
            ["Mass-normalized RMS (m)", "Mass-normalized RMS (m)", "Mass-normalized RMS (m)", "Energy (J)"]):
        ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Progressive cloth | controlled integration comparison", fontsize=15)
    fig.savefig(output / "metrics.png", dpi=150)
    plt.close(fig)


def render(data, methods, config, output, animate=True):
    k = len(config["segments"])-1
    positions = [data[f"{m}_{k}"] for m in methods]
    faces = data[f"faces_{k}"]
    all_x = np.concatenate([p.reshape(-1, 3) for p in positions])
    lo, hi = all_x.min(axis=0)-0.06, all_x.max(axis=0)+0.06
    center = (lo+hi)/2
    half = max(hi-lo)/2
    columns = min(2, len(methods))
    rows = (len(methods)+columns-1)//columns
    fig = plt.figure(figsize=(5.7*columns, 5.0*rows+0.6), facecolor="#fafafa")
    surfaces, captions = [], []
    for j, method in enumerate(methods):
        ax = fig.add_subplot(rows, columns, j+1, projection="3d")
        ax.set_facecolor("#fafafa")
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
               zlim=(center[2]-half, center[2]+half), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-65)
        ax.tick_params(labelsize=7)
        surf = Poly3DCollection(positions[j][0][faces], facecolors=COLORS[method], edgecolors="#34434a", linewidths=0.25, alpha=0.95)
        ax.add_collection3d(surf)
        floor = config.get("floor")
        if floor:
            z = floor["height"]
            plane = [[lo[0], lo[1], z], [hi[0], lo[1], z], [hi[0], hi[1], z], [lo[0], hi[1], z]]
            ax.add_collection3d(Poly3DCollection([plane], facecolors="#cbd5df", alpha=0.12, edgecolors="#a0aab5", linewidths=0.5))
        surfaces.append(surf)
        captions.append(ax.set_title(LABELS[method], fontsize=11, y=0.99, pad=5))
    def update(frame):
        for j, surf in enumerate(surfaces):
            surf.set_verts(positions[j][frame][faces])
            captions[j].set_text(f"{LABELS[methods[j]]}\nt = {frame*config['dt']:.2f} s")
        return surfaces + captions
    fig.subplots_adjust(left=0.03, right=0.95, bottom=0.06, top=0.83 if rows == 1 else 0.89, wspace=0.12, hspace=0.24)
    fig.suptitle("Progressive cloth: thin-shell dynamics", fontsize=14, y=0.98)
    fig.text(0.5, 0.93 if rows == 1 else 0.945, "Neo-Hookean membrane + dihedral bending", ha="center", fontsize=10)
    update(config["steps"])
    fig.savefig(output / "comparison.png", dpi=120)
    if animate:
        frames = np.unique(np.linspace(0, config["steps"], min(49, config["steps"]+1)).astype(int))
        anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
        anim.save(output / "comparison.gif", writer=PillowWriter(fps=20), dpi=75)
    plt.close(fig)


def generate(output, animate=True):
    output = Path(output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    logs = json.loads((output / "solver.json").read_text(encoding="utf-8"))
    system = ClothSystem(config)
    with np.load(output / "trajectories.npz") as archive:
        data = dict(archive)
    methods = [m for m in METHODS if f"{m}_0" in data]
    metrics = analyze(system, data, methods)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    summary = {"scene": output.name, "duration_s": config["steps"]*config["dt"],
               "vertices_per_level": [(s+1)**2 for s in config["segments"]], "methods": {}}
    for method in methods:
        records = [r for level in logs[method] for r in level]
        m = metrics[method]
        summary["methods"][method] = {
            "solver_warnings": sum(not r["success"] for r in records),
            "max_gradient_inf": max(r["gradient_inf"] for r in records),
            "all_level_step_seconds": sum(r["seconds"] for r in records),
            "minimum_floor_clearance_m": m["minimum_z"]-config["floor"]["height"] if config.get("floor") else None,
            "max_pin_error_m": m["max_pin_error_m"],
            "peak_continuity_rms_m": max(m["levels"][-1]["continuity_rms_m"], default=0),
            "peak_consistency_rms_m": max(m["coarse_fine_consistency_rms_m"]),
            "peak_fine_detail_rms_m": max(m["fine_detail_rms_m"]),
            "max_principal_stretch": max(max(l["max_principal_stretch"]) for l in m["levels"]),
            "peak_speed_m_s": max(max(l["max_speed_m_s"]) for l in m["levels"]),
        }
    summary["notes"] = ["Each method's timing includes the same cached coarse trajectory once; not a benchmark speedup.",
                        "RMS metrics divide by total included mass; compare with paper norms using the documented scaling.",
                        "Printed squared normalization gives independent baseline 1/4; near-zero denominator entries are null.",
                        "Independent baseline uses the shared finest-mesh proxy energy, not separate per-level materials."]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_metrics(metrics, config, output)
    render(data, methods, config, output, animate)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    generate(args.output, not args.no_gif)
