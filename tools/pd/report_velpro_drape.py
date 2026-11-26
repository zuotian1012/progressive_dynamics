"""Measure bulk deformation, enrichment and velocity changes in Dynamic Drape."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg
import numpy as np
from numba import njit
from scipy.sparse.linalg import splu

from tools.pd.run_integrator_benchmark import lift, read_mesh, read_p
from tools.pd.run_velpro_drape import DT


def fit(matrix, fine):
    """PD++ Eq. (13) least-squares projection onto the coarse subspace."""
    solver = splu((matrix.T @ matrix).tocsc())
    rhs = matrix.T @ fine.reshape(len(fine), -1).T
    return solver.solve(rhs).T.reshape(len(fine), -1, 3)


def rest_weights(rest, faces):
    a, b, c = rest[faces].transpose(1, 0, 2)
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    weights = np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=len(rest))
    return weights / weights.sum()


def spatial_rms(values, weights):
    return np.sqrt(np.einsum("tvi,tvi,v->t", values, values, weights))


def stats(curves):
    return {name: {"rms": float(np.sqrt(np.mean(data**2))), "p95": float(np.percentile(data, 95)),
                   "max": float(np.max(data)), "last": float(data[-1])}
            for name, data in curves.items()}


def load_case(directory):
    with np.load(directory / "state.npz", allow_pickle=False) as data:
        if data["dt"] != DT:
            raise ValueError(f"{directory}: unexpected timestep")
        x, v, times = (data[key] for key in ("x", "v", "step_wall_s"))
    if (x.shape != v.shape or len(x) != len(times) + 1
            or not (np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(times).all())):
        raise ValueError(f"{directory}: incomplete or nonfinite state")
    np.testing.assert_allclose(v[1:], np.diff(x, axis=0) / DT, rtol=1e-12, atol=1e-12)
    log = (directory / "solver.log").read_text()
    return x, v, dict(solver_wall_s=float(times.sum()),
                      newton_cap_steps=log.count("Newton Iteration Exits with Max Iteration:"),
                      line_search_cap_steps=log.count("Line Search Exits with Max Iteration:"))


def resolutions(directory):
    manifest = json.loads((directory / "hierarchy/manifest.json").read_text())
    return tuple(level["cells_per_side"] for level in manifest["levels"])


def analyze(directory):
    meshes = directory / "hierarchy"
    cells = resolutions(directory)
    weights = [rest_weights(*read_mesh(meshes, level)) for level in range(len(cells))]
    last = len(cells) - 1
    preview, _, base_timing = load_case(directory / "preview_L0")
    direct, direct_v, direct_timing = load_case(directory / "direct_fine")
    p_bulk = read_p(meshes, "direct", 0, last)
    direct_bulk = fit(p_bulk, direct)
    all_curves, modes = {}, {}

    def compare_finest(x, v):
        bulk = fit(p_bulk, x)
        return {
            "total_vs_direct_mm": 1000 * spatial_rms(x - direct, weights[last]),
            "bulk_vs_direct_mm": 1000 * spatial_rms(bulk - direct_bulk, weights[0]),
            "bulk_vs_preview_mm": 1000 * spatial_rms(bulk - preview, weights[0]),
            "detail_rms_mm": 1000 * spatial_rms(x - lift(p_bulk, bulk), weights[last]),
            "center_z_m": np.einsum("tv,v->t", x[:, :, 2], weights[last]),
            "accel_rms_m_s2": spatial_rms(np.diff(v, axis=0), weights[last]) / DT,
            "d2_velocity_rms_m_s": spatial_rms(np.diff(v, n=2, axis=0), weights[last]),
        }

    all_curves["direct_fine"] = compare_finest(direct, direct_v)
    for mode in ("paper_eq5", "velpro"):
        prior_x, prior_v = preview, None
        modes[mode] = []
        for level in range(1, last + 1):
            x, v, timing = load_case(directory / f"{mode}_L{level}")
            p = read_p(meshes, "adjacent", level - 1, level)
            if len(x) != len(direct) or x.shape[1] != len(weights[level]):
                raise ValueError(f"{mode} L{level}: inconsistent time axis or mesh")
            coarse_fit = fit(p, x)
            curves = {
                "adjacent_total_mm": 1000 * spatial_rms(x - lift(p, prior_x), weights[level]),
                "adjacent_bulk_mm": 1000 * spatial_rms(coarse_fit - prior_x, weights[level - 1]),
                "adjacent_detail_mm": 1000 * spatial_rms(x - lift(p, coarse_fit), weights[level]),
                "d2_velocity_rms_m_s": spatial_rms(np.diff(v, n=2, axis=0), weights[level]),
            }
            for name, values in curves.items():
                all_curves[f"{mode}_L{level}_{name}"] = {name: values}
            modes[mode].append({"level": level, "cells": cells[level], **timing, **stats(curves)})
            prior_x, prior_v = x, v
        all_curves[mode] = compare_finest(prior_x, prior_v)

    result = {"resolutions": cells, "steps": len(direct) - 1, "dt_s": DT,
              "timings": {"preview": base_timing, "direct": direct_timing},
              "levels": modes, "finest": {name: stats(all_curves[name])
                                         for name in ("direct_fine", "paper_eq5", "velpro")}}
    (directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (directory / "curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("case", "metric", "time_s", "value"))
        for case, curves in all_curves.items():
            for name, values in curves.items():
                offset = len(direct) - len(values)
                writer.writerows((case, name, (i + offset) * DT, value) for i, value in enumerate(values))
    return all_curves, result


def plot(directory, curves):
    names = ("direct_fine", "paper_eq5", "velpro")
    labels = ("Direct IPC", "PD 2024 (semi-diagonal)", "PD++ (VelPro)")
    metrics = ("bulk_vs_direct_mm", "detail_rms_mm", "center_z_m", "d2_velocity_rms_m_s")
    titles = ("Bulk error vs Direct IPC (mm)", "Fine detail above the common coarse subspace (mm)",
              "Cloth center of mass height (m)", "Vertex RMS second velocity difference (m/s)")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    for ax, metric, title in zip(axes.ravel(), metrics, titles, strict=True):
        for name, label in zip(names, labels, strict=True):
            y = curves[name][metric]
            offset = 2 if metric == "d2_velocity_rms_m_s" else 0
            ax.plot((np.arange(len(y)) + offset) * DT, y, label=label, lw=1.35)
        ax.set(title=title, xlabel="Time (s)")
        ax.grid(alpha=.22)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(directory / "drape_metrics.png", dpi=160)
    plt.close(fig)


@njit
def rasterize(triangles, colors, width, height):
    """Visibility with an orthographic per-pixel depth test."""
    depth = np.full((height, width), -np.inf)
    rgb = np.full((height, width, 3), 251, dtype=np.uint8)
    for triangle, color in zip(triangles, colors):
        a, b, c = triangle
        u, v = b - a, c - a
        det = u[0] * v[1] - v[0] * u[1]
        if abs(det) < 1e-12:
            continue
        for y in range(max(0, int(np.floor(np.min(triangle[:, 1])))),
                       min(height, int(np.ceil(np.max(triangle[:, 1]))))):
            for x in range(max(0, int(np.floor(np.min(triangle[:, 0])))),
                           min(width, int(np.ceil(np.max(triangle[:, 0]))))):
                dx, dy = x + .5 - a[0], y + .5 - a[1]
                s, t = (v[1] * dx - v[0] * dy) / det, (-u[1] * dx + u[0] * dy) / det
                if s >= 0 and t >= 0 and s + t <= 1:
                    z = a[2] + s * u[2] + t * v[2]
                    if z > depth[y, x]:
                        depth[y, x], rgb[y, x] = z, color
    return rgb


def render(directory):
    last = len(resolutions(directory)) - 1
    _, faces = read_mesh(directory / "hierarchy", last)
    with np.load(directory / "gear.npz", allow_pickle=False) as gear:
        gear_triangles = gear["vertices"][gear["faces"]]
    names = ("preview_L0", f"paper_eq5_L{last}", f"velpro_L{last}", "direct_fine")
    labels = ("Shared coarse preview", "PD 2024 · semi-diagonal", "PD++ · VelPro", "Direct IPC")
    states = []
    for name in names:
        with np.load(directory / name / "state.npz", allow_pickle=False) as data:
            states.append(data["x"])
    _, coarse_faces = read_mesh(directory / "hierarchy", 0)
    topologies = (coarse_faces, faces, faces, faces)
    render_comparison(gear_triangles, states, topologies, labels,
                      ("#899bbe", "#6d59a6", "#087e7c", "#31668e"),
                      directory / "drape_comparison.mp4")


def render_comparison(gear_triangles, states, topologies, labels, panel_colors, output, *, rows=1,
                      stride=2, camera_angles=(23., -57.), focus=None, keyframes=(), figsize=None,
                      stills_only=False):
    cols = len(labels) // rows
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (3.5 * cols, 3.6 * rows + .4), squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88 if rows == 1 else .92, wspace=0.02, hspace=.13)
    fig.patch.set_facecolor("#edf1f5")
    panels = []
    # Fixed orthographic camera and framing across all branches.
    elevation, azimuth = np.deg2rad(camera_angles)
    towards_camera = np.array([np.cos(elevation) * np.cos(azimuth),
                               np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    down = -np.cross(towards_camera, right)
    camera = np.column_stack((right, down, towards_camera))
    size = 644 if rows == 1 else 440
    if focus is None:
        projected_gear = (gear_triangles @ camera).reshape(-1, 3)
        minima, maxima = [projected_gear.min(axis=0)], [projected_gear.max(axis=0)]
        for x in states:
            samples = x[list(keyframes)] if stills_only else x[::stride]
            projected = (samples @ camera).reshape(-1, 3)
            minima.append(projected.min(axis=0))
            maxima.append(projected.max(axis=0))
        low, high = np.min(minima, axis=0), np.max(maxima, axis=0)
        center = (low + high)[:2] / 2
        span = np.max((high - low)[:2]) / .88
    else:
        origin, span = focus
        center = (np.asarray(origin) @ camera)[:2]
    scale = size / span
    pan = size / 2 - scale * center
    for ax, label, topology, color in zip(axes, labels, topologies, panel_colors, strict=True):
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_axis_off()
        artist = ax.imshow(np.full((size, size, 3), 251, dtype=np.uint8), interpolation="hanning")
        panels.append((artist, topology, matplotlib.colors.to_rgb(color)))
    label = fig.suptitle("", fontsize=14, y=.98)
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    sun = np.array([.35, -.42, .84])
    gear_color = np.array(matplotlib.colors.to_rgb("#bc9160"))
    indices = list(keyframes) if stills_only else list(range(0, len(states[0]), stride))
    if not stills_only and indices[-1] != len(states[0]) - 1:
        indices.append(len(states[0]) - 1)
    with nullcontext() if stills_only else writer.saving(fig, str(output), dpi=120):
        for t in indices:
            title = f"Dynamic Drape   ·   t = {t * DT:.2f} s   ·   step {t}/{len(states[0])-1}"
            label.set_text(title if stills_only else f"{title}   ·   {25 * stride * DT:g}× speed")
            for x, (artist, topology, color) in zip(states, panels, strict=True):
                triangles = np.concatenate((x[t][topology], gear_triangles))
                normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                length = np.linalg.norm(normal, axis=1)
                light = .38 + .61 * np.abs(np.einsum("ij,j->i", normal, sun)) / np.maximum(length, 1e-15)
                colors = np.tile(color, (len(triangles), 1))
                colors[len(topology):] = gear_color
                screen = triangles @ camera
                screen[:, :, :2] = screen[:, :, :2] * scale + pan
                artist.set_data(rasterize(screen, (light[:, None] * colors * 255).astype(np.uint8),
                                          size, size))
            if not stills_only:
                writer.grab_frame()
            if t in keyframes:
                fig.savefig(output.with_name(f"{output.stem}_frame{t:03d}.png"), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    curves, results = analyze(args.directory)
    plot(args.directory, curves)
    if args.video:
        render(args.directory)
    for name, metrics in results["finest"].items():
        print(name, {key: round(metrics[key]["rms"], 4) for key in (
            "total_vs_direct_mm", "bulk_vs_direct_mm", "detail_rms_mm", "d2_velocity_rms_m_s")})


if __name__ == "__main__":
    main()
