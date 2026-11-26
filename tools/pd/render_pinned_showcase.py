"""Render saved pinned-cloth trajectories for the meeting, without advancing physics."""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image

from tools.pd.run_square_fold import ROOT


SOURCE = ROOT / "outputs/pd_eq5_pinned_cloth_300"
OUTPUT = ROOT / "outputs/pd_showcase"
FPS = 15
PANELS = {
    "pinned_paper_eq5_16_32_64": (
        ("paper_eq5_L2", 2, "16 × 16", "paper_eq5", "#247eb1"),
        ("paper_eq5_L3", 3, "32 × 32", "paper_eq5", "#247eb1"),
        ("paper_eq5_L4", 4, "64 × 64", "paper_eq5", "#247eb1"),
    ),
    "pinned_integrator_comparison": (
        ("direct_fine", 4, "Direct IPC", "64 × 64 · BDF1", "#267e7b"),
        ("paper_eq5_L4", 4, "paper_eq5", "64 × 64 · coarse-state prediction", "#247eb1"),
    ),
}


def load_case(source, name, level):
    with np.load(source / name / "state.npz", allow_pickle=False) as state:
        x, dt = state["x"], state["dt"].item()
    with np.load(source / "hierarchy" / f"mesh_L{level}.npz", allow_pickle=False) as mesh:
        faces = mesh["triangles"]
    cells = int(round(np.sqrt(x.shape[1]))) - 1
    if not np.isfinite(x).all() or (cells + 1)**2 != x.shape[1]:
        raise ValueError(f"Invalid saved mesh in {name}")
    np.testing.assert_array_equal(x[:, [0, cells]], np.broadcast_to(x[0, [0, cells]], x[:, [0, cells]].shape))
    return x, faces, cells, dt


def boundary(cells):
    ids = np.arange((cells + 1)**2).reshape(cells + 1, cells + 1)
    return np.concatenate((ids[0], ids[1:, -1], ids[-1, -2::-1], ids[-2::-1, 0]))


def render(source, output, name, panels):
    cases = [load_case(source, item[0], item[1]) for item in panels]
    reference, _, reference_cells, dt = load_case(source, "proxy_L0", 0)
    for x, _, _, step in cases:
        if len(x) != len(reference) or step != dt:
            raise ValueError("Panel trajectories have incompatible time samples")
    forward = np.array([0.9, -1.5, 0.8])
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 0, 1], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    camera = np.column_stack((right, up, forward))
    projections = [x @ camera for x, _, _, _ in cases]
    ref_projected = reference @ camera
    joined = np.concatenate([v.reshape(-1, 3) for v in projections] + [ref_projected.reshape(-1, 3)])
    minimum, maximum = joined[:, :2].min(0), joined[:, :2].max(0)
    center = (maximum + minimum) / 2
    pair = len(panels) == 2
    extent = max(maximum - minimum) * (1.045 if pair else 1.16)

    fig = plt.figure(figsize=(18, 10 if pair else 7.2), dpi=100, facecolor="#f5f7fa")
    polygons, outlines, pins = [], [], []
    for i, (_, _, title, subtitle, _) in enumerate(panels):
        ax = fig.add_axes([0.012 + i / len(panels), 0.14 if pair else 0.16, 1 / len(panels) - 0.024, 0.67 if pair else 0.66])
        ax.set(xlim=(center[0] - extent / 2, center[0] + extent / 2),
               ylim=(center[1] - extent / 2, center[1] + extent / 2), aspect="equal")
        ax.axis("off")
        fig.text((i + 0.5) / len(panels), 0.94, title, ha="center", fontsize=18, weight="bold", color="#24364a")
        fig.text((i + 0.5) / len(panels), 0.897, subtitle, ha="center", fontsize=12, color="#52667c")
        polygons.append(ax.add_collection(PolyCollection([], linewidths=0, antialiased=False)))
        outlines.append(ax.add_collection(LineCollection([], colors="#233348", linewidths=0.8,
                                                         linestyles="--", zorder=3)))
        pins.append(ax.scatter([], [], s=24, color="#ba3654", zorder=4))
    clock = fig.text(0.5, 0.095, "", ha="center", fontsize=15, color="#24364a")
    overlay = name == "pinned_integrator_comparison"
    fig.text(0.5, 0.045, "Recorded states · 120 simulation steps/s · 1/8-speed playback · red points fixed",
             ha="center", fontsize=12, color="#52667c")
    if overlay:
        fig.text(0.5, 0.83, "Dashed outline in every panel: saved 4 × 4 proxy preview at the same time",
                 ha="center", fontsize=11, color="#52667c")
    light = np.array([0.2, -0.3, 1.0])
    light /= np.linalg.norm(light)
    video = output / f"{name}.mp4"
    writer = imageio_ffmpeg.write_frames(str(video), fig.canvas.get_width_height(), fps=FPS, codec="libx264",
                                        pix_fmt_out="yuv420p", macro_block_size=1,
                                        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"])
    writer.send(None)
    try:
        for frame in range(len(reference)):
            for i, ((x, faces, cells, _), projected, panel) in enumerate(zip(cases, projections, panels)):
                triangles = x[frame][faces]
                normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
                shade = 0.42 + 0.58 * np.abs(normal @ light)
                colors = np.minimum(np.asarray(to_rgb(panel[-1])) * shade[:, None] + 0.08, 1)
                view = projected[frame][faces]
                order = np.argsort(view[:, :, 2].mean(1))
                polygons[i].set_verts(view[order, :, :2])
                polygons[i].set_facecolors(colors[order])
                contour = ref_projected[frame, boundary(reference_cells), :2] if overlay else (
                    projected[frame, boundary(cells), :2])
                outlines[i].set_segments([contour])
                pins[i].set_offsets(projected[frame, [0, cells], :2])
            clock.set_text(f"Physical time {frame * dt:.3f} s   ·   saved IPC state {frame}/{len(reference) - 1}")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.send(rgb)
            if frame in (0, 60, len(reference) - 1):
                Image.fromarray(rgb).save(output / f"{name}_{frame:03d}.png")
    finally:
        writer.close()
        plt.close(fig)
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    if frames != len(reference) or abs(seconds - frames / FPS) > 0.02:
        raise ValueError(f"Video verification failed for {video}")
    return dict(file=video.name, saved_states=frames, fps=FPS, playback_seconds=seconds,
                physical_seconds=(len(reference) - 1) * dt, view="shared orthographic camera and scale",
                source_cases=[item[0] for item in panels])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((args.source / "comparison.json").read_text())
    records = []
    for name, panels in PANELS.items():
        records.append(render(args.source, args.output, name, panels))
        print(f"Rendered and verified {name}", flush=True)
    fig = plt.figure(figsize=(8.3, 5.2))
    ax = fig.add_axes([0.12, 0.16, 0.85, 0.65])
    x = np.arange(4)
    for offset, mode, label, color in ((-0.18, "vel_inc_gravity", "Velocity increments + gravity", "#c17727"),
                                        (0.18, "paper_eq5", "paper_eq5", "#247eb1")):
        values = [evidence["errors"][f"{mode}_L{level}_vs_L{level-1}"]["rms_mm"] for level in range(1, 5)]
        ax.bar(x + offset, values, width=0.32, label=label, color=color)
        for i, value in enumerate(values):
            ax.text(i + offset, value * 1.14, f"{value:.2f}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=["4 → 8", "8 → 16", "16 → 32", "32 → 64"],
           ylabel="Adjacent-level trajectory RMS (mm, log scale)", xlabel="Grid cells per side",
           yscale="log", ylim=(0.2, 3000))
    fig.suptitle("Pinned cloth · saved 1.0 s experiment", y=0.98, fontsize=14)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.54, 0.93), ncols=2, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(args.output / "pinned_adjacent_rms.png", dpi=180)
    plt.close(fig)
    (args.output / "replay_manifest.json").write_text(json.dumps(
        dict(source=str(args.source.resolve()), physics_advanced=False, videos=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
