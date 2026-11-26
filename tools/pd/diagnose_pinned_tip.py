"""Ablate the physical source of the pinned-cloth Direct/PD free-edge difference."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from tools.pd.run_integrator_benchmark import DT, lift, predictor_input, read_p, rms_curve
from tools.pd.run_square_fold import ROOT, runtime_environment

SOURCE = ROOT / "outputs/pd_eq5_pinned_cloth_300"
MESHES = SOURCE / "hierarchy"
OUT = ROOT / "outputs/pd_pinned_tip_diagnosis"
BACKEND = ROOT / "outputs/pd_eq5_cuda/native"
STEPS = 120


def worker(name, level, *, proxy=False, predictor=None):
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", "-m", "tools.pd.run_integrator_benchmark", "--worker",
           "--solver-info", "--case", str(case), "--meshes", str(MESHES), "--backend", str(BACKEND),
           "--steps", str(STEPS), "--level", str(level), "--mode", "paper_eq5" if predictor else "bdf1"]
    if proxy:
        cmd += ["--proxy", str(MESHES / f"proxy_L{level}.json")]
    if predictor:
        cmd += ["--input", str(predictor)]
    (case / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    print(name, flush=True)
    with (case / "solver.log").open("w") as log:
        subprocess.run(cmd, env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    return case / "state.npz"


def run_controls():
    direct = worker("direct64_repeat", 4)
    worker("pd64_repeat", 4, predictor=SOURCE / "paper_eq5_L4/predictor.f64")
    start32 = worker("bdf32_proxy64", 3, proxy=True)
    p = read_p(MESHES, "adjacent", 3, 4)
    path = OUT / "from_bdf32.f64"
    predictor_input(start32, p, "paper_eq5", STEPS, path)
    worker("pd64_from_bdf32", 4, predictor=path)

    # Diagnostic reference: exact fine trajectory sampled at the nested 32-grid nodes.
    ids = np.arange(65**2).reshape(65, 65)[::2, ::2].ravel()
    with np.load(MESHES / "mesh_L3.npz") as c, np.load(MESHES / "mesh_L4.npz") as f:
        np.testing.assert_array_equal(c["rest"], f["rest"][ids])
    with np.load(direct) as history:
        sample = OUT / "reference_sampled32.npz"
        np.savez(sample, x=history["x"][:, ids], v=history["v"][:, ids], dt=DT)
    path = OUT / "from_reference_sampled32.f64"
    predictor_input(sample, p, "paper_eq5", STEPS, path)
    worker("pd64_from_sampled_reference", 4, predictor=path)
    run_start16()


def run_start16():
    previous = worker("bdf16_proxy64", 2, proxy=True)
    for level, name in ((3, "pd32_from_bdf16"), (4, "pd64_from_bdf16")):
        path = OUT / f"{name}_predictor.f64"
        predictor_input(previous, read_p(MESHES, "adjacent", level - 1, level), "paper_eq5", STEPS, path)
        previous = worker(name, level, proxy=level == 3, predictor=path)


def history(case, field="x"):
    with np.load(case / "state.npz", allow_pickle=False) as data:
        assert data["dt"].item() == DT
        values = data[field]
    assert len(values) == STEPS + 1 and np.isfinite(values).all()
    return values


def mm_rms(vectors):
    return float(1000 * np.sqrt(np.mean(np.sum(vectors**2, axis=-1))))


def metrics(x, ref, frame):
    delta = x - ref
    tip = delta[:, -65:]  # Rest-grid row opposite the two fixed corners
    return {
        "all_rms_mm": mm_rms(delta[1:]), "free_edge_rms_mm": mm_rms(tip[1:]),
        "snapshot_all_rms_mm": mm_rms(delta[frame]), "snapshot_edge_rms_mm": mm_rms(tip[frame]),
        "last_all_rms_mm": mm_rms(delta[-1]), "last_edge_rms_mm": mm_rms(tip[-1]),
    }


def preview_decomposition(coarse, ref):
    # Fit the best possible 4-grid shape at *each* Direct frame with the same
    # two pins, in the same Euclidean vertex metric used in the benchmark.
    weights = read_p(MESHES, "direct", 0, 4)[::3, ::3].tocsr()
    basis = weights[:, np.setdiff1d(np.arange(25), [0, 4])]
    residual = ref - lift(read_p(MESHES, "direct", 0, 4), coarse)
    rhs = residual.transpose(1, 0, 2).reshape(len(ref[0]), -1)
    correction = np.linalg.solve((basis.T @ basis).toarray(), basis.T @ rhs)
    representable = (basis @ correction).reshape(len(ref[0]), len(ref), 3).transpose(1, 0, 2)
    floor = mm_rms((residual - representable)[1:])
    excess = mm_rms(representable[1:])
    total = mm_rms(residual[1:])
    np.testing.assert_allclose(floor**2 + excess**2, total**2, rtol=1e-11)
    return dict(best_shape_floor_mm=floor, preview_to_best_fit_mm=excess, preview_total_mm=total)


def log_metrics(case):
    log = (case / "solver.log").read_text()
    frames = list(map(int, re.findall(r">>> Begin Frame: (\d+)", log)))
    end = list(map(int, re.findall(r"<<< End Frame: (\d+)", log)))
    iters = list(map(int, re.findall(r"Newton Iteration Converged with Iteration Count: (\d+)", log)))
    assert frames == end == list(range(1, STEPS + 1)) and len(iters) == STEPS
    contacts = {}
    for term in ("IPCSimplexNormalContact", "IPCVertexHalfPlaneNormalContact",
                 "IPCSimplexFrictionalContact", "IPCVertexHalfPlaneFrictionalContact"):
        counts = re.findall(term + r"> DyTopo Grad3 count: (\d+)", log)
        assert counts, (case, term)
        contacts[term] = max(map(int, counts))
    return dict(converged_steps=len(iters), max_newton_iterations=1 + max(iters),
                newton_limit_warnings=log.count("Newton Iteration Exits with Max Iteration:"),
                line_search_limit_warnings=log.count("Line Search Exits with Max Iteration:"),
                peak_contact_gradient_block_counts=contacts,
                warnings=[line for line in log.splitlines() if "[warning]" in line])


def plot(frame, ref, trajectories):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(11.4, 7.1), layout="constrained")
    grid = GridSpec(2, 2, figure=fig, height_ratios=(1.1, 0.8))
    ax = fig.add_subplot(grid[0, 0])
    pd = trajectories["pd64_repeat"]
    distances = np.linalg.norm(pd[frame] - ref[frame], axis=1).reshape(65, 65) * 1000
    pic = ax.imshow(distances, origin="lower", cmap="magma", aspect="auto")
    ax.scatter([0, 64], [0, 0], s=20, color="deepskyblue", label="fixed corners")
    ax.axhline(64, color="white", lw=1)
    ax.set(xlabel="Rest grid column", ylabel="Rest grid row (64 = free edge)",
           title=f"Material-point error at t = {frame*DT:.3f} s")
    ax.legend(loc="lower center", fontsize=8)
    fig.colorbar(pic, ax=ax, label="3D distance (mm)")
    ax = fig.add_subplot(grid[0, 1])
    for name, state, color in (("Direct IPC 64", ref, "#287b76"), ("PD 64, start 4", pd, "#be631b")):
        q = state[frame, -65:] * 1000
        ax.plot(q[:, 0], q[:, 2], lw=2, label=name, color=color)
        ax.scatter(q[[0, -1], 0], q[[0, -1], 2], color=color, s=22)
    ax.set(xlabel="World X (mm)", ylabel="World Z (mm)", title="Same free edge, X–Z projection",
           aspect="equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(grid[1, :])
    for key, label, color in (("pd64_repeat", "4 → 8 → 16 → 32 → 64", "#be631b"),
                              ("pd64_from_bdf16", "16 → 32 → 64 (16-grid BDF1 preview)", "#8a4b92"),
                              ("pd64_from_bdf32", "32 → 64 (32-grid BDF1 preview)", "#21609b"),
                              ("pd64_from_sampled_reference", "Direct states sampled on 32 → 64", "#287b76")):
        edge = trajectories[key][:, -65:]
        ax.plot(np.arange(len(edge))*DT, 1000*rms_curve(edge, ref[:, -65:]), label=label, color=color)
    ax.axvline(frame*DT, color="#777777", ls=":", lw=1)
    ax.set(xlabel="Physical time (s)", ylabel="Free-edge RMS (mm)", title="Error against Direct IPC 64")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(OUT / "free_edge_diagnosis.png", dpi=155)
    plt.close(fig)


def analyze(frame):
    ref = history(OUT / "direct64_repeat")
    old = history(SOURCE / "direct_fine")
    pd_old = history(SOURCE / "paper_eq5_L4")
    cases = ["direct64_repeat", "pd64_repeat", "bdf32_proxy64", "pd64_from_bdf32",
             "pd64_from_sampled_reference", "bdf16_proxy64", "pd32_from_bdf16", "pd64_from_bdf16"]
    logs = {name: log_metrics(OUT / name) for name in cases}
    trajectories = {name: history(OUT / name) for name in cases if name.startswith(("direct64", "pd64"))}
    settings = {}
    for name in ("direct_fine", "paper_eq5_L4"):
        settings[name] = json.loads((SOURCE / name / "settings.json").read_text())
    for key in ("material", "config", "cells", "initial_pins", "proxy", "contact_friction", "contact_resistance"):
        assert settings["direct_fine"][key] == settings["paper_eq5_L4"][key], key
    level_metrics = {}
    path = SOURCE / "proxy_L0"
    for level in range(5):
        current = SOURCE / (f"paper_eq5_L{level}" if level else "proxy_L0")
        if level:
            p = read_p(MESHES, "adjacent", level-1, level)
            pred = np.memmap(current / "predictor.f64", dtype="<f8", mode="r",
                             shape=(STEPS, 2, p.shape[0]))
            for plane, field in enumerate(("x", "v")):
                values = history(path, field)[:-1]
                np.testing.assert_array_equal(pred[:, plane], (p @ values.reshape(STEPS, -1).T).T)
        x = lift(read_p(MESHES, "direct", level, 4), history(current))
        level_metrics[current.name] = metrics(x, old, frame)
        path = current
    results = dict(
        snapshot_step=frame, time_s=frame * DT,
        original_levels_against_saved_direct64=level_metrics,
        controls_against_rerun_direct64={k: metrics(v, ref, frame) for k, v in trajectories.items()},
        repeat_deviation_mm=dict(direct=mm_rms((ref - old)[1:]),
                                 paper_eq5=mm_rms((trajectories["pd64_repeat"] - pd_old)[1:])),
        pinned_4grid_fit=preview_decomposition(history(SOURCE / "proxy_L0"), old),
        snapshot_xyz_mm_pd_minus_direct={str(i): ((pd_old-old)[frame, i]*1000).tolist()
                                         for i in (4160, 4192, 4224)},
        logs=logs,
    )
    for data in (ref, pd_old, old, *trajectories.values()):
        np.testing.assert_array_equal(data[:, [0, 64]], np.broadcast_to(old[0, [0, 64]], (STEPS+1, 2, 3)))
    # Could one uniform time shift align the tip shape in the screenshot?
    edge_error = 1000 * rms_curve(pd_old[:, -65:], old[:, -65:])
    edge_at_frame = pd_old[frame, -65:]
    shifted_error = 1000 * rms_curve(edge_at_frame, old[:, -65:])
    best_step = int(np.argmin(shifted_error))
    above = np.flatnonzero(edge_error > 1)
    results["free_edge_over_time"] = {
        "first_step_above_1mm": int(above[0]) if above.size else None,
        "peak_step": int(np.argmax(edge_error)), "peak_rms_mm": float(edge_error.max()),
        "snapshot_nearest_direct_step": best_step,
        "snapshot_nearest_direct_edge_rms_mm": float(shifted_error[best_step]),
    }
    # The fine solve uses the preceding frame's predictor row. The identical
    # gravity term cancels, and both fixed corners have identical inputs.
    planes = np.memmap(SOURCE / "paper_eq5_L4/predictor.f64", dtype="<f8", mode="r",
                       shape=(STEPS, 2, 65**2, 3))
    center_delta = (planes[:, 0] + DT * planes[:, 1]) - (old[:-1] + DT * history(SOURCE / "direct_fine", "v")[:-1])
    results["center_comparison"] = {
        "all_rms_mm": mm_rms(center_delta), "free_edge_rms_mm": mm_rms(center_delta[:, -65:]),
        "difference_in_corrections_rms_mm": mm_rms((pd_old - old)[1:] - center_delta),
        "difference_in_free_edge_corrections_rms_mm": mm_rms(((pd_old - old)[1:] - center_delta)[:, -65:]),
    }
    with (OUT / "curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "step", "time_s", "all_rms_mm", "free_edge_rms_mm", "edge_midpoint_mm",
                    "edge_xmin_corner_mm", "edge_xmax_corner_mm", "edge_mean_signed_dz_mm"])
        for name, x in trajectories.items():
            delta = x - ref
            for step, d in enumerate(delta):
                w.writerow([name, step, step*DT, mm_rms(d), mm_rms(d[-65:]),
                            *(1000*np.linalg.norm(d[[4192, 4160, 4224]], axis=1)), 1000*d[-65:, 2].mean()])
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    plot(frame, ref, trajectories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="Generate the GPU control trajectories first")
    parser.add_argument("--frame", type=int, choices=range(STEPS+1), metavar="0..120", default=98)
    args = parser.parse_args()
    if args.rerun:
        run_controls()
    analyze(args.frame)
