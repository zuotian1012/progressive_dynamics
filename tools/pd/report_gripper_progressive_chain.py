"""Verify first-fold runs and package the fixed Scene527 Newton assembly for comparison."""

import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re

import imageio_ffmpeg
import numpy as np
from scipy import sparse

from tools.pd.run_gripper_progressive_chain import (
    AUDIT_ACTION_FRAME, AUDIT_DIRECTORY, AUDIT_FRAME, CASES, EXTENDED_MESHES, FRAMES, MESHES, OUTPUT,
    SETTLE, SUBSTEPS, TRAJECTORY, case_path, mesh_directory,
)
from tools.pd.run_integrator_benchmark import DT, GRAVITY, lift, read_p, rms_curve


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, headers, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def distribution(values):
    return {"mean": float(np.mean(values)), "rms": float(np.sqrt(np.mean(np.square(values)))),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def newton_steps(log, directory):
    start = re.compile(r"^\[([^\]]+)\].*>>> Begin Frame: (\d+)")
    stop = re.compile(r"^\[([^\]]+)\].*<<< End Frame: (\d+)")
    converged = re.compile(r"Newton Iteration Converged with Iteration Count: (\d+)")
    capped = re.compile(r"Newton Iteration Exits with Max Iteration: (\d+)")
    rows = {}
    for line in log.splitlines():
        if match := start.search(line):
            began, frame = datetime.fromisoformat(match[1]), int(match[2])
            n, checks, hit_cap = None, 0, False
        elif ">> Newton Tolerance Checker" in line:
            checks += 1
        elif match := converged.search(line):
            n = int(match[1]) + 1
        elif match := capped.search(line):
            n, hit_cap = int(match[1]), True
        elif match := stop.search(line):
            if frame != int(match[2]) or n is None or n != checks:
                raise ValueError(f"{directory.name} frame {frame}: inconsistent Newton log ({n}, {checks})")
            # The build probe has frame=1. The real initial step with the same ID replaces it.
            rows[frame] = n, int(hit_cap), (datetime.fromisoformat(match[1]) - began).total_seconds()
    expected = SETTLE + SUBSTEPS * FRAMES
    if sorted(rows) != list(range(1, expected + 1)):
        raise ValueError(f"{directory.name}: incomplete IPC log ({len(rows)} of {expected})")
    write_csv(directory / "newton_substeps.csv",
              ("ipc_frame", "phase", "source_frame", "source_substep", "newton_iterations",
               "hit_limit", "solver_wall_s"),
              ((i, "settle" if i <= SETTLE else "action",
                (i - SETTLE - 1) // SUBSTEPS if i > SETTLE else "",
                (i - SETTLE - 1) % SUBSTEPS + 1 if i > SETTLE else "", *row)
               for i, row in sorted(rows.items())))
    action = np.array([rows[i] for i in range(SETTLE + 1, expected + 1)])
    return {"action_iterations": distribution(action[:, 0]),
            "action_iterations_total": int(action[:, 0].sum()),
            "action_limit_frames": [i for i, (_, cap, _) in rows.items() if i > SETTLE and cap],
            "settle_limit_frames": [i for i, (_, cap, _) in rows.items() if i <= SETTLE and cap],
            "action_solver_wall_s": float(action[:, 2].sum())}


def hands(log, directory, metrics, replay):
    with (directory / "fold_tcp_trajectory.csv").open() as file:
        rows = list(csv.DictReader(file))
    frames = replay["source_frames"]
    tcp = {hand: np.array([[float(row[f"tcp_{axis}_m"]) for axis in "xyz"]
                           for row in rows if row["hand"] == hand])
           for hand in ("left", "right")}
    separation = np.linalg.norm(tcp["left"] - tcp["right"], axis=1)[285:501]
    transforms = replay["ipc_link_transforms"]
    link_names = replay["ipc_link_names"].tolist()
    cloth = replay["cloth_pos"]
    summaries = {}
    for hand, link in (("left", "left_link16"), ("right", "right_link26")):
        events = metrics["contact_grasp_summary"][hand]["events"]
        if len(events) != 1:
            raise ValueError(f"{directory.name}: expected one close/open event for {hand}")
        event = dict(events[0])
        match = re.search(rf"contact_probe frame={event['close_frame']} hand={hand} .*?ids=(\[[^\]]*\])", log)
        event["patch_ids"] = json.loads(match[1]) if match else []
        if event["patch_ids"]:
            hold = np.flatnonzero((frames >= 270) & (frames < 548) & (frames < event["end_frame"]))
            if hold.size:
                rel = cloth[hold][:, event["patch_ids"]].mean(axis=1, dtype=np.float64) - tcp[hand][hold]
                rotation = transforms[hold, link_names.index(link), :3, :3]
                local = np.einsum("fi,fij->fj", rel, rotation)
                event["closed_material_patch_drift_mm"] = float(
                    1000 * np.linalg.norm(local - local[0], axis=1).max())
                event["closed_max_patch_tcp_mm"] = float(1000 * np.linalg.norm(rel, axis=1).max())
        summaries[hand] = event
    return summaries, {"frame_range": [285, 500],
                       "start_mm": float(separation[0] * 1000),
                       "end_mm": float(separation[-1] * 1000),
                       "change_pct": float(100 * (separation[-1] / separation[0] - 1)),
                       "max_increase_per_frame_mm": float(max(0, np.diff(separation).max()) * 1000)}


def run_stats(directory, require_video):
    log = (directory / "simulation.log").read_text()
    metrics = json.loads((directory / "fold.json").read_text())
    if not metrics["contact_grasp_test"] or metrics["virtual_grasp"] or metrics["kinematic_grasp_demo"]:
        raise ValueError(f"Expected pure friction in {directory}")
    if metrics["executed_frames"] != FRAMES:
        raise ValueError(f"Incomplete action in {directory}")
    with np.load(directory / "replay.npz", allow_pickle=False) as replay:
        np.testing.assert_array_equal(replay["source_frames"], np.arange(FRAMES))
        contact, separation = hands(log, directory, metrics, replay)
        replay_cloth = replay["cloth_pos"]
    x_span = 1000 * np.ptp(replay_cloth[[0, FRAMES - 1], :, 0].astype(np.float64), axis=1)
    geometry = {"initial_x_span_mm": float(x_span[0]), "final_x_span_mm": float(x_span[1]),
                "max_vertex_lift_above_table_mm": float(1000 * (replay_cloth[..., 2].max() - 0.8))}
    with np.load(directory / "state.npz", allow_pickle=False) as state:
        x, v = state["x"], state["v"]
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError(f"Nonfinite trajectory in {directory}")
        np.testing.assert_allclose(x[SETTLE + SUBSTEPS::SUBSTEPS], replay_cloth, atol=2e-6, rtol=0)
    video_path = directory / "fold.mp4"
    video = None
    if require_video:
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video_path))
        if frames != FRAMES or abs(seconds - FRAMES / 60) > 1 / 60:
            raise ValueError(f"Incomplete full-speed video {video_path}: {frames} frames, {seconds}s")
        video = {"file": str(video_path), "verified_frames": frames, "duration_s": seconds}
    return {"action_frames": FRAMES, "vertices": x.shape[1], "video": video,
            "action_wall_s": metrics["wall_seconds"], "hands": contact, "tcp_separation": separation,
            "cloth_geometry": geometry,
            **newton_steps(log, directory)}, x


def errors(output, states):
    results = {}
    for level in sorted(states):
        if level < 2 or level - 1 not in states:
            continue
        source, target = level - 1, level
        name = f"{8 << source}_to_{8 << target}"
        matrix = read_p(mesh_directory(target), "adjacent", source, target)
        coarse, fine = states[source], states[target]
        if len(coarse) != len(fine):
            raise ValueError(f"State histories of {name} have different time samples")
        delta = np.empty(len(fine))
        for start in range(0, len(fine), 32):
            batch = slice(start, start + 32)
            delta[batch] = 1000 * rms_curve(lift(matrix, coarse[batch]), fine[batch])
        after_action_frame = delta[SETTLE + SUBSTEPS::SUBSTEPS]
        write_csv(output / f"rms_{name}.csv", ("source_frame", "rms_mm"),
                  enumerate(after_action_frame))
        write_csv(output / f"rms_{name}_ipc.csv", ("ipc_frame", "rms_mm"), enumerate(delta))
        results[name] = {"action_frame_rms_mm": distribution(after_action_frame),
                         "action_substep_rms_mm": distribution(delta[SETTLE + 1:]),
                         "closed_hold_rms_mm": distribution(after_action_frame[270:548]),
                         "worst_action_source_frame": int(after_action_frame.argmax()),
                         "final_rms_mm": float(after_action_frame[-1])}
    return results


def canonical_matrix(matrix):
    matrix = sparse.csr_matrix(matrix)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Nonfinite sparse result")
    return matrix


def matrix_audit(path, matrix):
    matrix = canonical_matrix(matrix)
    sparse.save_npz(path, matrix)
    row, col = matrix.nonzero()
    pieces = [np.asarray(matrix.shape, dtype="<i8"), matrix.indptr.astype("<i8"),
              matrix.indices.astype("<i8"), matrix.data.astype("<f8")]
    return {"shape": list(matrix.shape), "nnz": matrix.nnz,
            "sha256": sha256(b"".join(piece.tobytes() for piece in pieces)).hexdigest(),
            "first_10": [[int(i), int(j), float(value)]
                         for i, j, value in zip(row[:10], col[:10], matrix.data[:10])]}


def array_audit(path, array):
    array = np.asarray(array, dtype="<f8")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite dense result in {path}")
    np.save(path, array)
    return {"shape": list(array.shape), "sha256": sha256(array.tobytes(order="C")).hexdigest(),
            "first_10": array.ravel(order="C")[:10].tolist()}


def audit(output):
    directory = output / AUDIT_DIRECTORY
    raw = directory / "raw"
    adjacent = read_p(MESHES, "adjacent", 1, 2)
    direct = read_p(MESHES, "direct", 2, 3)
    # Verify the operator used by the CUDA proxy JSON, independently of the CSR file.
    asset = json.loads((MESHES / "proxy_L2.json").read_text())
    columns = np.asarray(asset["columns"])
    weights = np.asarray(asset["weights"])
    scalar = sparse.coo_matrix((weights.ravel(), (np.repeat(np.arange(len(weights)), 3),
                                                   columns.ravel())),
                               shape=(direct.shape[0] // 3, direct.shape[1] // 3))
    if canonical_matrix(sparse.kron(scalar, sparse.eye(3)) - direct).nnz:
        raise ValueError("Mismatch between GPU proxy weights and the exported operator")
    with np.load(case_path(output, 1) / "state.npz", allow_pickle=False) as prior:
        px, pv = prior["x"][AUDIT_FRAME - 1], prior["v"][AUDIT_FRAME - 1]
    px_lifted, pv_lifted = adjacent @ px.ravel(), adjacent @ pv.ravel()
    planes = np.memmap(case_path(output, 2) / "predictor.f64", dtype="<f8", mode="r",
                       shape=(SETTLE + SUBSTEPS * FRAMES, 2, adjacent.shape[0]))
    np.testing.assert_array_equal(planes[AUDIT_FRAME - 1], [px_lifted, pv_lifted])
    center = (px_lifted + DT * pv_lifted).reshape(-1, 3)
    count = len(center)
    tilde = np.fromfile(raw / "x_tilde.f64", dtype="<f8").reshape(count, 3)
    force_tilde = center + DT**2 * np.array(GRAVITY)
    np.testing.assert_allclose(tilde, force_tilde, rtol=0, atol=2e-12)
    x = np.fromfile(raw / "x_newton.f64", dtype="<f8").reshape(count, 3)
    with np.load(case_path(output, 2) / "state.npz", allow_pickle=False) as current:
        np.testing.assert_allclose(x, current["x"][AUDIT_FRAME - 1], atol=2e-12, rtol=0)
    g_offset, row_offset, col_offset = np.loadtxt(raw / "block_offsets.txt", dtype=int)
    gi = np.fromfile(raw / "g_indices.i32", dtype="<i4") - g_offset
    rows = np.fromfile(raw / "H_rows.i32", dtype="<i4") - row_offset
    cols = np.fromfile(raw / "H_cols.i32", dtype="<i4") - col_offset
    if np.any(rows > cols) or any(np.any((ids < 0) | (ids >= count)) for ids in (gi, rows, cols)):
        raise ValueError("Proxy assembly block indices are outside cloth-local XYZ order")
    g = np.zeros((count, 3))
    np.add.at(g, gi, np.fromfile(raw / "g_blocks.f64", dtype="<f8").reshape(-1, 3))
    # CUDA writes one upper-triangular 3×3 block per pair, in Eigen column-major order.
    blocks = np.fromfile(raw / "H_blocks.f64", dtype="<f8").reshape(-1, 3, 3).transpose(0, 2, 1)
    channels = np.arange(3)
    r = np.broadcast_to(rows[:, None, None] * 3 + channels[None, :, None], blocks.shape)
    c = np.broadcast_to(cols[:, None, None] * 3 + channels[None, None, :], blocks.shape)
    upper = sparse.coo_matrix((blocks.ravel(), (r.ravel(), c.ravel())), shape=(3 * count, 3 * count))
    cross = rows != cols
    lower = sparse.coo_matrix((blocks[cross].ravel(), (c[cross].ravel(), r[cross].ravel())),
                              shape=upper.shape)
    hessian = canonical_matrix(upper + lower)
    asymmetry = hessian - hessian.T
    records = {
        "selection": {"layer": "32x32", "source_frame_zero_based": AUDIT_ACTION_FRAME,
                      "source_substep_one_based": 2, "ipc_frame_one_based": AUDIT_FRAME,
                      "newton_iteration_zero_based": 0, "dt_s": DT, "elastic_scale": DT**2},
        "predictor_gpu_cpu_linf_m": float(np.max(np.abs(tilde - force_tilde))),
        "hessian_asymmetry_linf_entry": float(np.max(np.abs(asymmetry.data), initial=0)),
        "files": {
            "P_adjacent.npz": matrix_audit(directory / "P_adjacent.npz", adjacent),
            "P_direct.npz": matrix_audit(directory / "P_direct.npz", direct),
            "x_hat.npy": array_audit(directory / "x_hat.npy", center),
            "x_tilde_gpu.npy": array_audit(directory / "x_tilde_gpu.npy", tilde),
            "Pt_g.npy": array_audit(directory / "Pt_g.npy", g),
            "Pt_H_P.npz": matrix_audit(directory / "Pt_H_P.npz", hessian),
            "x_newton_gpu.npy": array_audit(directory / "x_newton_gpu.npy", x),
            "x_coarse_previous.npy": array_audit(directory / "x_coarse_previous.npy", px),
            "v_coarse_previous.npy": array_audit(directory / "v_coarse_previous.npy", pv),
        },
    }
    write_json(directory / "checksums.json", records)
    return records


def update_extension_report(output, stats, adjacent_rms):
    report = output / "REPORT.md"
    if not report.is_file():
        return
    mesh = json.loads((EXTENDED_MESHES / "manifest.json").read_text())
    transfer = next(op for op in mesh["operators"] if op["kind"] == "adjacent"
                    and op["source_level"] == 3 and op["target_level"] == 4)
    left, right = (stats["hands"][hand] for hand in ("left", "right"))
    verdict = "双手均通过纯摩擦抓持判定。" if left["verdict"] == right["verdict"] == "PASS" else (
        f"左手判定为 {left['verdict']}，右手判定为 {right['verdict']}。")
    action_error = adjacent_rms["action_frame_rms_mm"]
    grip_drift = " / ".join(f"{value:.2f} mm" if value is not None else "—" for value in (
        hand.get("closed_material_patch_drift_mm") for hand in (left, right)))
    span = stats["cloth_geometry"]["final_x_span_mm"]
    title = "## 增补 · 128×128"
    lines = [
        title,
        "",
        "沿用同一 600 帧中部抓取轨迹，由已保存的 64×64 状态驱动 128×128 的 "
        "`paper_eq5`。128×128 使用本层原生弹性与 IPC 接触，历史粗层仍使用原实验的 "
        "64×64 弹性目标。这次运行追加一级，沿用原固定子步对账包。",
        "",
        f"新网格有 {stats['vertices']:,} 个顶点、32,768 张三角形，间距 5 mm。"
        f"64→128 延拓矩阵的静止重建最大误差为 {transfer['rest_error_inf_m']:.2e} m，"
        "全部行权重和严格等于 1。",
        "",
        f"动作完成 {stats['action_frames']}/600 帧。{verdict}闭合时跟踪点为 "
        f"{left['tracked_points']} / {right['tracked_points']}，"
        f"闭爪段最大漂移为 {grip_drift}（左 / 右）。末帧布料 X 向跨度为 {span:.1f} mm。",
        "",
        f"64→128 的全动作 RMS 为 **{action_error['rms']:.2f} mm**、"
        f"P95 为 {action_error['p95']:.2f} mm、峰值为 {action_error['max']:.2f} mm"
        f"（源帧 {adjacent_rms['worst_action_source_frame']}）、"
        f"末帧为 {adjacent_rms['final_rms_mm']:.2f} mm。",
        "",
        f"动作耗时 **{stats['action_wall_s']:.2f} s**，"
        f"Newton 共 {stats['action_iterations_total']:,} 次，"
        f"平均 {stats['action_iterations']['mean']:.2f} 次 / 子步，"
        f"有 {len(stats['action_limit_frames'])} 个子步达到 50 次上限。",
        "",
        "[128×128 回放](L4_128/fold.mp4) · [逐帧 RMS](rms_64_to_128.csv) · "
        "[逐子步 RMS](rms_64_to_128_ipc.csv) · [Newton 数据](L4_128/newton_substeps.csv) · "
        "[实际命令](L4_128/command.json) · [网格与延拓自检](../pd_square_cloth_128/manifest.json)",
        "",
        "```bash",
        "envs/genesis-ipc/bin/python -B -m tools.pd.run_gripper_progressive_chain simulate 4 --from-previous",
        "envs/genesis-ipc/bin/python -B -m tools.pd.run_gripper_progressive_chain render 4",
        "envs/genesis-ipc/bin/python -B -m tools.pd.report_gripper_progressive_chain",
        "```",
    ]
    original, marker, old_extension = report.read_text().partition(title)
    next_section = re.search(r"^## ", old_extension, re.M) if marker else None
    following = old_extension[next_section.start():] if next_section else ""
    report.write_text(original.rstrip() + "\n\n" + "\n".join(lines) + "\n"
                      + ("\n" + following if following else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--without-videos", action="store_true", help="Check simulation outputs before rendering")
    args = parser.parse_args()
    stats, states = {}, {}
    for level, name in CASES.items():
        if level == 4 and not (case_path(args.output, level) / "fold.json").is_file():
            continue
        stats[name], states[level] = run_stats(case_path(args.output, level), not args.without_videos)
    results = {
        "trajectory_sha256": sha256(TRAJECTORY.read_bytes()).hexdigest(),
        "levels": stats,
        "adjacent_rms": errors(args.output, states),
        "audit": audit(args.output)["selection"],
    }
    write_json(args.output / "comparison.json", results)
    if not args.without_videos and CASES[4] in stats:
        update_extension_report(args.output, stats[CASES[4]], results["adjacent_rms"]["64_to_128"])
    print("Validated trajectories, Newton logs, and fixed substep exports", flush=True)


if __name__ == "__main__":
    main()
