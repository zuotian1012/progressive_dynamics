"""Analyze pose consistency, tooth coverage and solve costs for the gear sweep."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from numba import njit

from tools.pd.audit_drape_contact import EXTERIOR_INSET_M, gear_clip_geometry, intersecting_frames
from tools.pd.report_velpro_drape import (
    fit, load_case, plt, render_comparison, rest_weights, spatial_rms, stats,
)
from tools.pd.run_gear_drape_50k_sweep import CHAIN, GROUPS
from tools.pd.run_integrator_benchmark import lift, read_mesh, read_p
from tools.pd.run_velpro_drape import DT


@njit
def side_coverage(samples, weights, outline, bottom, top, band):
    """Rest-area fraction below the top and within the specified band of a side wall."""
    result = np.zeros(len(samples))
    for t, cloud in enumerate(samples):
        for i, (x, y, z) in enumerate(cloud):
            if not bottom <= z < top or not .18**2 < x*x + y*y < .27**2:
                continue
            for j, (ax, ay) in enumerate(outline):
                bx, by = outline[(j + 1) % len(outline)]
                dx, dy = bx - ax, by - ay
                along = max(0., min(1., ((x - ax)*dx + (y - ay)*dy) / (dx*dx + dy*dy)))
                if (x - ax - along*dx)**2 + (y - ay - along*dy)**2 <= band*band:
                    result[t] += weights[i]
                    break
    return result


def newton_numbers(path, steps):
    frame = None
    last_translation = None
    counts, capped, capped_residuals = {}, [], {}
    begin = re.compile(r'>>> Begin Frame: (\d+)')
    converged = re.compile(r'Newton Iteration Converged with Iteration Count: (\d+)')
    cap = re.compile(r'Newton Iteration Exits with Max Iteration: (\d+) \(Frame=(\d+)\)')
    translate = re.compile(r'MaxTranslationChecker> Residual/AbsTol: ([0-9.eE+\-]+)/([0-9.eE+\-]+)')
    with path.open() as log:
        for line in log:
            if match := begin.search(line):
                frame = int(match[1])
                last_translation = None
            elif match := translate.search(line):
                last_translation = [float(match[1]), float(match[2])]
            elif match := converged.search(line):
                if frame in counts:
                    raise ValueError(f"Duplicate frame summary in {path}: {frame}")
                counts[frame] = int(match[1]) + 1
            elif match := cap.search(line):
                if frame in counts or frame != int(match[2]):
                    raise ValueError(f"Ambiguous frame summary in {path}: {line}")
                counts[frame] = int(match[1])
                capped.append(frame)
                if last_translation:
                    capped_residuals[frame] = dict(residual_m=last_translation[0], tolerance_m=last_translation[1])
    if sorted(counts) != list(range(1, steps + 1)):
        raise ValueError(f"Missing Newton frame summaries in {path}")
    return [counts[index] for index in range(1, steps + 1)], capped, capped_residuals


def series(directory, n, steps):
    positions, _, timing = load_case(directory)
    if positions.shape != (steps + 1, n, 3):
        raise ValueError(f"Unexpected trajectory shape in {directory}: {positions.shape}")
    counts, capped, capped_residuals = newton_numbers(directory / "solver.log", steps)
    assert len(capped) == timing["newton_cap_steps"]
    wall = json.loads((directory / "wall_seconds.json").read_text())
    return positions, timing | dict(
        wall_s=wall, mean_step_ms=1000 * timing["solver_wall_s"] / steps,
        mean_newton=float(np.mean(counts)), max_newton=max(counts),
        newton_counts=counts, capped_frames=capped, capped_translation=capped_residuals,
    )


def load_gear(root):
    with np.load(root / "A/gear.npz", allow_pickle=False) as gear:
        verts, faces = gear["vertices"], gear["faces"]
    bottom, top = verts[:, 2].min(), verts[:, 2].max()
    # gear_mesh stores the outline first, in increasing polar angle.
    outline = verts[:(len(verts) - 2) // 2, :2]
    return verts[faces], outline, bottom, top


def verify_saved_runs(root, config):
    """Recheck the executed scene and predictor inputs from preserved artifacts."""
    cases = [(name, case, level, previous, proxy) for name in GROUPS for case, level, previous, proxy in (
        ("direct_coarse", 0, None, False),
        (CHAIN[0], 0, None, True),
        (CHAIN[1], 1, CHAIN[0], True),
        (CHAIN[2], 2, CHAIN[1], False),
    )]
    cases.append(("A", "direct_fine", 2, None, False))
    gate = json.loads(Path(config["gate"]).read_text())
    expected_material = gate["runtime"]["effective_material"]
    reference = json.loads((root / "A/direct_fine/settings.json").read_text())
    fine_cells = GROUPS["A"][-1]
    energy = (f"PD proxy compute_energy: membrane_elements={2 * fine_cells**2} "
              f"bending_hinges={3 * fine_cells**2 - 2 * fine_cells}")
    route = "proxies=1 native_energy_slots=0 native_gradient_blocks=0 native_hessian_blocks=0"
    fine_rest, fine_faces = read_mesh(root / "A/hierarchy", 2)
    with np.load(root / "A/gear.npz", allow_pickle=False) as reference_gear:
        for name in GROUPS:
            with np.load(root / name / "gear.npz", allow_pickle=False) as gear:
                for key in ("vertices", "faces"):
                    np.testing.assert_array_equal(gear[key], reference_gear[key])
            rest, faces = read_mesh(root / name / "hierarchy", 2)
            np.testing.assert_array_equal(rest, fine_rest)
            np.testing.assert_array_equal(faces, fine_faces)
    ipc_blocks = re.compile(r'IPCSimplex(Normal|Frictional)Contact> DyTopo Grad3 count: (\d+)')
    native_shell = {"DiscreteShellBending", "StrainLimitingBaraffWitkinShell2D"}
    checked = {}
    for name, case, level, previous, proxy in cases:
        group = root / name
        run = group / case
        settings = json.loads((run / "settings.json").read_text())
        for key in ("config", "material", "contact_friction", "contact_resistance", "initial_pins"):
            if settings[key] != reference[key]:
                raise ValueError(f"Scene setting {key} differs from Direct Fine in {run}")
        if settings["effective_material"] != expected_material:
            raise ValueError(f"Effective material differs from Part 1 in {run}")
        if settings["cells"] != GROUPS[name][level] or settings["mode"] != ("velpro" if previous else "bdf1"):
            raise ValueError(f"Wrong grid or integrator in {run}")
        meshes = group / "hierarchy"
        asset = str(meshes / f"proxy_L{level}.json") if proxy else None
        if settings["proxy"] != asset:
            raise ValueError(f"Wrong proxy asset in {run}")
        systems = json.loads((run / "systems.json").read_text())["sim_systems"]
        systems = {item["name"].rsplit("::", 1)[-1] for item in systems}
        expected_native_shell = set() if proxy else native_shell
        if systems & native_shell != expected_native_shell or "FiniteElementBDF1Kinetic" not in systems:
            raise ValueError(f"Wrong FEM elasticity or inertia registration in {run}")
        log = (run / "solver.log").read_text()
        if (energy in log) != proxy or (route in log) != proxy:
            raise ValueError(f"Wrong proxy route or integration target in {run}")
        maxima = {"Normal": 0, "Frictional": 0}
        for effect, count in ipc_blocks.findall(log):
            maxima[effect] = max(maxima[effect], int(count))
        if config["steps"] >= 20 and not all(maxima.values()):
            raise ValueError(f"Missing normal or frictional IPC assembly in {run}: {maxima}")
        predictor = run / "predictor.f64"
        if settings["predictor_input"] != (str(predictor) if previous else None):
            raise ValueError(f"Wrong predictor asset in {run}")
        if previous:
            with np.load(group / previous / "state.npz", allow_pickle=False) as state:
                velocity = state["v"][:-1]
            transfer = read_p(meshes, "adjacent", level - 1, level)
            actual = np.memmap(predictor, dtype="<f8", mode="r")
            expected = lift(transfer, velocity)
            np.testing.assert_array_equal(actual, expected.ravel())
        checked[f"{name}/{case}"] = dict(proxy=proxy, native_shell=sorted(systems & native_shell),
                                         inertia="FiniteElementBDF1Kinetic", peak_ipc_normal_gradient_blocks=maxima["Normal"],
                                         peak_ipc_friction_gradient_blocks=maxima["Frictional"],
                                         velocity_prolongation_bit_exact=True if previous else None)
    return checked


def analyze(root, steps):
    meshes = root / "A/hierarchy"
    fine_rest, fine_faces = read_mesh(meshes, 2)
    fine_w = rest_weights(fine_rest, fine_faces)
    direct, timing = series(root / "A/direct_fine", len(fine_w), steps)
    shared_basis = read_p(meshes, "direct", 1, 2)
    _, outline, bottom, top = load_gear(root)
    wall_band = .010
    clip_geometry = gear_clip_geometry(root / "A/gear.npz")
    gear_intersections = {"Direct Fine": intersecting_frames(direct, fine_faces, clip_geometry)}
    curves = {
        "direct_fine_side_area_pct": side_coverage(direct, fine_w, outline, bottom, top, wall_band) * 100,
        "direct_fine_detail_mm": 1000 * spatial_rms(direct - lift(shared_basis, fit(shared_basis, direct)), fine_w),
    }
    timings = {"Direct Fine": timing}
    for name in GROUPS:
        directory = root / name
        meshes = directory / "hierarchy"
        topology = [read_mesh(meshes, level) for level in range(3)]
        weights = [rest_weights(*mesh) for mesh in topology]
        native, timings[f"{name}/direct_coarse"] = series(directory / "direct_coarse", len(weights[0]), steps)
        gear_intersections[f"{name}/direct_coarse"] = intersecting_frames(native, topology[0][1], clip_geometry)
        states = []
        for level, case in enumerate(CHAIN):
            positions, timings[f"{name}/{case}"] = series(directory / case, len(weights[level]), steps)
            gear_intersections[f"{name}/{case}"] = intersecting_frames(positions, topology[level][1], clip_geometry)
            states.append(positions)
        coarse, middle, fine = states
        p = read_p(meshes, "direct", 0, 2)
        coarse_fine = lift(p, coarse)
        native_fine = lift(p, native)
        p_adj0 = read_p(meshes, "adjacent", 0, 1)
        p_adj1 = read_p(meshes, "adjacent", 1, 2)
        metrics = {
            "coarse_to_middle_mm": 1000 * spatial_rms(lift(p_adj0, coarse) - middle, weights[1]),
            "middle_to_fine_mm": 1000 * spatial_rms(lift(p_adj1, middle) - fine, fine_w),
            "coarse_to_fine_mm": 1000 * spatial_rms(coarse_fine - fine, fine_w),
            "fine_to_direct_mm": 1000 * spatial_rms(fine - direct, fine_w),
            "native_to_direct_mm": 1000 * spatial_rms(native_fine - direct, fine_w),
            "coarse_to_direct_mm": 1000 * spatial_rms(coarse_fine - direct, fine_w),
            "native_to_proxy_mm": 1000 * spatial_rms(native - coarse, weights[0]),
            "fine_detail_mm": 1000 * spatial_rms(fine - lift(shared_basis, fit(shared_basis, fine)), fine_w),
            "native_side_area_pct": side_coverage(native_fine, fine_w, outline, bottom, top, wall_band) * 100,
            "coarse_side_area_pct": side_coverage(coarse_fine, fine_w, outline, bottom, top, wall_band) * 100,
            "fine_side_area_pct": side_coverage(fine, fine_w, outline, bottom, top, wall_band) * 100,
        }
        curves.update({f"{name}_{key}": value for key, value in metrics.items()})
        print(f"Analyzed group {name}", flush=True)
    np.savetxt(root / "curves.csv", np.column_stack([np.arange(steps + 1)*DT, *curves.values()]),
               delimiter=",", header=",".join(["time_s", *curves]), comments="")
    # Common contact-window mask gives every group exactly the same time samples.
    # It does not discard the influence of capped steps on subsequent states.
    first_contact = int(np.ceil(np.sqrt(2 * (.40 - top) / 9.81) / DT))
    uncapped = np.arange(steps + 1) >= first_contact
    for row in timings.values():
        uncapped[row["capped_frames"]] = False
    result = {
        "timings": timings,
        "summary": stats({key: value[1:] for key, value in curves.items()}),
        "tooth_snapshot": {
            "time_s": 46 * DT,
            "side_area_pct": {key: float(value[46]) for key, value in curves.items()
                              if key.endswith("_side_area_pct")},
        } if steps >= 46 else None,
        "gear_intersection_audit": {
            "sampled_frames_per_run": steps + 1,
            "gear_exterior_inset_m": EXTERIOR_INSET_M,
            "by_run": gear_intersections,
        },
        "uncapped_contact_frames": np.flatnonzero(uncapped).tolist(),
        "uncapped_contact_rms_mm": {
            name: float(np.sqrt(np.mean(curves[f"{name}_fine_to_direct_mm"][uncapped]**2)))
            for name in GROUPS
        } if np.any(uncapped) else {},
    }
    return result, curves


def plot(root, steps, curves):
    t = np.arange(steps + 1) * DT
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    for name in GROUPS:
        for ax, metric in zip(axes.flat, ("coarse_to_fine_mm", "fine_to_direct_mm", "coarse_to_middle_mm",
                                          "middle_to_fine_mm", "fine_detail_mm", "fine_side_area_pct"), strict=True):
            ax.plot(t, curves[f"{name}_{metric}"], label=name)
    axes[2, 0].plot(t, curves["direct_fine_detail_mm"], "k:", label="Direct Fine")
    axes[2, 1].plot(t, curves["direct_fine_side_area_pct"], "k:", label="Direct Fine")
    labels = ("Coarse to final", "Final to Direct Fine", "Coarse to middle", "Middle to final",
              "Detail above common 56² basis", "Gear side coverage")
    for index, (ax, label) in enumerate(zip(axes.flat, labels, strict=True)):
        ax.set(title=label, xlabel="Time (s)", ylabel="Rest-area %" if index == 5 else "RMS (mm)")
        ax.grid(alpha=.2)
        ax.legend(ncols=3)
    fig.savefig(root / "consistency.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(15, 5.8), sharex=True, sharey="row", layout="constrained")
    branches = (("native", "Direct Coarse", "#9d5d47"), ("coarse", "PD Coarse", "#396ba3"))
    for column, name in enumerate(GROUPS):
        top, bottom = axes[:, column]
        for branch, label, color in branches:
            top.plot(t, curves[f"{name}_{branch}_to_direct_mm"], label=label, color=color)
            bottom.plot(t, curves[f"{name}_{branch}_side_area_pct"], label=label, color=color)
        bottom.plot(t, curves["direct_fine_side_area_pct"], color="#30343d", ls=":", label="Direct Fine")
        top.set_title(f"{name} · {(GROUPS[name][0] + 1)**2:,} coarse vertices")
        bottom.set_xlabel("Time (s)")
        top.grid(alpha=.2)
        bottom.grid(alpha=.2)
    axes[0, 0].set_ylabel("RMS vs. Direct Fine (mm)")
    axes[1, 0].set_ylabel("Gear side coverage (%)")
    axes[1, 0].legend(fontsize=8)
    fig.savefig(root / "coarse_proxy_comparison.png", dpi=140)
    plt.close(fig)


def render(root, steps):
    gear, _, _, _ = load_gear(root)
    direct, _ = series(root / "A/direct_fine", (GROUPS["A"][-1] + 1)**2, steps)
    states, faces, labels = [], [], []
    native_states, native_faces, zoom_labels = [], [], []
    for name, (coarse, _, fine) in GROUPS.items():
        directory = root / name
        files = ((CHAIN[0], 0), (CHAIN[2], 2), ("direct_coarse", 0))
        x, f = [], []
        for case, level in files:
            with np.load(directory / case / "state.npz", allow_pickle=False) as data:
                x.append(data["x"])
            f.append(read_mesh(directory / "hierarchy", level)[1])
        states.extend(x[:2])
        faces.extend(f[:2])
        labels.extend((f"{name} · PD Coarse ({(coarse+1)**2:,} vertices)", f"{name} · PD Fine ({(fine+1)**2:,} vertices)"))
        native_states.extend((x[2], x[0], x[1], direct))
        native_faces.extend((f[2], f[0], f[1], f[1]))
        zoom_labels.extend((f"{name} · Direct Coarse", f"{name} · PD Coarse", f"{name} · PD Fine", "Direct Fine · shared"))
    frames = tuple(sorted({frame for frame in (20, 46, 80, 120, steps) if frame <= steps}))
    order = [*range(0, len(states), 2), *range(1, len(states), 2)]
    ordered = [[items[index] for index in order] for items in (
        states, faces, labels, ("#788dad", "#097c7b") * len(GROUPS))]
    render_comparison(gear, *ordered, root / "overview_wide.mp4", rows=2,
                      keyframes=frames, figsize=(16, 9))
    print("Rendered landscape overview", flush=True)
    for start, suffix in ((0, "ab"), (8, "cd")):
        group = slice(start, start + 8)
        render_comparison(gear, native_states[group], native_faces[group], zoom_labels[group],
                          ("#987468", "#788dad", "#097c7b", "#326c99") * 2,
                          root / f"gear_zoom_{suffix}_wide.mp4", rows=2, keyframes=frames,
                          camera_angles=(16., -42.), focus=((.075, -.13, .22), .41), figsize=(16, 9))
        print(f"Rendered landscape gear zoom {suffix}", flush=True)


def report(root, config, result, videos):
    steps = config["steps"]
    summary, timings = result["summary"], result["timings"]
    asset_links = "[逐帧曲线](curves.csv) · [Newton 与误差原始数据](metrics.json) · [运行配置](setup.json)"
    wide = all((root / name).is_file() for name in
               ("overview_wide.mp4", "gear_zoom_ab_wide.mp4", "gear_zoom_cd_wide.mp4"))
    if videos and wide:
        asset_links += (" · [横屏全景](overview_wide.mp4) · [A/B 横屏特写](gear_zoom_ab_wide.mp4)"
                        " · [C/D 横屏特写](gear_zoom_cd_wide.mp4)")
    elif videos:
        asset_links += " · [全景并排](overview.mp4) · [尖齿特写](gear_zoom.mp4)"
    comparison = root.parent / "pd_gear_drape_50k_vertices_direct_compare/REPORT.md"
    if comparison.is_file():
        asset_links += " · [同一 50,625 顶点网格的原生 IPC 对照](../pd_gear_drape_50k_vertices_direct_compare/REPORT.md)"
    lines = ["# Dynamic Drape · 50K 顶点分辨率扫描", "", asset_links, "",
             "[代理弹性验收](../pd_gear_proxy_audit/REPORT.md)", "", "## 场景与测量口径", "",
             f"1 m 方布落在相同的固定 12 齿障碍物上。全部运行共 {steps} 步，每步 {DT:.2f} s。"
             "摩擦系数 0.5，IPC 激活距离 1.5 mm，单侧厚度 0.07 mm。粗预览和独立 IPC 使用 BDF1，渐进中间层与最细层使用 VelPro。"
             "所有代理都用 224×224 细网格计算弹性。各级在本层处理 IPC，并分别验证了实际材料属性一致。 "
             "[PD 2024 Fig. 5 与 §5.1](https://eriszhang.github.io/assets/pdf/progressive-dynamics-main.pdf#page=10)",
             "",
             "方网格标签以格子数计。A/B 和 C/D 分别共享中间网格，B 为嵌套细分对照，其余组的最粗层相对细层不嵌套。"
             "独立 Direct Fine 使用同一个最细网格。墙钟包括求解进程启动、场景初始化和写盘。单步时间取 advance 与 sync，均开启相同日志。",
             "", "| 组 | 方网格分辨率 | 各层顶点 | 各层三角面 |", "|---|---|---|---|"]
    p_errors, p_row_errors = [], []
    for name, cells in GROUPS.items():
        lines.append(f"| {name} | " + " → ".join(f"{c}×{c}" for c in cells) + " | "
                     + " → ".join(f"{(c+1)**2:,}" for c in cells) + " | "
                     + " → ".join(f"{2*c*c:,}" for c in cells) + " |")
        manifest = json.loads((root / name / "hierarchy/manifest.json").read_text())
        p_errors.extend(row["rest_error_inf_m"] for row in manifest["operators"])
        p_row_errors.extend(max(abs(row["row_sum_min"] - 1), abs(row["row_sum_max"] - 1))
                            for row in manifest["operators"])
    lines += ["", f"各层采用规则三角网格。直达与相邻延拓都从静止平面的三角形重心坐标构造，"
              f"每个顶点展开 XYZ 三个自由度。全部算子的静止构型误差上限为 {max(p_errors):.3g} m，"
              f"行和最大偏差为 {max(p_row_errors):.3g}。"]
    if verified := result.get("verified_runs"):
        num_predictors = sum(row["velocity_prolongation_bit_exact"] is True for row in verified.values())
        lines += ["", f"逐项复核了 {len(verified)} 个算例的实际场景配置与 CUDA 日志，"
                  f"并用上一层已保存的速度重算全部 {num_predictors} 个 VelPro 预测文件，逐项完全一致。"]
    lines += ["", "## 求解成本", "", "| 组 / 方法 | 顶点 | 单步平均 (ms) | 求解合计 (s) | 墙钟 (s) | Newton 平均 / 最大 | 触顶步 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    levels = {"direct_coarse": 0, **{case: level for level, case in enumerate(CHAIN)}}
    for name, row in timings.items():
        vertices = (GROUPS[name[0]][levels[name.split('/')[1]]] + 1)**2 if "/" in name else (GROUPS["A"][-1]+1)**2
        path = name if "/" in name else "A/direct_fine"
        lines.append(f"| [{name}]({path}/state.npz) | {vertices:,} | {row['mean_step_ms']:.1f} | {row['solver_wall_s']:.2f} | "
                     f"{row['wall_s']:.2f} | {row['mean_newton']:.2f} / {row['max_newton']} | {row['newton_cap_steps']} |")
    lines += ["", "| 组 | 渐进链求解合计 (s) | 渐进链进程墙钟合计 (s) | 粗预览相对独立细解的单步加速 |",
              "|---|---:|---:|---:|"]
    for name in GROUPS:
        runs = [timings[f"{name}/{case}"] for case in CHAIN]
        lines.append(f"| {name} | {sum(row['solver_wall_s'] for row in runs):.2f} | "
                     f"{sum(row['wall_s'] for row in runs):.2f} | "
                     f"{timings['Direct Fine']['mean_step_ms'] / runs[0]['mean_step_ms']:.2f}× |")
    limited = {name: row for name, row in timings.items() if row["newton_cap_steps"]}
    if limited:
        lines += ["", "触顶步的最后一轮 MaxTranslationChecker 记录如下。"
                  "有触顶步的轨迹以求解器接受的状态统计，独立细网格也按同一标准记录。",
                  "", "| 算例 | 有记录的触顶步 | 退出残差中位数 (mm) | 退出残差最大值 (mm) |",
                  "|---|---:|---:|---:|"]
        for name, timings_row in limited.items():
            residuals = [1000 * row["residual_m"] for row in timings_row["capped_translation"].values()]
            if residuals:
                lines.append(f"| {name} | {len(residuals)} / {timings_row['newton_cap_steps']} | "
                             f"{np.median(residuals):.4g} | {max(residuals):.4g} |")
            else:
                lines.append(f"| {name} | 0 / {timings_row['newton_cap_steps']} | — | — |")
    lines += ["", "## 轨迹与包裹程度", "",
              "表中位置差以同一静止材料坐标比对，先按静止面积加权取逐帧 RMS，再对时间取 RMS，排除初始帧。"
              "侧壁覆盖取最后一帧位于齿轮顶面以下、且距侧壁横截面 10 mm 内的布料面积占比，是局部包裹的几何指标。"
              "粗层先延拓到同一最细采样网格。细节 RMS 从细轨迹中扣除统一 56×56 基底的最小二乘投影。",
              "", "| 组 | 粗→中 (mm) | 中→细 (mm) | 粗→细 (mm) | 细→独立 (mm) | 原生粗→独立 (mm) | 代理粗→独立 (mm) | 细节 (mm) | 侧壁覆盖 (%) 原生粗 / 代理粗 / 细 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in GROUPS:
        values = [summary[f"{name}_{metric}_mm"]["rms"] for metric in (
            "coarse_to_middle", "middle_to_fine", "coarse_to_fine", "fine_to_direct", "native_to_direct", "coarse_to_direct", "fine_detail")]
        wrap = [summary[f"{name}_{m}_side_area_pct"]["last"] for m in ("native", "coarse", "fine")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + " | " + " / ".join(f"{v:.2f}" for v in wrap) + " |")
    direct_wrap = summary["direct_fine_side_area_pct"]["last"]
    direct_detail = summary["direct_fine_detail_mm"]["rms"]
    lines += ["", f"独立细网格参考的末帧侧壁覆盖为 {direct_wrap:.2f}%，细节 RMS 为 {direct_detail:.2f} mm。"]
    closest = min(GROUPS, key=lambda name: summary[f"{name}_fine_to_direct_mm"]["rms"])
    helped = [name for name in GROUPS if summary[f"{name}_coarse_to_direct_mm"]["rms"] < summary[f"{name}_native_to_direct_mm"]["rms"]]
    wrapped = [name for name in GROUPS
               if summary[f"{name}_coarse_side_area_pct"]["last"] > summary[f"{name}_native_side_area_pct"]["last"]]
    closest_error = summary[f"{closest}_fine_to_direct_mm"]["rms"]
    fine_errors = [summary[f"{group}_fine_to_direct_mm"]["rms"] for group in GROUPS]
    detail_delta = summary["D_fine_detail_mm"]["rms"] - summary["A_fine_detail_mm"]["rms"]
    trend = "随起始分辨率单调下降" if np.all(np.diff(fine_errors) <= 0) else "随起始分辨率未呈单调下降"
    lines += [f"最细层相对独立细解的 RMS {trend}，最接近的是 {closest} 组，"
              f"全时序 RMS 为 {closest_error:.2f} mm。起跑分辨率从 A 提高到 D，细节 RMS 变化 {detail_delta:+.2f} mm。",
              f"粗层代理使全场误差下降的组数为 {len(helped)}/{len(GROUPS)}"
              + (f"（{'、'.join(helped)}）" if helped else "") + "。"
              f"粗层代理使末帧侧壁覆盖增加的组数为 {len(wrapped)}/{len(GROUPS)}"
              + (f"（{'、'.join(wrapped)}）" if wrapped else "") + "。"
              "侧壁覆盖及全场误差还受接触采样和滑移轨迹影响，需和尖齿特写合读。"]
    closest_native = summary[f"{closest}_native_to_direct_mm"]["rms"]
    closest_proxy = summary[f"{closest}_coarse_to_direct_mm"]["rms"]
    lines += ["", f"{closest} 组的原生粗层、代理粗层和渐进细层对独立细解的全时序 RMS 分别为 "
              f"{closest_native:.2f}、{closest_proxy:.2f} 和 {closest_error:.2f} mm。"
              "全场 RMS 同时衡量布料形状与滑移轨迹，细层新增的形变见上表的细节 RMS 和特写。"]
    if snapshot := result.get("tooth_snapshot"):
        values = snapshot["side_area_pct"]
        deltas = "、".join(f"{group} {values[f'{group}_coarse_side_area_pct'] - values[f'{group}_native_side_area_pct']:+.2f}"
                         for group in GROUPS)
        lines += ["", f"齿尖特写对应的 {snapshot['time_s']:.2f} s，"
                  f"代理相对原生粗层的侧壁覆盖变化分别为 {deltas} 个百分点。"]
    uncapped = result["uncapped_contact_rms_mm"]
    if uncapped:
        rows = "、".join(f"{name} {err:.2f} mm" for name, err in uncapped.items())
        lines += ["", f"进入名义接触时间后，剔除任一算例在当帧触顶的时间点，"
                  f"共同剩余 {len(result['uncapped_contact_frames'])} 帧。各组的细解对独立参考 RMS 为 {rows}。"
                  "后续帧仍继承之前接受的状态。"]
    intersection_audit = result["gear_intersection_audit"]
    intersections = intersection_audit["by_run"]
    flagged = sum(len(frames) for frames in intersections.values())
    checked = intersection_audit["sampled_frames_per_run"] * len(intersections)
    lines += ["", f"对布料原始三角形和内缩 {EXTERIOR_INSET_M * 1e6:g} μm 的齿轮实体做独立裁剪检查。"
              f"检查了 {checked} 个已保存状态，其中 {flagged} 个有三角面与齿轮实体相交。"
              "逐算例帧号和面数见原始数据，此检查的范围为已保存帧上的布料与齿轮。"]
    lines += ["", "![原生和代理粗层的参考误差及侧壁覆盖](coarse_proxy_comparison.png)",
              "", "![横向及跨级误差与细节统计](consistency.png)"]
    energy_path = root / "same_pose_energy.json"
    if energy_path.is_file():
        energy_checks = json.loads(energy_path.read_text())
        lines += ["", "## 同位形弹性能量诊断", "",
                  "分别在原生粗层和代理粗层的五个保存时刻固定粗位形，再用两种能量求值。表中给出原生轨迹 "
                  "0.46 s 的结果，最后一列取两条轨迹十个样本的最大值。采用独立 CPU 能量求值，单位为 J。 "
                  "[逐位形数据](same_pose_energy.json)", "",
                  "| 组 | 原生膜能量 | 代理膜能量 | 原生弯曲能量 | 代理弯曲能量 | 采样最大膜能量差 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in GROUPS:
            row = next(row for row in energy_checks[name]["poses"]
                       if row["case"] == "direct_coarse" and row["frame"] == 46)
            values = [row[key] for key in ("native_membrane_J", "proxy_membrane_J",
                                           "native_bending_J", "proxy_bending_J")]
            values.append(energy_checks[name]["max_membrane_difference_J"])
            lines.append(f"| {name} | " + " | ".join(f"{value:.5g}" for value in values) + " |")
        lines += ["", "B 组的共形细分在全部抽样位形保持膜能量一致。弯曲能量和其他三组的膜能量会随代理离散改变，"
                  "因此同分辨率粗层的包裹变化要结合该组的实际能量差和接触轨迹解释。"]
    keyframe = root / ("gear_zoom_cd_wide_frame046.png" if wide else "gear_zoom_frame046.png")
    if videos and keyframe.is_file():
        lines += ["", f"![0.46 s 局部特写]({keyframe.name})"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    config = json.loads((args.directory / "setup.json").read_text())
    steps = config["steps"]
    verified = verify_saved_runs(args.directory, config)
    result, curves = analyze(args.directory, steps)
    result["verified_runs"] = verified
    (args.directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(args.directory, steps, curves)
    if args.video:
        render(args.directory, steps)
    report(args.directory, config, result, args.video)


if __name__ == "__main__":
    main()
