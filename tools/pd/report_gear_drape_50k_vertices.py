"""Compare a fresh native IPC trajectory with A–D on their shared finest grid."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.pd.audit_drape_contact import EXTERIOR_INSET_M, gear_clip_geometry, intersecting_frames
from tools.pd.report_gear_drape_50k_sweep import load_gear, report as sweep_report, series, side_coverage
from tools.pd.report_velpro_drape import plt, render_comparison, rest_weights, spatial_rms, stats
from tools.pd.run_gear_drape_50k_sweep import GROUPS
from tools.pd.run_integrator_benchmark import read_mesh
from tools.pd.run_velpro_drape import DT


COLORS = {"Direct": "#284a64", "A": "#ae704d", "B": "#80618b", "C": "#29857c", "D": "#457fbd"}


def compare(root, sweep, steps):
    direct = root / "direct_50k_vertices"
    original = sweep / "A/direct_fine"
    paths = {"Direct": direct, **{group: sweep / group / "velpro_fine" for group in GROUPS},
             "Previous Direct": original}
    with np.load(root / "gear.npz", allow_pickle=False) as gear:
        for group in GROUPS:
            with np.load(sweep / group / "gear.npz", allow_pickle=False) as comparison:
                for key in ("vertices", "faces"):
                    np.testing.assert_array_equal(comparison[key], gear[key])
    rest, faces = read_mesh(root / "hierarchy", 0)
    assert len(rest) == 50625 and len(faces) == 100352
    reference = json.loads((original / "settings.json").read_text())
    states, timings = {}, {}
    for name, directory in paths.items():
        if name != "Direct":
            group = name if name in GROUPS else "A"
            other_rest, other_faces = read_mesh(sweep / group / "hierarchy", 2)
            np.testing.assert_array_equal(other_rest, rest)
            np.testing.assert_array_equal(other_faces, faces)
        settings = json.loads((directory / "settings.json").read_text())
        for key in ("config", "material", "effective_material", "initial_pins", "contact_friction", "contact_resistance"):
            if settings[key] != reference[key]:
                raise ValueError(f"{directory}: {key} differs from the original Direct Fine run")
        assert settings["cells"] == 224 and settings["proxy"] is None
        assert (settings["mode"], bool(settings["predictor_input"])) == (
            ("velpro", True) if name in GROUPS else ("bdf1", False))
        systems = {entry["name"].rsplit("::", 1)[-1] for entry in
                   json.loads((directory / "systems.json").read_text())["sim_systems"]}
        assert {"StrainLimitingBaraffWitkinShell2D", "DiscreteShellBending", "FiniteElementBDF1Kinetic"} <= systems
        states[name], timings[name] = series(directory, len(rest), steps)
        np.testing.assert_array_equal(states[name][0], rest)

    weights = rest_weights(rest, faces)
    gear, outline, bottom, top = load_gear(sweep)
    clip = gear_clip_geometry(root / "gear.npz")
    curves, intersections = {}, {}
    for name, x in states.items():
        curves[f"{name}_error_mm"] = 1000 * spatial_rms(x - states["Direct"], weights)
        curves[f"{name}_coverage_pct"] = 100 * side_coverage(x, weights, outline, bottom, top, .010)
        intersections[name] = intersecting_frames(x, faces, clip)
    first_contact = int(np.ceil(np.sqrt(2 * (rest[0, 2] - top) / 9.81) / DT))
    uncapped = np.arange(steps + 1) >= first_contact
    for name in (*GROUPS, "Direct"):
        uncapped[timings[name]["capped_frames"]] = False
    result = {
        "timings": timings,
        "summary": stats({k: v[1:] for k, v in curves.items()}),
        "common_uncapped_frames": np.flatnonzero(uncapped).tolist(),
        "uncapped_rms_mm": {name: float(np.sqrt(np.mean(curves[f"{name}_error_mm"][uncapped]**2)))
                            for name in GROUPS} if np.any(uncapped) else {},
        "gear_intersections": intersections,
    }
    (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    t = np.arange(steps + 1) * DT
    np.savetxt(root / "curves.csv", np.column_stack([t, *curves.values()]), delimiter=",",
               header=",".join(["time_s", *curves]), comments="")
    return states, faces, gear, curves, result, t


def render(root, states, faces, gear, curves, t):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")
    for ax, key, title, units in zip(axes, ("error_mm", "coverage_pct"),
                                     ("Distance to freshly run Direct IPC", "Gear side coverage"),
                                     ("RMS (mm)", "Rest-area (%)"), strict=True):
        for name, color in COLORS.items():
            if name != "Direct" or key != "error_mm":
                ax.plot(t, curves[f"{name}_{key}"], label=f"VelPro {name}" if name in GROUPS else "Direct IPC", color=color)
        ax.set(title=title, xlabel="Time (s)", ylabel=units)
        ax.grid(alpha=.2)
        ax.legend(ncols=2)
    fig.savefig(root / "comparison_curves.png", dpi=140)
    plt.close(fig)
    order = ("Direct", "A", "B", "Direct", "C", "D")
    labels = [f"{name} · VelPro, 50,625 vertices" if name in GROUPS else "Direct IPC · 50,625 vertices"
              for name in order]
    views = {"overview": {}, "zoom": dict(camera_angles=(16., -42.), focus=((.075, -.13, .22), .41))}
    for name, camera in views.items():
        render_comparison(gear, [states[case] for case in order], [faces] * len(order), labels,
                          [COLORS[case] for case in order], root / f"{name}_wide.mp4", rows=2, figsize=(16, 9),
                          keyframes=(20, 46, 80, 120), **camera)
        print(f"Rendered {name}", flush=True)


def write_report(root, sweep, curves, result):
    reference_md = Path("..") / sweep.name / "REPORT.md"
    lines = ["# 方布落齿轮 · 50,625 顶点原生 IPC 与 A–D 最细层对照", "",
             "[横屏全景](overview_wide.mp4) · [横屏齿尖特写](zoom_wide.mp4) · [逐帧曲线](curves.csv) · "
             "[求解与接触数据](metrics.json) · [新增原始轨迹](direct_50k_vertices/state.npz)", "",
             "直接仿真复用四组共同的最细网格，224×224 格，50,625 顶点、100,352 三角面。"
             "新算例使用原生膜和弯曲弹性、BDF1 与摩擦 IPC，独立从静止状态跑 120 步。"
             "A–D 四条既有 VelPro 轨迹在最细层同样使用原生弹性，保留各自的上一层速度预测。"
             "齿轮、材料、步长 0.01 s、摩擦和求解容差一致。 "
             f"[原四组扫描]({reference_md})", "",
             "逐帧位置差按完全一致的顶点编号直接相减，再用静止面积加权。表中的全时序 RMS 排除初始帧。"
             "侧壁覆盖为齿轮顶面以下、距侧壁 10 mm 内的布料静止面积百分比。"
             "墙钟含初始化与写盘，新直接解启动时 GPU 上另有计算负载。", "",
             "| 算例 | 对新直接解 RMS (mm) | 末帧 RMS (mm) | 侧壁覆盖 0.46 / 1.20 s (%) | 最细层墙钟 (s) | Newton 触顶步 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("Direct", *GROUPS, "Previous Direct"):
        measure = result["summary"][f"{name}_error_mm"]
        coverage = curves[f"{name}_coverage_pct"]
        timing = result["timings"][name]
        title = {"Direct": "新原生 IPC", "Previous Direct": "原扫描的原生 IPC"}.get(name, f"{name} 组 VelPro 最细层")
        lines.append(f"| {title} | {measure['rms']:.2f} | {measure['last']:.2f} | {coverage[46]:.2f} / {coverage[-1]:.2f} "
                     f"| {timing['wall_s']:.1f} | {timing['newton_cap_steps']} |")
    rank = sorted(GROUPS, key=lambda name: result["summary"][f"{name}_error_mm"]["rms"])
    lines += ["", "按全时序 RMS 排名，从最接近新直接解到差异最大依次为 " + "、".join(rank) + "。"]
    if result["uncapped_rms_mm"]:
        listed = "、".join(f"{name} {value:.2f} mm" for name, value in result["uncapped_rms_mm"].items())
        lines += [f"五条轨迹在名义接触时间后共同未触顶的 {len(result['common_uncapped_frames'])} 帧中，"
                  f"各组相对新直接解的 RMS 为 {listed}。筛选仍保留早期触顶步对后续状态的影响。"]
    caps = [row["residual_m"] * 1000 for row in result["timings"]["Direct"]["capped_translation"].values()]
    if caps:
        lines += [f"新直接解触顶步的最大退出平移残差为 {max(caps):.3f} mm。"]
    intersections = result["gear_intersections"]
    lines += [f"在六条轨迹共 {len(intersections) * len(curves['Direct_coverage_pct']):,} 个保存状态中，"
              f"原始三角面与内缩 {EXTERIOR_INSET_M * 1e6:g} μm 的齿轮实体相交的状态共 "
              f"{sum(len(frames) for frames in intersections.values())} 个。", "",
              "![逐帧误差和包裹曲线](comparison_curves.png)", "", "![0.46 s 齿尖特写](zoom_wide_frame046.png)"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory.resolve()
    setup = json.loads((root / "setup.json").read_text())
    sweep = Path(setup["sweep"])
    states, faces, gear, curves, result, t = compare(root, sweep, setup["steps"])
    render(root, states, faces, gear, curves, t)
    write_report(root, sweep, curves, result)
    sweep_config = json.loads((sweep / "setup.json").read_text())
    sweep_report(sweep, sweep_config, json.loads((sweep / "metrics.json").read_text()), True)


if __name__ == "__main__":
    main()
