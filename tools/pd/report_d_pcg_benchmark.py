"""Summarize solver cost and matched trajectories for the D-group PCG experiment."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from tools.pd.audit_drape_contact import gear_clip_geometry, intersecting_frames
from tools.pd.benchmark_d_pcg import LEVELS
from tools.pd.report_gear_drape_50k_sweep import series
from tools.pd.report_velpro_drape import rest_weights, spatial_rms
from tools.pd.run_integrator_benchmark import lift, read_mesh, read_p


def mm_distance(a, b):
    distance = 1000 * np.linalg.norm(a - b, axis=-1)
    return dict(p50=float(np.median(distance)), p95=float(np.percentile(distance, 95)),
                rms=float(np.sqrt(np.mean(distance**2))), maximum=float(distance.max()))


def rms_mm(x, other, weights):
    return float(1000 * np.sqrt(np.mean(spatial_rms(x[1:] - other[1:], weights)**2)))


def log_counts(log):
    pcg = list(map(int, re.findall(r"Iterative linear solver iteration count: (\d+)", log)))
    pairs = [sum(map(int, found)) for found in re.findall(
        r"SimplexTrajectoryFilter PTs: (\d+), EEs: (\d+), PEs: (\d+), PPs: (\d+)", log)]
    dimensions = {int(value) for value in re.findall(r"GlobalLinearSystem has (\d+) DoFs", log)}
    assert pcg and pairs and len(dimensions) == 1
    return dict(pcg_iterations=sum(pcg), linear_calls=len(pcg), system_dofs=dimensions.pop(),
                peak_active_pairs=max(pairs), p95_active_pairs=float(np.percentile(pairs, 95)))


def profile(directory):
    tree = json.loads((directory / "profile.json").read_text())
    durations = {}

    def visit(node):
        name = node["name"]
        durations[name] = durations.get(name, 0.) + node["duration"]
        for child in node["children"]:
            visit(child)

    visit(tree)
    frame_log = (directory / "solver.log").read_text().split(">>> Begin Frame: 32")[-1]
    return dict(frame_32=log_counts(frame_log), durations_s=durations)


def compare(root, source):
    runs = {key: {} for key in ("base", "candidate")}
    comparisons, adjacent = {}, {}
    previous_x = {}
    clip = gear_clip_geometry(source / "gear.npz")
    direct_dir = source.parent / "A/direct_fine"
    direct, direct_t = series(direct_dir, 50625, 120)
    fine_r, fine_f = read_mesh(source / "hierarchy", 2)
    fine_w = rest_weights(fine_r, fine_f)
    previous_v = {}
    for level, name in enumerate(LEVELS):
        rest, faces = read_mesh(source / "hierarchy", level)
        w = rest_weights(rest, faces)
        states = {}
        for variant, rate in (("base", 1e-5), ("candidate", 1e-3)):
            case = root / variant / name
            settings = json.loads((case / "settings.json").read_text())
            reference = json.loads((source / name / "settings.json").read_text())
            expected = reference["config"]
            expected["linear_system"]["tol_rate"] = rate
            assert settings["config"] == expected
            for key in ("mode", "effective_material", "material", "contact_friction", "contact_resistance", "initial_pins", "proxy"):
                assert settings[key] == reference[key], (name, key)
            x, t = series(case, len(rest), 120)
            np.testing.assert_array_equal(x[0], rest)
            if level:
                p = read_p(source / "hierarchy", "adjacent", level-1, level)
                expected_v = lift(p, previous_v[variant][:-1])
                np.testing.assert_array_equal(np.memmap(case / "predictor.f64", mode="r", dtype="<f8"), expected_v.ravel())
                adjacent.setdefault(name, {})[variant] = rms_mm(x, lift(p, previous_x[variant]), w)
            states[variant] = previous_x[variant] = x
            with np.load(case / "state.npz") as data:
                previous_v[variant] = data["v"]
                accel = spatial_rms(np.diff(data["v"], axis=0), w) / .01
            t.update(log_counts((case / "solver.log").read_text()))
            t.update(readback_s=settings["readback_s"], save_s=settings["save_s"],
                     sync_s=settings.get("sync_s"), transfer_s=json.loads((case / "transfer_seconds.json").read_text()),
                     build_s=settings["build_s"], control_endpoints=120,
                     acceleration_p95=float(np.percentile(accel, 95)), gear_hits=intersecting_frames(x, faces, clip))
            if level == 2:
                t["vs_direct_rms_mm"] = rms_mm(x, direct, fine_w)
            runs[variant][name] = t
        with np.load(source / name / "state.npz") as historical:
            repeated = mm_distance(states["base"][1:], historical["x"][1:])
        comparisons[name] = dict(all_steps=mm_distance(states["candidate"][1:], states["base"][1:]),
                                 last_step=mm_distance(states["candidate"][-1], states["base"][-1]),
                                 repeated_base_vs_previous=repeated)
    profiles = {name: profile(root / folder) for name, folder in zip(
        LEVELS, ("profile_proxy_coarse", "profile_velpro_middle", "diag_baseline"), strict=True)}
    profiles["candidate_fine"] = profile(root / "diag_candidate")
    pilot = {}
    pilot_states = []
    for key in ("diag_baseline", "diag_candidate"):
        case = root / key
        with np.load(case / "state.npz") as state:
            pilot_states.append(state["x"][1:])
            pilot[key] = dict(steps_1_to_31_s=float(state["step_wall_s"][:-1].sum()),
                              **log_counts((case / "solver.log").read_text()))
    pilot["position_mm"] = mm_distance(*pilot_states)
    result = dict(runs=runs, positional_error_mm=comparisons, adjacent_rms_mm=adjacent, profiles=profiles,
                  pilot=pilot, archived_direct=direct_t)
    fine_only = root / "fine_only" / LEVELS[-1]
    if (fine_only / "wall_seconds.json").is_file():
        x, timing = series(fine_only, len(fine_r), 120)
        strict_input = root / "base" / LEVELS[-1] / "predictor.f64"
        np.testing.assert_array_equal(np.memmap(fine_only / "predictor.f64", dtype="<f8", mode="r"),
                                      np.memmap(strict_input, dtype="<f8", mode="r"))
        conf = json.loads((fine_only / "settings.json").read_text())
        expected = json.loads((root / "candidate" / LEVELS[-1] / "settings.json").read_text())
        assert conf["config"] == expected["config"] and conf["effective_material"] == expected["effective_material"]
        np.testing.assert_array_equal(x[0], fine_r)
        with np.load(root / "base" / LEVELS[1] / "state.npz") as middle:
            prolonged = lift(read_p(source / "hierarchy", "adjacent", 1, 2), middle["x"])
        result["fine_only"] = dict(timings=timing, position_mm=mm_distance(x[1:], previous_x["base"][1:]),
                                   final_position_mm=mm_distance(x[-1], previous_x["base"][-1]),
                                   pcg=log_counts((fine_only / "solver.log").read_text()),
                                   vs_direct_rms_mm=rms_mm(x, direct, fine_w),
                                   adjacent_rms_mm=rms_mm(x, prolonged, fine_w),
                                   gear_hits=intersecting_frames(x, fine_f, clip))
    return result


def write_report(root, result):
    refs = json.loads((root / "references.json").read_text())
    fold = refs["fold"]
    rows = ["# D 组 VelPro 内层 PCG 停止比例对照", "",
            "三级网格分别有 5,041、12,769、50,625 个顶点。齿轮固定，重力、布料、IPC、初态和物理时钟与历史 D 组一致。"
            "两套配置各自独立跑完整条链。粗层用 BDF1，中细层保留 VelPro 的本层位置与上层上一步速度。"
            "配置差异仅为 fused PCG 的预条件残差比值阈值，检查间隔为 5。", "",
            "D 组每步 0.01 s，Newton 上限 100，线搜索上限 8，velocity_tol 为 0.001 m/s。"
            "MaxTranslationChecker 检查轴向最大位移 ≤ 0.00001 m，同时要求其他启用的检查器通过、CCD alpha ≥ 1、"
            "已到动画目标且满足 min_iter=0。semi-implicit 关闭。"
            "同步剖析只在单独的短测第 32 步启用。完整运行的求解时间覆盖 advance 与 sync，"
            "端到端从生成本层预测文件到 worker 退出，包括建场、轨迹读回、写盘。整链推进同一段 1.20 s。"
            "各版本串行运行，GPU 同时有其他进程。加速倍数采用本轮配对墙钟，跨次负载会影响其数值。", "",
            "| 算例 / PCG 比例 | 完成步数 | 模拟 s | 求解 s | 端到端 s | 墙钟 s/模拟 s | 配对墙钟加速 | Newton 平均 / 触顶 | 对本层基线布点 p50 / p95 (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    totals = {}
    for label, rate in (("base", "1e-5"), ("candidate", "1e-3")):
        runs = result["runs"][label]
        for name in LEVELS:
            t = runs[name]
            err = result["positional_error_mm"][name]["all_steps"]
            speed = result["runs"]["base"][name]["wall_s"] / t["wall_s"]
            error = "0 / 0" if label == "base" else f"{err['p50']:.3f} / {err['p95']:.3f}"
            rows.append(f"| {name} / {rate} | 120/120 | 1.20 | {t['solver_wall_s']:.2f} | {t['wall_s']:.2f} | "
                        f"{t['wall_s']/1.2:.2f} | {speed:.2f}× | {t['mean_newton']:.2f} / {t['newton_cap_steps']} | {error} |")
        solver = sum(t["solver_wall_s"] for t in runs.values())
        wall = sum(t["wall_s"] for t in runs.values())
        totals[label] = wall
        err = result["positional_error_mm"][LEVELS[-1]]["all_steps"]
        error = "0 / 0" if label=="base" else f"{err['p50']:.3f} / {err['p95']:.3f}"
        caps = sum(t["newton_cap_steps"] for t in runs.values())
        mean = np.mean([t["mean_newton"] for t in runs.values()])
        rows.append(f"| 整链 / {rate} | 3×120 | 1.20 | {solver:.2f} | {wall:.2f} | {wall/1.2:.2f} | "
                    f"{totals['base']/wall:.2f}× | {mean:.2f} / {caps} | {error} |")
    if "fine_only" in result:
        hybrid = result["fine_only"]
        end = hybrid["timings"]
        err = hybrid["position_mm"]
        prefix = [result["runs"]["base"][name] for name in LEVELS[:-1]]
        for label, stages, baseline_s in (("仅细层 / 1e-3", [end], result["runs"]["base"][LEVELS[-1]]["wall_s"]),
                                          ("整链 / 1e-5、1e-5、1e-3", [*prefix, end], totals["base"])):
            wall = sum(stage["wall_s"] for stage in stages)
            solve = sum(stage["solver_wall_s"] for stage in stages)
            caps = sum(stage["newton_cap_steps"] for stage in stages)
            rows.append(f"| {label} | {len(stages)}×120 | 1.20 | {solve:.2f} | {wall:.2f} | "
                        f"{wall/1.2:.2f} | {baseline_s/wall:.2f}× | {np.mean([s['mean_newton'] for s in stages]):.2f} / {caps} | "
                        f"{err['p50']:.3f} / {err['p95']:.3f} |")
    native = fold["direct_baseline/L3_64"]
    fold_chain = [fold[n] for n in ("L1_16", "L2_32", "L3_64")]
    fold_wall = sum(s["wall"] for s in fold_chain)
    fold_solve = sum(s["action_solve"] for s in fold_chain)
    fold_err = fold["fold_end_of_control_frame_p50_p95_mm"]
    for key in ("direct_baseline/L3_64", "L1_16", "L2_32", "L3_64", "L4_128"):
        entry = fold[key]
        speed = "1.00×" if key.startswith("direct") else "—"
        error = (f"{fold_err[0]:.2f} / {fold_err[1]:.2f}" if key=="L3_64" else
                 "0 / 0" if key.startswith("direct") else "—")
        rows.append(f"| 夹爪 {key} / 1e-3 | 600/600 帧 | 10.00 | {entry['action_solve']:.2f} | "
                    f"动作 {entry['wall']:.2f} | {entry['wall']/10:.2f} | {speed} | "
                    f"{entry['iterations']/1200:.2f} / {entry['cap']} | {error} |")
    rows.append(f"| 夹爪 16→32→64 / 1e-3 | 3×600 帧 | 10.00 | {fold_solve:.2f} | 动作 {fold_wall:.2f} | "
                f"{fold_wall/10:.2f} | {native['wall']/fold_wall:.3f}× | "
                f"{sum(s['iterations'] for s in fold_chain)/3600:.2f} / {sum(s['cap'] for s in fold_chain)} | "
                f"{fold_err[0]:.2f} / {fold_err[1]:.2f} |")
    rows += ["", "夹爪档案只在动作窗内有可比的墙钟，表中该组每模拟秒成本、加速比和布点误差采用同轨迹的 Direct 64 作参照。"
             "D 组同层误差取 120 步全体对应顶点距离的 p50 / p95，整链误差取最细层。", "",
             "## 瓶颈与布态", "",
             "下列占比是同步剖析的第 32 步。接触列包括 DCD 和 DyTopo，线搜索列包含其内部 CCD、轨迹候选和试探能量。"
             "每步接触对数取过滤后 PT、EE、PE、PP 总数的峰值。", ""]
    for name in LEVELS:
        p = result["profiles"][name]
        tm = p["durations_s"]
        fractions = [100*value/tm["Pipeline"] for value in (tm["FusedPCG"], tm["Build Linear System"],
                     tm["Detect DCD Candidates"]+tm["Compute DyTopo Effect"], tm["Line Search"])]
        base, cand = (result["runs"][v][name] for v in ("base","candidate"))
        rows.append(f"- {name} 全局系统 {base['system_dofs']:,} DoF，第 32 步最多 {p['frame_32']['peak_active_pairs']:,} 对。"
                    f"PCG、建矩阵、接触、线搜索分别占 {', '.join(f'{v:.1f}%' for v in fractions)}。"
                    f"全程 PCG 迭代 {base['pcg_iterations']:,} → {cand['pcg_iterations']:,}。"
                    f"候选的传层、读回、显式 sync 等待和写盘分别为 {cand['transfer_s']:.3f}、"
                    f"{cand['readback_s']:.3f}、{cand['sync_s']:.3f}、{cand['save_s']:.3f} s。")
    for key, vals in result["adjacent_rms_mm"].items():
        rows.append(f"- 到 {key} 的相邻层面积加权 RMS 为 {vals['base']:.3f} → {vals['candidate']:.3f} mm。")
    if "fine_only" in result:
        h = result["fine_only"]
        rows += ["", f"仅放宽细层时，输入预测文件与严格版逐字节数值一致。中→细 RMS 为 {h['adjacent_rms_mm']:.3f} mm，"
                 f"细层对独立 Direct 的 RMS 为 {h['vs_direct_rms_mm']:.3f} mm，"
                 f"末帧同点 p50 / p95 为 {h['final_position_mm']['p50']:.3f} / {h['final_position_mm']['p95']:.3f} mm，"
                 f"保存状态中齿轮相交帧数为 {len(h['gear_hits'])}。整链计时包含实测严格粗层和中层成本。"]
    pilot = result["pilot"]
    rows += ["", f"细层 32 步同输入短测中，第 1–31 步的非剖析推进为 "
             f"{pilot['diag_baseline']['steps_1_to_31_s']:.2f} → {pilot['diag_candidate']['steps_1_to_31_s']:.2f} s，"
             f"第 32 步同步剖析中的 PCG 为 {result['profiles'][LEVELS[-1]]['durations_s']['FusedPCG']:.2f} → "
             f"{result['profiles']['candidate_fine']['durations_s']['FusedPCG']:.2f} s。"
             f"短测同拓扑布点 P95 为 {pilot['position_mm']['p95']:.3f} mm。"
             "层间传递计时包括读取上层状态、稀疏乘法和写入预测文件。Newton 内代理弹性的位置延拓和导数限制计入矩阵装配与能量评估。"
             "显式 sync 的计时属于推进时间。"]
    rows += ["", "系统维数包括齿轮的 12 个仿射坐标。历史 D 组每层 120 步中分别有 26、45、48 步使用超过 50 轮 Newton。"
             "本次保留 100 轮预算。INFO 日志开销包含在执行时间中，轨迹读回、文件写出和层间传递另行计时。"]
    f = LEVELS[-1]
    b,c = [result["runs"][v][f] for v in ("base","candidate")]
    repeat = result["positional_error_mm"][f]["repeated_base_vs_previous"]
    final = result["positional_error_mm"][f]["last_step"]
    rows += ["", f"细层对同场景独立 50K 原生 IPC 的全时序 RMS 为 {b['vs_direct_rms_mm']:.3f} → {c['vs_direct_rms_mm']:.3f} mm。"
             f"候选末帧对基线的布点 p50 / p95 为 {final['p50']:.3f} / {final['p95']:.3f} mm。"
             f"相同严格参数的新旧渐进细层自身复跑 p50 / p95 为 {repeat['p50']:.3f} / {repeat['p95']:.3f} mm。"
             f"面积加权加速度 P95 为 {b['acceleration_p95']:.2f} → {c['acceleration_p95']:.2f} m/s²。"
             f"六条轨迹已保存状态的齿轮相交数合计 {sum(len(t['gear_hits']) for runs in result['runs'].values() for t in runs.values())}。",
             "", "## 折布与 AGSim 对照口径", ""]
    levels = [fold[n] for n in ("L1_16","L2_32","L3_64")]
    native = fold["direct_baseline/L3_64"]
    rows += [f"既有夹爪首折三级至 64×64 的动作墙钟合计 {sum(x['wall'] for x in levels):.2f} s，"
             f"同轨迹单层 Direct 64×64 为 {native['wall']:.2f} s，整链相对直接解为 "
             f"{native['wall']/sum(x['wall'] for x in levels):.3f}×。三层动作 IPC 合计 "
             f"{sum(x['action_solve'] for x in levels):.2f} s，直接解为 {native['action_solve']:.2f} s。"
             "这是 600 帧、60 Hz 控制、120 Hz 物理的 10 s 动作窗，另有 60 步沉降。"
             "128×128 是从原 64 级续接，16/32 级的代理弹性仍以 64 为目标。"
             "四层各自的 1,200 动作子步中触顶分别为 1、8、52、51，Newton 预算 50，位移阈值 0.08333 mm。",
             f"同一 64×64 拓扑在 600 个控制帧末的渐进解对 Direct 布点 p50 / p95 为 "
             f"{fold['fold_end_of_control_frame_p50_p95_mm'][0]:.2f} / {fold['fold_end_of_control_frame_p50_p95_mm'][1]:.2f} mm。"
             f"直接解抓持判定为左 {native['grasp']['left']}、右 {native['grasp']['right']}，"
             f"渐进 64 级为左 {levels[-1]['grasp']['left']}、右 {levels[-1]['grasp']['right']}。"
             "直接解左侧跟踪片区与 TCP 的距离增长为 47.8 mm。"
             "16 级粗预览、完整三级链与追加的 128 级各自服务于不同的输出分辨率。",
             "", "普通 Scene527 的 55,068 面衣服采用 60/120 Hz、N100、线搜索 8、PCG 1e-3、位移阈值 0.08333 mm。"
             "第一折 1,480 个动作步均未触顶，耦合耗时 2,211.12 s。"
             "AGSim 的 1 s 短测同时改变物理频率和迭代预算，30/N50 与 120/N100 的推进与观察耗时为 44.82 s 与 281.71 s。",
             f"AGSim 完整降频运行完成 {refs['agsim_repro']['final_time_s']:.2f} 模拟秒，执行与 IK 分别为 "
             f"{refs['agsim_repro']['timings_s']['execution']:.2f}、{refs['agsim_repro']['timings_s']['ik']:.2f} s。"
             f"七阶段配对快照以实际 episode 为准，低频 {refs['agsim_pair_snapshot']['low']['tick']} tick，"
             f"高频 {refs['agsim_pair_snapshot']['high']['tick']} tick，共同前缀为 "
             f"{refs['agsim_pair_snapshot']['common_prefix_s']:.2f} s。",
             "", "[完整计数与时间](metrics.json) · [夹爪及 AGSim 数据快照](references.json) · "
             "[夹爪原报告](../pd_gripper_progressive_chain/REPORT.md) · [配对实验入口](../../tools/pd/benchmark_d_pcg.py) · "
             "[PD++ 的 VelPro 定义](https://eriszhang.github.io/assets/pdf/PD%2B%2B.pdf)"]
    if "fine_only" in result:
        h = result["fine_only"]
        hybrid_wall = sum(result["runs"]["base"][name]["wall_s"] for name in LEVELS[:-1]) + h["timings"]["wall_s"]
        rows += ["", "## 本轮配置选择", "",
                 f"日常调试选粗、中层 1e-5 和细层 1e-3。相对同轮严格整链，节省 "
                 f"{totals['base']-hybrid_wall:.2f} s，最细层布点 P95 为 {h['position_mm']['p95']:.3f} mm。"
                 "最终验证选全层 1e-5，保留原有的数值精度标准，并记录 Newton 触顶残差。"
                 "夹爪归档已经使用 PCG 1e-3，此次 D 组实测针对齿轮披覆。"]
    (root / "REPORT.md").write_text("\n".join(rows)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    result = compare(root, args.source.resolve())
    (root/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    write_report(root, result)
