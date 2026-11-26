"""Compare uniform and mixed VelPro clocks on common physical samples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.pd.audit_drape_contact import gear_clip_geometry, intersecting_frames
from tools.pd.benchmark_d_pcg import LEVELS
from tools.pd.report_d_pcg_benchmark import log_counts, mm_distance
from tools.pd.report_gear_drape_50k_sweep import newton_numbers
from tools.pd.report_velpro_drape import rest_weights, spatial_rms
from tools.pd.run_integrator_benchmark import read_mesh, read_p, lift


def measure(source, root):
    cases, variants = {}, {}
    fine_rest, fine_faces = read_mesh(source / "hierarchy", 2)
    weights = rest_weights(fine_rest, fine_faces)
    clip = gear_clip_geometry(source / "gear.npz")
    with np.load(source.parent / "A/direct_fine/state.npz") as f:
        direct = f["x"][::2]
    baseline = {}
    specs = [("100/100/100",100,100,"100Hz","100Hz"), ("100/100/50",100,50,"100Hz","100-50Hz"),
             ("50/50/100",50,100,"50Hz","50-100Hz"), ("50/50/50",50,50,"50Hz","50Hz")]
    for name, prefix, suffix in (("100/100/100 仅细层 MAS","100Hz","100Hz-mas"),
                                 ("100/100/100 全层 MAS","100Hz-mas-all","100Hz-mas-all")):
        if (root/suffix/LEVELS[-1]/"wall_seconds.json").is_file():
            specs.append((name,100,100,prefix,suffix))
    for label, parent_hz, fine_hz, prefix, suffix in specs:
        paths = [Path(prefix)/LEVELS[0], Path(prefix)/LEVELS[1], Path(suffix)/LEVELS[2]]
        matched, previous_v, adjacent = [], None, []
        for level, (rel, hz) in enumerate(zip(paths, (parent_hz, parent_hz, fine_hz), strict=True)):
            p = root / rel
            settings = json.loads((p/"settings.json").read_text())
            config = settings["config"]
            original = json.loads((source / LEVELS[level] / "settings.json").read_text())
            expected = original["config"]
            expected["dt"] = 1/hz
            expected["newton"]["velocity_tol"] = 1e-5 * hz
            expected["linear_system"]["tol_rate"] = 1e-3 if level == 2 else 1e-5
            use_mas = rel.parts[0] in ("100Hz-mas", "100Hz-mas-all")
            expected["linear_system"]["fem_preconditioner"] = "mas" if use_mas else "diag"
            assert config == expected
            for key in ("mode","proxy","material","effective_material","initial_pins","contact_friction","contact_resistance"):
                assert settings[key] == original[key]
            assert settings["predictor_input"] == (str(p/"predictor.f64") if level else None)
            rest, faces = read_mesh(source / "hierarchy", level)
            with np.load(p/"state.npz") as state:
                x, v, times = state["x"], state["v"], state["step_wall_s"]
                assert state["dt"] == 1/hz
            steps = 120*hz//100
            assert x.shape == v.shape == (steps+1, len(rest), 3) and len(times) == steps
            np.testing.assert_array_equal(x[0], rest)
            assert np.isfinite(x).all() and np.isfinite(v).all()
            np.testing.assert_allclose(np.diff(x,axis=0)*hz,v[1:],rtol=1e-11,atol=1e-11)
            common = x[::hz//50]
            if level:
                matrix = read_p(source/"hierarchy","adjacent",level-1,level)
                indices = np.arange(steps) * (len(previous_v)-1) // steps
                assert np.all(indices/parent_hz <= np.arange(steps)/hz + 1e-14)
                if hz != parent_hz:
                    np.testing.assert_array_equal(indices, np.load(p/"parent_step_indices.npy"))
                np.testing.assert_array_equal(lift(matrix,previous_v[indices]).ravel(),
                                             np.memmap(p/"predictor.f64",dtype="<f8",mode="r"))
                delta = common[1:] - lift(matrix,matched[-1][1:])
                adjacent.append(float(1000*np.sqrt(np.mean(spatial_rms(delta,rest_weights(rest,faces))**2))))
            matched.append(common)
            previous_v = v
            if str(rel) not in cases:
                counts, caps, residuals = newton_numbers(p/"solver.log",steps)
                if label == "100/100/100":
                    baseline[level] = common
                log = (p/"solver.log").read_text()
                if use_mas:
                    active = {row["name"].rsplit("::", 1)[-1] for row in
                              json.loads((p/"systems.json").read_text())["sim_systems"]}
                    assert "FEMMASPreconditioner" in active and "FEMDiagPreconditioner" not in active
                    native = {"StrainLimitingBaraffWitkinShell2D","DiscreteShellBending"}
                    assert active & native == (set() if level < 2 else native)
                    if level < 2:
                        assert "proxies=1 native_energy_slots=0 native_gradient_blocks=0 native_hessian_blocks=0" in log
                        assert "PD proxy compute_energy: membrane_elements=100352 bending_hinges=150080" in log
                    marker = (f"FEM MAS initialized: covered_vertices={len(rest)} total_vertices={len(rest)} "
                              f"proxy_vertices={len(rest) if level < 2 else 0} partitions=")
                    if marker not in log:
                        raise ValueError(f"MAS vertex coverage was not verified in {p}")
                    cmd = json.loads((p/"command.json").read_text())
                    backend = Path(cmd[cmd.index("--backend")+1])
                    with (backend/"libuipc_backend_cuda.so").open("rb") as library:
                        digest = hashlib.file_digest(library, "sha256").hexdigest()
                    assert digest == json.loads((backend/"provenance.json").read_text())["mas_sha256"]
                cases[str(rel)] = dict(hz=hz,steps=steps,solve_s=float(times.sum()),
                    wall_s=json.loads((p/"wall_seconds.json").read_text()),
                    transfer_s=json.loads((p/"transfer_seconds.json").read_text()),
                    mean_newton=float(np.mean(counts)),max_newton=max(counts),caps=len(caps),
                    capped_frames=caps, cap_residuals=residuals, gear_hits=intersecting_frames(x,faces,clip),
                    error_mm=mm_distance(common[1:],baseline[level][1:]),
                    mas_covered_vertices=len(rest) if use_mas else 0,
                    captured_pcg_graph="LinearFusedPCG: captured CUDA graph" in log, **log_counts(log))
        stats = [cases[str(p)] for p in paths]
        err = mm_distance(matched[2][1:],baseline[2][1:])
        variants[label] = dict(
            steps=[s["steps"] for s in stats],caps=[s["caps"] for s in stats],
            pcg_iterations=[s["pcg_iterations"] for s in stats],
            mas_coverage=[s["mas_covered_vertices"] for s in stats],
            solve_s=sum(s["solve_s"] for s in stats),wall_s=sum(s["wall_s"] for s in stats),
            adjacent_rms_mm=adjacent, error_mm=err,
            final_error_mm=mm_distance(matched[2][-1],baseline[2][-1]),
            direct_rms_mm=float(1000*np.sqrt(np.mean(spatial_rms(matched[2][1:]-direct[1:],weights)**2))),
            gear_hit_states=sum(len(s["gear_hits"]) for s in stats))
    result = dict(cases=cases,variants=variants)
    earlier = source.parents[1] / "pd_D_pcg_speed/fine_only" / LEVELS[-1]
    if (earlier / "state.npz").is_file():
        old_settings = json.loads((earlier / "settings.json").read_text())
        settings = json.loads((root / "100Hz" / LEVELS[-1] / "settings.json").read_text())
        assert old_settings["config"] == settings["config"]
        with np.load(earlier / "state.npz") as state:
            result["repeat_100hz_mm"] = mm_distance(baseline[2][1:], state["x"][2::2])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=measure(args.source.resolve(),args.output.resolve())
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    reference=result["variants"]["100/100/100"]["wall_s"]
    rows=["# D 组三层物理频率对照", "",
          "5,041 → 12,769 → 50,625 顶点，固定齿轮，自由落布 1.20 s。PCG 停止比例为 1e-5 / 1e-5 / 1e-3，"
          "Newton 上限 100，线搜索上限 8。100 Hz 使用 dt=0.01 s、velocity_tol=0.001 m/s，"
          "50 Hz 使用 dt=0.02 s、velocity_tol=0.0005 m/s，实际轴向位移阈值均为 0.01 mm。", "",
          "同频使用原 VelPro。异频在子步起点取已经到达的最近父层状态，按子层步长乘延拓速度。"
          "50→100 时每份父层速度用两步，100→50 时读取第 0、2、4…步的父层速度。"
          "这是对同频 VelPro 的分层时钟扩展。该算例没有运动控制器，外力与障碍物在物理时间上相同。", "",
          "| 粗/中/细 (Hz) | 步数 | 求解 s | 端到端 s | 墙钟 s/模拟 s | 加速 | 各层 Newton 触顶 | 细层布点 p50 / p95 (mm) | 对独立 Direct RMS (mm) |",
          "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for k,v in result["variants"].items():
        fmt=lambda a:"/".join(map(str,a))
        rows.append(f"| {k} | {fmt(v['steps'])} | {v['solve_s']:.2f} | {v['wall_s']:.2f} | "
                    f"{v['wall_s']/1.2:.2f} | {reference/v['wall_s']:.2f}× | {fmt(v['caps'])} | "
                    f"{v['error_mm']['p50']:.3f} / {v['error_mm']['p95']:.3f} | {v['direct_rms_mm']:.2f} |")
    rows += ["","各行比较 60 个共同的非初始时刻，直接使用同编号布点。跨层 RMS 使用静止面积加权的相邻层延拓。"
             "每条链的端到端时间含下层运行、层间传递、建场、读回与写盘，混合行按实际共用前缀的耗时计入。",
             "", "[各层 Newton、接触裁剪、末帧误差及相邻层 RMS](metrics.json) · "
             "[实验入口](../../tools/pd/benchmark_d_pcg.py)"]
    if "repeat_100hz_mm" in result:
        rows += ["", f"同一套日常配置的历史 100 Hz 复跑在这些时刻的细层布点 P95 为 "
                 f"{result['repeat_100hz_mm']['p95']:.3f} mm。"]
    rows += ["", "libuipc 的 MAS 对照数据来自高刚度弹性兔。Case 89 同开 CUDA Graph 的 diag / MAS "
             "为约 74 / 29.8 s。Case 88 的全 diag、MAS 仅覆盖上方兔子、MAS 覆盖两只兔子，"
             "分别为 60、73、29 s。前四组布料频率实验均使用 diag。两组 MAS 的时钟均为 100 Hz。 "
             "[库内测量及覆盖范围](../../dependencies/libuipc/agent_docs/09-known-issues-and-roadmap.md)"]
    for name, v in result["variants"].items():
        if "MAS" in name:
            rows += ["", f"{name} 的粗、中、细层实际 MAS 覆盖顶点数为 {v['mas_coverage']}，"
                     f"各层 PCG 总迭代为 {v['pcg_iterations']}。100 Hz 全 diag 的对应计数为 "
                     f"{result['variants']['100/100/100']['pcg_iterations']}。"]
    if "100/100/100 全层 MAS" in result["variants"]:
        rows += ["", "## 分层预条件器对照", "",
                 "| 布料层 | 全 diag 墙钟 s | 全 MAS 墙钟 s | diag / MAS 的 PCG 总迭代 | diag / MAS 的平均 Newton |",
                 "|---|---:|---:|---:|---:|"]
        for level in LEVELS:
            diag = result["cases"][f"100Hz/{level}"]
            mas = result["cases"][f"100Hz-mas-all/{level}"]
            rows.append(f"| {level} | {diag['wall_s']:.2f} | {mas['wall_s']:.2f} | "
                        f"{diag['pcg_iterations']} / {mas['pcg_iterations']} | "
                        f"{diag['mean_newton']:.1f} / {mas['mean_newton']:.1f} |")
        rows += ["", "全层 MAS 使用独立 CUDA 后端，给带有效 PD 代理的二维 Empty 布料分区。"
                 "粗、中层仍然使用受限细层弹性，惯性与 IPC 在当前分辨率装配。MAS 读取当前 FEM 矩阵构造预条件器。"
                 "PCG 比较的是归一化的预条件残差，变更预条件器也会改变这个范数。 "
                 "[后端与补丁摘要](proxy_mas_backend/provenance.json) · "
                 "[MAS 分区补丁](../../patches/pd-proxy-mas-libuipc-0.0.28.patch)"]
    (args.output/"REPORT.md").write_text("\n".join(rows)+"\n")


if __name__=="__main__":
    main()
