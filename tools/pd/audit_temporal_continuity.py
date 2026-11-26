"""Audit saved Scene527 trajectories and Newton logs without simulation."""
import argparse
import csv
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from tools.pd.report_gripper_progressive_chain import write_csv, write_json
from tools.pd.run_gripper_progressive_chain import CASES, FRAMES, OUTPUT, ROOT, SETTLE, SUBSTEPS

TITLE = "## 同层物理时间连续性定量检验（Temporal Continuity Audit）"
WINDOWS = {
    "全动作 0–599": (0, FRAMES - 1),
    "闭爪起吊 180–245": (180, 245),
    "收拢起折 270–310": (270, 310),
    "自接触压实与释放 400–590": (400, 590),
}
LEVELS = {f"{8 << level}×{8 << level}": CASES[level] for level in range(1, 5)}
CONTROL = "32×32 BDF1"
DEFAULT_CONTROL = ROOT / "outputs/pd_inward_transport/proxy"
ITERATION = re.compile(r"Frame (\d+) Newton Iteration (\d+)")
TRANSLATION = re.compile(r"MaxTranslationChecker> Residual/AbsTol: ([0-9.eE+-]+)/([0-9.eE+-]+)")
LIMIT = re.compile(r"Newton Iteration Exits with Max Iteration: (\d+) \(Frame=(\d+)\)")
CAP_FIELDS = ("ipc_frame", "source_frame", "source_substep", "iteration_zero_based",
              "residual_m", "abs_tol_m", "residual_over_tol")


def summarize(values):
    return dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)), max=float(values.max()))


def audit_velocity(path):
    steps = FRAMES * SUBSTEPS
    with np.load(path, allow_pickle=False) as archive:
        h = archive["dt"].item()
        velocity = archive["v"]
        if (archive["settle_steps"].item(), archive["substeps"].item()) != (SETTLE, SUBSTEPS):
            raise ValueError(f"{path}: unexpected time indexing")
    if (velocity.ndim != 3 or velocity.shape[0] != SETTLE + steps + 1 or velocity.shape[2] != 3
            or not np.isclose(h * SUBSTEPS, 1 / 60, rtol=0, atol=1e-15)):
        raise ValueError(f"{path}: unexpected velocity shape or timestep")

    # Label by ending IPC substep (t+1). The first sample uses the last two settle states.
    a, d2 = np.empty(steps), np.empty(steps)
    for i in range(0, steps, 32):
        j = min(i + 32, steps)
        batch = velocity[SETTLE + i - 1:SETTLE + j + 1]
        if not np.isfinite(batch).all():
            raise ValueError(f"{path}: nonfinite velocities")
        dv = np.diff(batch, axis=0)
        a[i:j] = np.linalg.norm(dv[1:], axis=(1, 2)) / (h * np.sqrt(batch.shape[1]))
        d2[i:j] = np.linalg.norm(np.diff(dv, axis=0), axis=(1, 2))
    return a, d2, h, velocity.shape[1]


def audit_residuals(directory):
    caps = []
    iteration, check = None, None
    with (directory / "simulation.log").open() as log:
        for line in log:
            if match := ITERATION.search(line):
                iteration, check = tuple(map(int, match.groups())), None
            elif match := TRANSLATION.search(line):
                check = tuple(map(float, match.groups()))
            elif match := LIMIT.search(line):
                limit, ipc = map(int, match.groups())
                if limit != 50 or iteration != (ipc, limit - 1) or check is None:
                    raise ValueError(f"{directory.name}: missing final iteration check at IPC {ipc}")
                residual, tolerance = check
                if not (np.isfinite(check).all() and residual >= 0 and tolerance > 0):
                    raise ValueError(f"{directory.name}: invalid final check at IPC {ipc}")
                caps.append(dict(zip(CAP_FIELDS, (
                    ipc, (ipc - SETTLE - 1) // SUBSTEPS, (ipc - SETTLE - 1) % SUBSTEPS + 1,
                    limit - 1, residual, tolerance, residual / tolerance))))
    with (directory / "newton_substeps.csv").open() as file:
        expected = [int(row["ipc_frame"]) for row in csv.DictReader(file) if row["hit_limit"] == "1"]
    if [row["ipc_frame"] for row in caps] != expected:
        raise ValueError(f"{directory.name}: cap list differs from newton_substeps.csv")
    return caps


def plot(directory, curves):
    fig, axes = plt.subplots(len(curves), 1, sharex=True, sharey=True, figsize=(10, 8.4),
                             layout="constrained")
    x = np.arange(FRAMES * SUBSTEPS) / SUBSTEPS
    colors = ("#25719c", "#177568", "#7956a4", "#a5521d", "#54575b")
    for ax, (label, (a, _, _, _)), color in zip(axes, curves.items(), colors, strict=True):
        for (start, end), shade in zip(list(WINDOWS.values())[1:], ("#d6ebf5", "#f7dfab", "#e7def0")):
            ax.axvspan(start, end + 1, color=shade, alpha=0.65, lw=0)
        ax.plot(x, a, color=color, lw=0.75)
        suffix = " BDF1" if label == "16×16" else " paper_eq5" if label in LEVELS else " (control)"
        ax.set_title(label + suffix, loc="left", fontsize=10)
        ax.grid(axis="y", color="#ced2d7", alpha=0.55, linewidth=0.5)
    axes[-1].set(xlim=(0, FRAMES), xlabel="Source action frame (zero-based)")
    fig.supylabel("Vertex RMS acceleration (m/s²)")
    fig.suptitle("Scene527 cloth · acceleration at 120 IPC substeps/s", fontsize=14)
    fig.savefig(directory / "temporal_acceleration.png", dpi=160)
    plt.close(fig)


def update_report(report, statistics, residuals, comparison_note):
    lines = [
        TITLE, "",
        "用已保存的布料速度计算，步长 $h=1/120$ s。源帧从 0 起算，每帧两子步。"
        "窗口两端均含，指标按 $t+1$ 所在动作帧归类，首个动作子步使用末尾两个沉降状态。", "",
        r"$a_{\rm RMS}=\|v^{t+1}-v^t\|_2/(h\sqrt{n_l})$ 为顶点三维加速度的 RMS。"
        r"$\|\Delta^2v\|_2=\|v^{t+1}-2v^t+v^{t-1}\|_2$ 为二阶速度差分，"
        r"另列除以 $\sqrt{n_l}$ 的值，便于比较不同顶点数。各格依次为均值、P95、最大值。", "",
        "| 层级 | 动作窗口 | 子步数 | 加速度 RMS (m/s²) | 二阶速度差分 L₂ (m/s) | 二阶差分顶点 RMS (m/s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, windows in statistics.items():
        for name, stats in windows.items():
            values = [" / ".join(f"{stats[metric][key]:.3g}" for key in ("mean", "p95", "max"))
                      for metric in ("accel_rms", "d2_l2", "d2_rms")]
            lines.append(f"| {label} | {name} | {stats['samples']} | {' | '.join(values)} |")
    lines += ["", "![各层逐子步离散加速度](temporal_acceleration.png)", "",
              comparison_note, "",
              "加速度和二阶差分的峰值也可能来自真实接触冲击。不同分辨率的对照同时改变了网格，"
              "同分辨率对照沿用已有运行，轨迹与代理资产相同，CUDA 后端构建不同。"
              "这些运动学指标可以定位异常候选时段，尚不足以把峰值归因为非物理抖动。", "",
              "**Newton 上限核查**。第 50 次迭代的日志编号为 49。MaxTranslationChecker 记录"
              "线搜索接受试步前的全局 Newton 方向最大绝对坐标分量，单位为 mm。", "",
              "| 层级 | 达上限子步 | 阈值 (mm) | 中位残差 (mm) | 最大残差 (mm) | 残差 / 阈值 中位 / 最大 | 超阈值子步 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, rows in residuals.items():
        residual = np.array([row["residual_m"] for row in rows])
        tolerances = sorted({row["abs_tol_m"] * 1000 for row in rows})
        ratio = np.array([row["residual_over_tol"] for row in rows])
        tol = " / ".join(f"{value:.4g}" for value in tolerances)
        lines.append(f"| {label} | {len(rows)} | {tol} | {1000*np.median(residual):.4g} | "
                     f"{1000*residual.max():.4g} | {np.median(ratio):.2f} / {ratio.max():.2f} | "
                     f"{np.count_nonzero(ratio > 1)} |")
    above = sum(row["residual_over_tol"] > 1 for rows in residuals.values() for row in rows)
    total = sum(map(len, residuals.values()))
    lines += ["", f"**{total} 个达到迭代上限的子步中，有 {above} 个超出位移收敛阈值。** "
              "这些日志无法证明所有上限子步都处于小残差收敛域。"
              "接受最后一次线搜索步后的力平衡残差需要另外测量。", "",
              "[逐子步曲线](temporal_continuity_substeps.csv) · "
              "[逐条退出残差](temporal_newton_limit_residuals.csv) · "
              "[统计 JSON](temporal_continuity_summary.json) · "
              "[后处理脚本](../../tools/pd/audit_temporal_continuity.py)"]
    text = report.read_text()
    before, marker, after = text.partition(TITLE)
    next_section = re.search(r"^## ", after, re.M) if marker else None
    following = after[next_section.start():] if next_section else ""
    report.write_text(before.rstrip() + "\n\n" + "\n".join(lines) + "\n"
                      + ("\n" + following if following else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bdf1-control", type=Path, default=DEFAULT_CONTROL)
    args = parser.parse_args()
    directory = args.output.resolve()
    curves = {label: audit_velocity(directory / case / "state.npz") for label, case in LEVELS.items()}
    curves[CONTROL] = audit_velocity(args.bdf1_control / "state.npz")
    if curves[CONTROL][2:] != curves["32×32"][2:]:
        raise ValueError("BDF1 control differs in timestep or vertex count")
    stats = {}
    for label, (a, d2, _, vertices) in curves.items():
        stats[label] = {}
        for name, (start, end) in WINDOWS.items():
            window = slice(start * SUBSTEPS, (end + 1) * SUBSTEPS)
            stats[label][name] = dict(samples=len(a[window]), accel_rms=summarize(a[window]),
                                      d2_l2=summarize(d2[window]), d2_rms=summarize(d2[window] / np.sqrt(vertices)))
    residuals = {label: audit_residuals(directory / case) for label, case in LEVELS.items()}
    write_csv(directory / "temporal_continuity_substeps.csv",
              ("level", "ipc_frame", "source_frame", "source_substep",
               "accel_rms_m_s2", "d2_l2_m_s", "d2_rms_m_s"),
              ((label, SETTLE + i + 1, i // SUBSTEPS, i % SUBSTEPS + 1, rms, change, change / np.sqrt(vertices))
               for label, (a, d2, _, vertices) in curves.items()
               for i, (rms, change) in enumerate(zip(a, d2, strict=True))))
    write_csv(directory / "temporal_newton_limit_residuals.csv", ("level", *CAP_FIELDS),
              ((label, *[row[key] for key in CAP_FIELDS]) for label, rows in residuals.items() for row in rows))
    all_action = next(iter(WINDOWS))
    pd = stats["32×32"][all_action]
    bdf = stats[CONTROL][all_action]
    pd_a, bdf_a = pd["accel_rms"], bdf["accel_rms"]
    pd_d2, bdf_d2 = pd["d2_rms"], bdf["d2_rms"]
    control_command = os.path.relpath((args.bdf1_control / "command.json").resolve(), directory)
    comparison_note = (
        f"**同分辨率 32×32 对照**。全动作加速度 P95 为 {pd_a['p95']:.3g} / {bdf_a['p95']:.3g} m/s²，"
        f"最大值为 {pd_a['max']:.3g} / {bdf_a['max']:.3g} m/s²。"
        f"顶点归一化二阶速度差分 P95 为 {pd_d2['p95']:.3g} / {bdf_d2['p95']:.3g} m/s，"
        f"最大值为 {pd_d2['max']:.3g} / {bdf_d2['max']:.3g} m/s，顺序均为 paper_eq5 / BDF1。"
        f" [既有 BDF1 对照命令]({control_command})"
    )
    write_json(directory / "temporal_continuity_summary.json", dict(
        windows=WINDOWS, statistics=stats,
        inputs={label: dict(vertices=n, dt_s=h) for label, (_, _, h, n) in curves.items()},
        newton_cap_rows=residuals,
    ))
    plot(directory, curves)
    update_report(directory / "REPORT.md", stats, residuals, comparison_note)
    print(f"Audited {len(curves)} trajectories and {sum(map(len, residuals.values()))} Newton cap events.")


if __name__ == "__main__":
    main()
