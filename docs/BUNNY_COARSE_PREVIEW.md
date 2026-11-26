# Bunny 粗层非下降停止接入（2026-11-16）

针对 `BUNNY_NATIVE_STALL_AUDIT.md` 中原生PSQ Bunny第18步的失败，新增显式 `--coarse-preview`。默认仍为严格模式；物理参数、梯度、Hessian、接触及应变限制不变。

## 实测验证

结果目录：`output/bunny_preview_20261116/`。独立源码构建成功；同一最终构建的三组测试如下：

| 测试 | 结果 |
|---|---|
| L0→90k代理，开启粗层预览 | 完成25/25步（0.25秒），18步严格收敛，7步非下降预览停止，0步强制接受 |
| 同输入，关闭预览 | 接受17步，第18步仍按严格线搜索失败退出 |
| 恒等映射，初始拉伸1.045 | 两步均严格收敛，0步预览停止 |

预览停止发生在18、19、20、22、23、24、25步。第21步恢复严格收敛，表明停止状态没有跨帧残留。保存轨迹均有限；逐帧重算最大主拉伸1.0570188746，小于1.068上限。首次预览停止之前与严格控制轨迹最大位置差1.02895e-10 m。每个预览停止的两尺度斜率、噪声阈值与结构化状态均通过检查。

恒等映射开启预览、预览与强制接受同时开启，两项非法配置都在启动仿真前被拒绝。`guard_checks_and_provenance.json`保存检查结果及源码摘要；每个run.json另存实际加载DLL摘要。可用以下命令复核已有结果：

```powershell
.venv/Scripts/python.exe scripts/validate_bunny_preview.py output/bunny_preview_20261116
```

`preview/contact.gif`和`preview/final.png`展示保存的实际轨迹，并标注预览停止；没有把它们标为全程Newton收敛。原生CCD及应变可行步限制保持启用，本轮未增加独立逐帧碰撞审计。此结果只验证短程粗层策略接入，不证明完整披覆、多层细化、最细层实际长轨迹或论文性能已复现。

## 接受规则

- 正常达到原生Newton容差：`newton_converged`，`converged=true`。
- 当前层确实低于代理最细层，且显式开启预览；经过至少12次回溯仍无可信进展时，对完整非线性能量做两个尺度的正向试探。只有确认方向非下降才停止为 `preview_non_descent`，`converged=false`。
- 预览停止恢复并保留本次Newton迭代起点，不应用任何不下降试探；仍按当前物理时间步的起始状态更新速度。它不是恢复整步初态，也不是强制接受一个更新。
- 无法确认非下降、仅耗尽Newton预算、状态不可行等情况仍按严格失败处理。预览与强制接受互斥。
- Python拒绝为恒等/最细层开启预览；原生端还检查代理顶点数严格大于当前层顶点数且正在替换弹性。不能只靠一个环境变量让恒等或原生reference绕过严格收敛。

检测器在已经过可行性和CCD检查的正向扫掠范围内，用原可行步长的1e-3和1e-4评估真实总能量。两次能量增加必须大于 `64*machine_epsilon*energy_scale`，且两次正向差分斜率相差不超过较大斜率的25%。若原线搜索试探已取得超过噪声尺度的能量下降，则不触发预览停止。检测未通过时恢复原试探状态，继续原有分支。

这些数值阈值是本项目的保守检测规则，并非已找回的作者实现；两个有限差分尺度也不是任意状态下真实方向导数的数学证明。PD2024 §4.6支持的原则是粗层允许非下降退出、最细层严格收敛。当前接入应通过短测逐步验证，不能把预览停止统计为Newton成功。

## 记录与入口

原生 `engine.frame_stats()` 新增 `preview_non_descent` 和 `termination_reason`。Bunny运行器将每步原生状态写入 `steps.json`，并在 `run.json` 分别记录 `strictly_converged_steps`、`preview_stopped_steps`。报告及动画单独标注预览停止，保留历史强制接受分类。

```powershell
.venv/Scripts/python.exe scripts/run_psq_probe.py --bundle output/libuipc_psq_native/L0_to_90k.bin --steps 25 --coarse-preview --output output/NEW_BUNNY_PREVIEW
.venv/Scripts/python.exe scripts/report_libuipc_probe.py output/NEW_BUNNY_PREVIEW
```

实现源文件：`native/libuipc/psq_preview_stop.inl`；接入可通过 `apply_psq_patch.py` 重建。仅修改独立源码版后端，没有替换已安装wheel。这个入口仍是L0预览，不包含VelPro细层轨迹。

## L0驱动L1的0.25秒尝试

2026-11-16：新增 `--level 1 --lower <L0目录>`。输入为本页已完成的L0预览轨迹。L1为2663顶点、5199面，90k最细网格只用于PSQ弹性代理；L1仍低于全局最细层，因此使用相同的粗层预览停止策略。

每步求解前只通过原生FEM状态接口写入 `U01 @ (x_L0[n]-x_L0[n-1])/dt`，第一步初速度为零。L1位置保持自身上一帧结果，使用水平warm start。接口使用经核验的布料offset=3135，不向固定兔子写入速度。这是VelPro细化，不是独立L1仿真，也不是把L0末态直接插值成L1初态。

输出 `output/bunny_L1_velpro_20261116/run`：请求25步，实际接受20步（0.20秒），其中17步严格收敛、3步预览停止、0步强制接受；第21步第66次零基Newton迭代达到32次回溯上限。CCD/CFL系数为1；预览检测的两个斜率为8.53934e-15和8.72709e-16，相差过大，未通过一致性判据，按严格失败保留结果。没有放宽判据以完成目标时长，也没有计算第21–25步的动态结果。

已接受轨迹最大主拉伸1.0530003321，小于1.068；全部状态有限。每步预测量与CPU `progressive.integrators.predict('velpro',...)` 对照最大差1.11e-16 m；实际速度写入回读误差、位置被意外改动量及固定兔子运动均为零。结果及源码/DLL摘要在 `validation.json`；未做额外独立碰撞审计。动画 `run/contact.gif`仅展示0–0.20秒并标注INCOMPLETE，不能作为0.25秒完成证明。

```powershell
.venv/Scripts/python.exe scripts/run_psq_probe.py --bundle output/bunny_L1_velpro_20261116/L1_to_90k.bin --level 1 --lower output/bunny_preview_20261116/preview --steps 25 --coarse-preview --output output/NEW_BUNNY_L1
```

后续定位对象为L1第21步。当前斜率随试探步长明显变化，不能将它直接归为此前L0已确认的同一种非下降状态；应检查更小尺度的方向导数、曲率和能量精度后再决定求解策略。
