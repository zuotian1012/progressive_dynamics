# 基于已完成 L0 的旗帜 L1

**Direct 定义修正：用户要求的是不用 PD/PSQ、直接在 L1 本层仿真。符合此定义的结果是 `L1_direct_local`，最新四栏动画位于 `report/with_local_direct/`。旧 `L1_direct` 保留了细网格 PSQ 弹性能，不符合该定义，仅作历史记录。**

L0：`output/flag_L0_two_pins_wind300_20261122/L0`。
L1：`output/flag_L1_two_pins_wind300_20261122`，包含 `L1_semi` 和 `L1_velpro`。

复用从原始细网格实际简化得到的同一层级，L0 为 911 顶点、1,044 三角形，L1 为 1,189 顶点、1,600 三角形。两个 L1 版本都使用已完成的 L0 位置轨迹，不重新运行 L0。L1 重新导出的是 L1→原始 35,000 顶点网格的 PSQ 弹性代理算子，实际求解的是 L1 自由度。

仍固定左侧上下两个端点；风载为 `m_i [24 + 15 sin(2π × 1.3 t), 0, 0]`；初始位置采用相同解析 1 mm 微扰场在各层节点上的采样，初始物理速度为零。材料静止形状仍为平面，E=30,000 Pa，ν=0.3，密度 200 kg/m³，全厚度 0.4 mm。步长 0.04 s，200 步，8 s。

## 两种层间更新

记当前帧为 n、时间步长为 h，L0 位移增量为 `Δx0 = x0[n] − x0[n−1]`，n=0 时增量取零。

- Semi-diagonal：惯性预测的无外力部分为 `P01(x0[n]) + U01 Δx0`。
- VelPro：惯性预测的无外力部分为 `x1[n] + U01 Δx0`。

通过原生状态接口注入 `v = (预测位置 − x1[n]) / h`，两个固定点的注入速度置零；每一步回读确认位置未被覆盖、速度正确写入。原生求解器随后加入 L1 的重力、风载、惯性、PSQ 弹性、应变限制和自接触响应。没有额外软位置一致性约束，没有仅将 L0 轨迹上采样充当 L1 仿真。

本例平面参考网格的 PSQ 静止偏移为数值零，故位置延拓 P01 等于 U01 的线性延拓至舍入精度；这不意味着一般曲面上的 PSQ 可以忽略非线性偏移。Semi-diagonal 第一帧的细层解析初态与粗层延拓初态可能存在插值差异，此差异按上述预测公式处理。

## 验证约定

所有保存时间步要求原生 Newton 收敛，不接受预览或迭代预算耗尽。验证端点不动、保存状态无自交、主伸长低于 1.1。保存实际注入速度，并由独立报告逻辑逐步重算上述预测公式；保存和核验父轨迹哈希。检查每步质量加权体力、材料网格、相同求解配置和运行库。

接触刚度请求值仍为 1e9，原生求解器按网格自适应限制：L0 初始化值为 66,890,778.10893504，L1 为 51,251,050.34250633。因此物理输入和接触算法设置一致，但不能声称内部自适应接触刚度逐项一致。

报告中 L1 相对 L0 延拓位置的 RMS 距离只表示层间差异，不是相对于独立细层真值的误差，也不足以据此判断哪种方法更准确。

## 本次结果

| 指标 | Semi-diagonal L1 | VelPro L1 |
|---|---:|---:|
| 收敛步数 | 200/200 | 200/200 |
| 相对 L0 延拓的逐帧 RMS 距离均值（不含初始帧） | 4.637 mm | 17.023 mm |
| 相对 L0 延拓的逐帧 RMS 距离最大值 | 10.700 mm | 34.485 mm |
| 8 秒时相对 L0 延拓的 RMS 距离 | 5.471 mm | 16.191 mm |
| 最后 2 秒自由边平均高度 | 0.518625 m | 0.510664 m |
| 最大主伸长 | 1.0995816 | 1.0995868 |
| 逐步预测速度核验最大差 | 0 | 0 |
| 固定端点最大位移误差 | 0 | 0 |

两种 L1 的 PSQ 细网格对应端点也保持固定，全部 201 个保存状态未检测到 L1 自交。两种 L1 的质量均为 0.12 kg，逐步体力按当前层面积集总质量核验。

整体展开形态与 L0 接近。Semi-diagonal 与 L0 延拓结果更接近；VelPro 与其偏离更多，符合保留 L1 自身位置历史的更新方式。以上是相对于 L0 延拓的指标，不将偏离程度解释成精度优劣，也不单凭位置差推断褶皱数量。新增 Direct L1 的对照见下节。

## 历史对照：独立时间积分但仍保留 PSQ（不符合用户的 Direct 定义）

`L1_direct` 使用相同的 1,189 顶点网格、初始位置和零初始速度、两端固定、风载、材料、接触设置、求解器配置以及 L1→35,000 顶点的 PSQ 弹性代理。仅取消层间更新，不读取 L0 轨迹，也不向每个时间步注入外部速度。由原生求解器保留自身速度历史，按完整独立时间积分推进。

每步在推进前回读 Direct 自身速度，核验其等于自身前两帧的位置差除以 0.04 s（第一步速度为零）。记录在 `L1_direct/pre_step_velocities.npy`。审计确认未提供父轨迹、没有逐步速度注入文件，并对比三种 L1 的初始位置和实际风力完全相同。

该 Direct 是“相同 PSQ 弹性能下的独立 L1 时间积分”，不是另换成本层弹性能的模型，也不是更细网格收敛真值。相对 Direct 的距离衡量与这个独立积分结果的一致程度，不能单独作为所有场景中的准确性结论。

四组动画位于 `report/with_direct/comparison.gif`、`comparison.mp4`；对应数值位于该目录的 `comparison.json` 和 `comparison_metrics.csv`。左下角曲线为 Semi-diagonal 和 VelPro 相对 Direct L1 的逐帧节点位置 RMS 距离。

Direct L1 完成 200/200 步，全部收敛；两个粗层固定端点和 PSQ 细层对应端点位移误差均为 0，201 个保存状态未检测到 L1 自交。最大主伸长为 1.099590941，低于 1.1。自身速度历史的最大核验误差为 4.44e-16 m/s。

| 相对 Direct L1 的位置 RMS 距离 | Semi-diagonal | VelPro |
|---|---:|---:|
| 200 个推进帧的平均值 | 73.870 mm | 70.651 mm |
| 最大值 | 154.495 mm | 157.823 mm |
| 8 秒时的值 | 105.978 mm | 108.360 mm |

按全程平均位置差，VelPro 略接近独立 Direct L1；按末帧和最大值则 Semi-diagonal 略小，不能据此宣称某方法在本例所有时刻均更接近。Direct 的整体展开程度接近两种渐进方法，但局部形状及面外运动过程有明显差异：Direct 最大面外位移约 0.459 m，而两种渐进方法约为 0.641 m。

## 正确的 Direct L1：本层直接仿真

`L1_direct_local` 在 L1 的 1,189 个顶点、1,600 个三角形和本层弯曲铰链上计算原生 Neo-Hookean 膜能与 DiscreteShellBending 弯曲能。质量、惯性、碰撞和时间积分均使用 L1。本次不读取 L0 轨迹，也不使用 35,000 顶点细网格的 PSQ 能量、梯度或延拓。

仍保持相同的两端固定、初始扰动、材料常数、厚度、风载、接触参数及 1.1 主伸长限制。原生运行库的 `reference` 分支关闭弹性能替换；加载的适配数据只含 L1 自身几何及恒等映射，用于沿用应变限制基础设施。恒等数据的存在不表示运行了跨层 PSQ：初始化日志明确为 `coarse=1189 proxy=1189 triangles=1600 ... replace=false strain_limit=1.1`，原生膜能和弯曲能保留，本层无细网格代理。

当前 `--level 1 --method direct` 默认选择本层弹性能；历史独立 PSQ 对照必须显式指定 `--elastic proxy`。

新结果输出：`L1_direct_local/`、`report/L1_direct_local/audit.json` 和 `report/with_local_direct/`。使用 `.venv/Scripts/python.exe scripts/report_flag_l1.py --local-direct` 生成正确的四栏比较。

正确 Direct 完成 200/200 步并全部收敛，两端固定误差为零，201 个保存状态未检测到 L1 自交，最大主伸长为 1.099594055。相对该 Direct 的全程平均位置 RMS 距离：Semi-diagonal 92.325 mm，VelPro 89.095 mm；末帧分别为 94.126 mm、95.607 mm。这些新指标替代上面历史 PSQ 对照的 Direct 比较数值；此时两者与 Direct 的区别同时包含跨层时间推进和弹性能离散方式，不能归因于单一因素。

## 输出和复现

- `report/comparison.gif`、`comparison.mp4`：L0、Semi-diagonal L1、VelPro L1 的同时间和同尺度动画。
- `report/comparison.json`、`comparison_metrics.csv`：层间位置差和自由边高度。
- `report/L1_semi/audit.json`、`report/L1_velpro/audit.json`：各组审计。
- `L1_semi/`、`L1_velpro/` 内的 `positions.npy`、`injected_velocities.npy`、`steps.json`、`run.json`、`source_snapshot.py`：实际轨迹、注入速度和运行证据。

```powershell
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_L1_new --reuse-assets output/flag_L0_two_pins_wind300_20261122 --seed 0.001 --wind-scale 3 --pin-mode corners --level 1 --method semi --lower output/flag_L0_two_pins_wind300_20261122/L0
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_L1_new --seed 0.001 --wind-scale 3 --pin-mode corners --level 1 --method velpro --lower output/flag_L0_two_pins_wind300_20261122/L0
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_L1_new --seed 0.001 --wind-scale 3 --pin-mode corners --level 1 --method direct --elastic local
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py --root output/flag_L1_new --run-name L1_semi --audit-only
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py --root output/flag_L1_new --run-name L1_velpro --audit-only
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py --root output/flag_L1_new --run-name L1_direct_local --audit-only
```

本次固定目录的三组比较可由 `scripts/report_flag_l1.py` 重新生成。
四组比较使用 `.venv/Scripts/python.exe scripts/report_flag_l1.py --direct`。
