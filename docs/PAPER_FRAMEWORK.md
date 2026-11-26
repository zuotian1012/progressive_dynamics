# 论文算法框架：独立参考求解器

更新：2026-11-10。代码入口为 `progressive/`。这是独立多层薄壳求解器，并将 Scene527 衣服、碰撞几何和夹爪轨迹接入它。最新公式修正与最终验证见 [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md)；本文保留早期实验作为历史记录，不能与新势能模型混用。

**已完成主要算法计算链的参考实现，尚不是全部几何预处理、场景和性能的完整论文复现。** 旧 `shirt_soft_fold` 的完整折叠视频及 GPU 耗时不是本实现的结果。

## 论文与代码的对应关系

| 环节 | 新实现 | 验证及边界 |
| --- | --- | --- |
| 当前层接触 + 共享最细层弹性能 | `solver.py` | 每次迭代计算细层膜能和弯曲能，再限制梯度；不是插值动画 |
| Semi、Full、VelPro | `integrators.py` | 按层、按时间计算；另有各层自身能量的 Direct 对照 |
| PSQ 非线性延拓 | `prolongation.py` | `P(x)=Ux+n(x)γ+R(F(x))t`，包括切向旋转、法向偏移、静止形状恢复 |
| 内在层级预处理 | `native/intrinsic_builder` / `prepare_intrinsic.py` | 接入 Liu 等 2021 作者实现，边折叠与联合自参数化生成 U 和锚定面；原始最细网格保留 |
| 直接到最细层的延拓 | `Hierarchy.to_finest` | 组合 U 后重新计算直接偏移；不把多次非线性 P 的复合冒充直接延拓 |
| 速度传递 | `integrators.py` | 用 U 传递速度，不向速度加入位置偏移 |
| Newton 与线搜索 | `solver.py` | 当前层弹性 Hessian 预条件器，或投影细层 Hessian 对照；半正定投影、Armijo、CCD 与应变可行性 |
| 一致性控制 | `integrators.py` / `solver.py` | 下一时刻粗层目标、质量加权能量、梯度及 Hessian |
| 自接触、障碍物、摩擦 | `contact.py` | IPC Toolkit 屏障、摩擦势能、梯度、Hessian、连续碰撞检测 |
| 应变限制 | `shell.py` | 主拉伸对数屏障及线搜索可行性，不是旧 demo 的 `strain_rate=100` |
| 实体夹爪 | `affine.py` | 每体 12 个仿射自由度，正交性惩罚及姿态驱动；衣服只受接触/摩擦 |
| 定量指标 | `metrics.py` | 内在 U 的伪逆、质量范数、力修正中点残差；近零分母写 null |
| 可行初始化 | `run.py` | 论文允许的水平 warm start：每层从自己的初态及上一帧推进 |

来源：[PD 2024](https://pcs-sim.github.io/pd/)、[PD++ 2025](https://pcs-sim.github.io/pd%2B%2B/)、[PSQ 2023](https://pcs-sim.github.io/psq/)、[内在延拓作者代码](https://github.com/HTDerekLiu/surface_multigrid_code)、[IPC Toolkit](https://github.com/ipc-sim/ipc-toolkit)。本地另存 PSQ 论文 `papers/psq-combined.pdf`。这里实现已有论文的方法，不声称提出了一篇新论文或一个新算法。

## 实际求解的能量

```text
I_l(x) = (x-xhat)^T M_l (x-xhat)/(2 h²)
       + Ψ_L(P_l(x)) + B_l(x) + D_l(x; lagged state) + S_l(x)
       - x^T M_l g
       + w (x-P_lower(x_lower_next))^T M_l (x-P_lower(x_lower_next)).
```

`B,D,S` 都在当前层几何上计算。夹爪另有姿态驱动和正交性势能，接触项将夹爪和衣服一起耦合求解。衣服没有抓取点、固定顶点、缝合或机器人目标约束。夹爪属于有正交性惩罚的近似刚体，其实际变换会保存，不能假定它完全达到指定姿态。

一致性项连接不同分辨率的衣服状态，不是绑定衣服和夹爪。默认 `w=0`；本项目采用上式定义，梯度系数为 `2w`，相对惯性强度为 `2wh²`，不直接照搬论文图中的权重数值。

PSQ 非线性偏移在每次 Newton 迭代重算，求梯度时冻结；默认线搜索在候选状态重新计算偏移，评估完整非线性势能，对应 PSQ Algorithm 1 的 `LineSearch(F_l,...)`。另保留 `--offset-linesearch frozen` 作为局部冻结模型对照。冻结梯度是 PD §4.6 的设计，不是完整非线性 P 的精确导数。当前膜/弯曲 Hessian 已改为精确求导，逐元素半正定投影；弯曲使用 C-IPC 原始有符号角差。应变屏障使用解析谱 Hessian。生产路径不再使用弹性梯度中心差分，旧模式只作对照；尚未采用 GPU 批处理。

摩擦基及法向力每个时间步滞后一次；IPC Toolkit 调用采用 C-IPC 厚度偏移屏障与 Additive CCD。应变屏障已补区间归一化，刚度从 1000 Pa 开始按连续两个已接受迭代的间隙自适应增加至 100000 Pa。当前层预条件器与 RMS 速度增量为默认设置。材料和固定接触刚度等仍是显式场景参数，不声称恢复了作者所有场景配置。

**预览停止不等于 Newton 收敛。** PD 2024 §4.6 允许粗层预览在近似方向不再下降时停止；PSQ Algorithm 1 另有 `alpha < gamma` 的停止条件。当前只对渐进方法的非最细层采用这项政策，并分别记录 `accepted`、`converged`、`preview_stop`。`--preview-min-alpha 1e-8` 是本项目显式阈值，不声称是作者原值；`--strict-preview` 可关闭预览早停。最细层与全部 Direct 层始终要求 Newton 收敛，达到迭代上限不予接受。该区别是算法要求，不能把预览早停计为求解器收敛。

## 早期本机验证记录（以下旧模型实验保留原数值）

最新 43 项测试及最终内核从初态独立运行的验证见 [论文对齐记录](PAPER_ALIGNMENT.md)。下文的 35 项测试和 `pd_*` 实验反映此前阶段，不是新公式的验证证据。

`output/pd_stability_tests.log`：**35 项测试通过**，覆盖非零 PSQ 偏移、冻结代理梯度、非线性势能下降、Hessian、质量加权一致性闭式解、IPC/摩擦梯度、CCD、夹爪梯度、未收敛标记，以及预览早停不能冒充 Newton 收敛。应变 Hessian 包括主拉伸为 1.09999、接近 1.10 上限的数值验证。新增检查还覆盖原始碰撞静止几何、固定线程数、断点续算与连续运行逐位一致、保持阶段的夹爪目标、固定衣料点追踪及接触反力。

`output/pd_framework_audit.json` 逐帧审计完整实验：状态有限、步数一致、主拉伸小于 1.10；最细层和 Direct 每步 Newton 收敛，粗层允许的预览停止单独计数。失败试运行保留但未纳入完整实验审计。

### 三层布料

`output/pd_framework_flag`：9 / 25 / 81 顶点，30 步，步长 0.01 s。渐进方法共享同一代理能粗层轨迹，Direct 用各层自己的弹性能。

| 最细层指标 | Semi | Full | VelPro | Direct |
| --- | ---: | ---: | ---: | ---: |
| 平均粗空间一致性 RMS，mm | 0.495 | 0.723 | 5.132 | 26.256 |
| 最大主拉伸 | 1.08016 | 1.08095 | 1.08926 | 1.09738 |

`output/pd_framework_flag_w500`：仅把 VelPro 的一致性权重改为 500，最细层平均一致性误差从 **5.132 mm 降至 2.498 mm**。这验证控制项在此算例起作用，不证明 VelPro 在所有场景最好，也不代表论文性能结论已复现。

`output/pd_framework_curved`：非平面静止曲面，PSQ 非零偏移标记为 `[true,true,false]`，完成 30 步，避免只在零偏移平面上验证非线性算子。此历史结果使用冻结偏移的线搜索模型。

`output/pd_curved_paper_preview`：完整非线性线搜索、三层、四种方法各 30 步完成；全部最细层与 Direct 收敛。共享最粗层 11 步按论文预览规则停止，第二层 Semi/Full 各 10 步、VelPro 5 步预览停止，均单独记录。对应严格预览试跑 `pd_curved_nonlinear_linesearch` 在第 7 步因近似方向无法继续降低实际势能而失败，保留该记录。当前这些早停较频繁，不能声称粗层求解质量已与作者实验完全一致。

查看 [指标图](../output/pd_framework_flag/report/metrics.png) 和 [动态对照](../output/pd_framework_flag/report/comparison.gif)。

### 实体夹爪小算例

`output/pd_framework_friction_validated`：两个实体盒形夹爪夹持并抬升两层布料，9 / 25 顶点、60 步、0.6 s，VelPro 与 Direct 完成，衣服没有抓取约束。查看 [动画](../output/pd_framework_friction_validated/report/comparison.gif)。

**有/无摩擦匹配对照已通过**：`pd_grip_mu1_k1e4` 与 `pd_grip_mu0_k1e4` 仅摩擦系数不同，使用同一网格、夹爪动作、材料与求解参数。夹爪实际抬升 16.667 mm；有摩擦时粗/细层顶部衣料分别上升 **16.628 / 16.644 mm**；无摩擦时分别滑落 **1.780 / 1.724 m**。每步均收敛。结果见 [对照图](../output/pd_framework_validation/friction_control.png) 与 `friction_control.json`，由 `scripts/validate_friction_grasp.py` 校验。

早先 `pd_framework_grip` 使用接触刚度 `1e8` 时，零摩擦也能被法向接触托持，因此该组不能证明摩擦抓取。上述通过组使用 `1e4`，已改为小算例默认值。保留旧试跑作为校准记录，没有将其当作新验证结果。这个实验验证了求解框架中的摩擦抓取；不替代 Scene527 原夹爪的完整折衣测试。

### 真实 Scene527 衣服接入

**新的内在层级** `output/shirt_intrinsic_2level`：534 / 4090 顶点，1000 / 8000 面，保留原始细衣服拓扑。粗到细的 PSQ 偏移非零，最大为 10.823 mm，静止形状恢复误差低于 `1e-12 m`。两层静止网格均无自交。

**历史非线性算法结果** `output/pd_shirt_paper_linesearch`：该层级、非线性 PSQ、完整非线性线搜索、当前层弹性 Hessian 预条件器及 Scene527 原夹爪碰撞几何/动作，VelPro 和 Direct 均完成两层各 8 步。VelPro 粗层 6 步 Newton 收敛、2 步按预览规则停止；细层全部 8 步收敛；Direct 两层全部收敛。查看 [四组同步动画](../output/pd_shirt_paper_linesearch/report_all_levels/comparison.gif)。

早期 `pd_shirt_intrinsic_smoke` 使用冻结偏移线搜索，两层各 8 步收敛；`pd_shirt_nonlinear_linesearch` 则使用完整非线性线搜索并要求粗层严格收敛，第 7 步出现极小步长停滞并达到 150 次迭代上限。这些历史结果保留，没有与默认算法混用。

这只是 **0.0667 s 的接入验证**，没有覆盖闭爪、抬升或完整折叠。两层总耗时需要相加，不能只报细层时间作为渐进方案总耗时。单次短跑没有统计重复，不能声称论文加速结论已复现。材料采用本参考求解器的 Neo-Hookean 模型，不能和旧原生 Baraff-Witkin 演示直接比较速度。

这次历史短跑的渐进总计算 **101.82 s**（粗层 31.37 s + 细层 70.45 s），直接细层 **79.98 s**，本次没有总耗时优势。`scripts/compare_progressive_timing.py` 检查参数、网格、控制轨迹和物理内核摘要，计入全部渐进层级，并输出模拟时间/计算时间。当前数字来自单次且未隔离其他诊断任务的测量，只说明计时流程已接通；正式速度结论须在机器空闲时重复测量。早期冻结模型的 93.64 / 84.10 s 另存于其原输出目录。

内在预处理不是最近点映射：适配器调用作者 QSlim + successive self-parameterization，输出内在重心矩阵及锚定面。作者代码固定于 `adc49abe10a97200a3a5735bd370e98560091a37`；构建、依赖版本和 GPL 许可见 `native/intrinsic_builder/README.md`。该批历史资产采用普通 QSlim。现已接入概率二次误差度量，新资产为 `shirt_paper_2level`；具体参数和验证见 PAPER_ALIGNMENT.md。三层尝试 `shirt_intrinsic_assets` 的中间网格存在自交，已在 manifest 标记，未用于仿真；因此不声称预处理能自动保证任意层级无自交。

较早的中点层级 `pd_framework_shirt_calibrated`（807 / 3152 顶点）也完成 8 步，但其 PSQ 偏移为零。首次 `1e8` 接触刚度的短跑未收敛，后续衣服改为 `1e5`。扩展到 30 步后，`pd_framework_shirt_contact` 的细层第 14 步因接近应变上限，在 150 次迭代内未收敛，未将失败帧写入轨迹。解析应变 Hessian 的整段重跑 `pd_framework_shirt_spectral` **两层各 30 步全部收敛**，粗/细层计算 833.17 / 1396.01 s，仍只有 0.25 s 模拟时间。旧单步重放 `pd_shirt_spectral_replay` 的 40 次结果存在下述静止几何错误，不能作为仅改变 Hessian 的因果对照；整段主程序重跑不受该重放脚本错误影响。

内在衣服层级的较长冻结模型试跑 `pd_shirt_intrinsic_sequence`：粗层 30 步完成，细层完成 21 步后，第 22 步达到 150 次迭代上限，记录为失败，未纳入完整实验审计。这说明不能由 8 步通过推断完整折衣已经可靠。

**重放工具错误已修复**：旧 `pd_intrinsic_step21_replay` 的 141 次结果将变形后的状态传入 IPC CollisionMesh 作为静止几何，改变了边—边接触的 mollifier 阈值，因而不是同一增量问题，不能用于判断重复性。旧输出保留并撤回其对照效力。

修复后 `pd_step21_fixed_rest_repeats` 在固定 IPC/BLAS 单线程下独立重建并重放三次：输入摘要、最终状态字节及完整迭代轨迹全部一致，均在 150 次上限失败。同一增量问题继续优化 3 次便满足原 `1e-3` 容差（`pd_step21_continuation`）。随后 `pd_intrinsic_sequence_resumed` 从原已接受断点续算，两层各 30 步全部 Newton 收敛，细层最大迭代数 153，审计通过。该段仍是冻结偏移、严格粗层收敛的历史回归；不替代默认非线性算法的抓取验证，继承的计时也不用于性能比较。

真实衣服默认非线性长测 `pd_shirt_grasp_hold` 完成粗层 418/672 步，下一步达到 200 次上限。相同问题重放的前 200 次迭代轨迹完全一致，第 274 次在原容差下收敛，已以 500 次上限续算到 `pd_shirt_grasp_hold_resumed`。该历史实验最终粗层完成 672 步、细层完成 414 步，下一步达到 500 次迭代上限，已经停止。未完成部分不计为成功。稳定性、续算和抓取测量说明见 [STABILITY_AND_GRASP.md](STABILITY_AND_GRASP.md)。

## 运行

全部在当前 Windows 电脑执行：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-progressive.txt
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m progressive.run --scene flag --base-segments 2 --levels 3 --steps 30 --methods semi full velpro direct --output output/my_pd_flag
.venv/Scripts/python.exe -m progressive.report output/my_pd_flag
.venv/Scripts/python.exe -m progressive.run --scene grip --base-segments 2 --levels 2 --steps 60 --methods velpro direct --output output/my_pd_grip
.venv/Scripts/python.exe -m progressive.run --scene shirt --assets output/shirt_paper_2level --levels 2 --frames 3 --settle-steps 2 --methods velpro direct --hessian coarse --max-iterations 200 --output output/my_pd_shirt_smoke
.venv/Scripts/python.exe -m progressive.report output/my_pd_shirt_smoke --all-levels
.venv/Scripts/python.exe scripts/compare_progressive_timing.py output/my_pd_shirt_smoke output/my_pd_shirt_smoke --output output/my_pd_shirt_smoke/timing_comparison.json
```

完整首段折叠入口已接好，但**以下完整运行尚未验证**：

```powershell
.venv/Scripts/python.exe -m progressive.run --scene shirt --assets output/shirt_paper_2level --levels 2 --frames 740 --settle-steps 60 --methods velpro direct --hessian coarse --max-iterations 200 --output output/my_pd_shirt_fold
```

拒绝覆盖已有输出。`experiment.json` 记录参数、版本、网格及控制轨迹摘要；新增运行还记录物理模块代码摘要和资产 manifest。每层保存位置、实际夹爪姿态、逐迭代日志、指标及计时。最细层/Direct 未收敛或任意层达到迭代上限会立即停止，标记未完成；粗层仅允许上述显式预览停止。这些短跑用于诊断，没有统计重复，不用于声称加速。

## 尚未完成的论文工程部分

1. **论文完整的层级质量和碰撞安全处理**：已接入作者内在映射和概率二次误差简化；预处理检测并报告自交和厚度可行性，但没有一般性的碰撞安全折叠/修复。外部映射接口为 `map_i_j.npz`，锚定面索引为 `anchors_i_j.npy`。
2. **PSQ 安全边展开**：现已新增含随机扰动、依赖分组、逐对 CCD、代理松弛及应变修正的 CPU 参考分支，见 [SAFE_EXPANSION.md](SAFE_EXPANSION.md)。当前动力学垂直入口限定两层；多层全局最细代理、作者并行性能和全面压力测试仍待验证。默认水平 warm start 保留。
3. **完整新算法折衣与速度对照**：必须重新验证闭爪、抬升、折叠，并在相同材料、步长、接触参数和容差下与 Direct 比较；旧 GPU 视频不能补足这一步。
4. **性能实现**：当前是 CPU + SciPy + IPC Toolkit。移入原生 CUDA 求解器的能量、梯度和 Hessian 接口仍待实现。

准确结论：已从“借用 VelPro 的 demo”推进到“有独立能量与求解器的 PD / PD++ 算法参考框架”，并接入真实资产；尚不能说全部论文和完整折衣实验都已补齐。
