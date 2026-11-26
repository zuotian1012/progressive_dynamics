# Progressive Dynamics：布料系统复现

**当前工作目标（2026-11-11，用户调整）**：先复现 PD++ 的 Cloth Drop Over Bunny（图 11–12），用 Semi-diagonal、Full-diagonal、VelPro 和 Direct 做同条件验证，再回到 Scene527 叠衣服。入口、公开参数、尚缺的作者输入和逐阶段验收见 [布料兔子复现](docs/BUNNY_REPRODUCTION.md)。重建 bunny 已在零摩擦、零层间一致性权重下完成原版 PD 和 VelPro 的 4 秒 L0/L1 对照。复现实验、八层检查和同步视频见上述文档。

**最新验证（2026-11-11）**：[边展开后的顶点身份修正](docs/INHERITED_VERTEX_IDENTITY.md)。56 项测试通过；17→25 顶点渐进有摩擦和零摩擦均完成 60 步；真实衣服 3557 次展开复测通过，静态初始化形状偏差显著减小。[查看匹配的摩擦对照动画](output/lineage_grip_report/grip_controls.gif)。强粗化后续仍有失败，完整衬衫折叠和论文性能尚未复现。

**此前验证（2026-11-10）**：[初始化后的动力学与夹爪短测](docs/REFINEMENT_DYNAMICS.md)。55 项测试通过；同一细层问题的初值差异显著缩小；17→25 顶点实体夹爪有摩擦轨迹完成 60 步，并提供[实际轨迹动画](output/refinement_dynamics_audit/grasp.gif)。渐进零摩擦和部分粗化层级仍有应变修正失败，尚未完整复现。

**2026-11-10 论文算法对齐 `progressive/`**：已补齐 C-IPC 应变屏障归一化与自适应刚度、精确弹性 Hessian、有符号铰链角差、RMS 收敛度量、Additive CCD、当前层预条件器和概率二次误差简化。水平分支的 43 项测试通过；该阶段内核完成三层曲面四种方法对照、533→4090 顶点真实衣服短测及实体夹爪有/无摩擦对照。所有最终实验已核验代码摘要一致。现已继续新增[安全边展开与两层垂直初始化](docs/SAFE_EXPANSION.md)的 CPU 参考实现，该阶段总计 52 项测试通过。**这仍不是全部论文和完整折衣的复现**：更多层级的全局最细代理初始化、任意层级厚度修复、完整新模型抓取与折叠以及 GPU 性能实现仍未完成。详见 [逐项公式、依据与验证](docs/PAPER_ALIGNMENT.md)、[算法框架](docs/PAPER_FRAMEWORK.md) 和 [历史稳定性与抓取诊断](docs/STABILITY_AND_GRASP.md)。

新增 **Scene527 实体夹爪折衣原型**：已在本机 Windows / RTX 3060 Ti 上完成 1,536→6,144 面的第一段折叠、接触摩擦抓取、同步动画和与独立细网格的耗时对照。见 [运行与结果说明](docs/SHIRT_DEMO.md) 和 [同步动画](output/shirt_soft_fold/report/shirt_demo.gif)。当前是 VelPro 思路的原生 IPC 工程原型，不是完整论文或原仓库 73 秒流程的复现。

本项目直接实现三角形薄壳布料，用于对比 Progressive Dynamics (2024) 与 Progressive Dynamics++ (2025) 的时间/分辨率更新方式。

## 早期 `cloth/` 基线已实现

- 三维三角形薄壳：Neo-Hookean 膜能、离散二面角弯曲能，均使用解析梯度。
- 规则三角网格层级与重心线性延拓；各层通过延拓评估最细层弹性能，再将梯度限制回当前层。
- 隐式增量势能求解，质量矩阵采用面积集总，固定顶点消元。
- 共享代理势能的独立逐层积分，以及 Semi-diagonal、Full-diagonal、VelPro。
- 质量加权的跨层位置一致性惩罚，可通过配置调节。
- 无摩擦地面接触：对数屏障和可行高度边界。
- 轨迹保存、求解日志、定量指标、对比图片与 GIF 动画。
- 竖直旗帜、固定旗杆边及沿旗面变化相位的预设风力；固定网格端点的 4/10 层实验。

这是可运行的布料研究基线。当前使用线性延拓及 L-BFGS-B 优化器，尚未实现 PSQ 非线性延拓、论文的 Newton 求解器、一般障碍物接触、自碰撞、摩擦和应变限制。因此还不能把当前结果视为完整论文场景的精确复现。

## 运行

使用 Python 3.11；建议建立项目虚拟环境并安装 `requirements.txt`。本次已用现有 Python 环境运行，未修改全局依赖。

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m cloth --config configs/hanging.json --output output/hanging
python -m cloth --config configs/drop.json --output output/drop --methods velpro
python -m cloth --config configs/hanging_consistent.json --output output/hanging_consistent --methods velpro
python -m cloth --config configs/flag_4levels.json --output output/flag_4levels --no-gif
python -m cloth --config configs/flag_10levels.json --output output/flag_10levels --methods semi full velpro --no-gif
python scripts/compare_flag.py
```

`--no-gif` 可以跳过动画编码。已有轨迹无需重算模拟，可单独重新生成报告：

```powershell
python -m cloth.report output/hanging
python scripts/validate_results.py output/hanging output/drop output/hanging_consistent
```

每个输出目录包括：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 实际使用的场景和求解参数 |
| `environment.json` | 新运行记录的软件版本与求解方式 |
| `trajectories.npz` | 各层全时序位置、网格拓扑及质量 |
| `solver.json` | 每步终止状态、迭代数、梯度及耗时 |
| `metrics.json` | 原始逐帧指标 |
| `summary.json` | 指标峰值、约束误差、求解告警汇总 |
| `comparison.png` / `comparison.gif` | 末帧和动画对比 |
| `metrics.png` | 连续性、一致性、细节和动能曲线 |

## 场景

`hanging.json`：1 m × 1 m 布料，两个相邻角点固定，重力和周期性体加速度驱动。四层分别有 25、49、81、169 个顶点，步长 0.02 s，共 60 步。配置中的 wind 是确定性的周期体加速度，不是空气动力学模型。

`drop.json`：无固定点布料，从地面上方带小倾角落下。测试屏障接触与层间更新。步长 0.02 s，共 50 步。

`hanging_consistent.json`：与悬挂场景相同，增加权重为 50 的一致性惩罚。该权重针对本项目 SI 能量定义，不等同于直接复制论文中的权重取值。

## 数学实现与对照原则

每一步求解

    min_x  (x-xhat)^T M (x-xhat)/(2 h²) + F_l(x) + optional_consistency_penalty

其中 F_l 包含延拓到最细层计算的弹性能、当前层外力势能及地面屏障。膜能密度为

    μ/2 (tr(F^T F)-2) - μ log J + λ/2 (log J)²,
    J = sqrt(det(F^T F)).

μ、λ 已乘厚度；λ 采用平面应力参数化。弯曲能为每条内部边的 `0.5 * kb * weight * (theta-theta_rest)²`，`kb=E*t³/[12*(1-nu²)]`，`weight=3*edge_length²/(adjacent_area_sum)`。

三种渐进积分器按论文在线性延拓假设下构造 xhat。初始虚拟时间步由初速度确定，Full-diagonal 的第一细层单独处理。不同方法在一个比较运行中共享同一条粗层轨迹，采用相同能量、边界条件、求解器和容差。

Independent 对照也使用最细层代理势能，以隔离积分方式的影响；它不是各层分别使用自身离散材料模型的传统直接仿真。

地面是仿射平面。全部顶点位于可行半空间时，三角面及两帧间线性轨迹也不会穿过该平面。这个性质不能推广到一般碰撞物和自碰撞。

## 指标约定

- 连续性：在自由顶点上计算带物理力修正的中点残差。图中显示除以自由部分总质量后的加权 RMS，单位 m；不是原论文未归一化的质量范数数值。
- 一致性：细层位置用延拓矩阵的伪逆投影回粗层，计算质量归一化 RMS。
- 细节：细层位置中不能被粗层线性子空间表示的部分的加权 RMS。大数值可能来自细节，也可能来自伪影，必须结合动画判断。
- `literal_squared_ratio`：按论文印刷残差公式直接计算平方范数比。中点残差带 1/2，而求解残差不带，因此独立积分控制组的基准是 1/4；分母 RMS 小于 1e-10 m 时写为 null，避免制造虚假的大比值。原始分子、分母始终另行保存。
- 固定点不纳入动力学残差，因为它们含约束反力。
- 一致性惩罚只加入渐进细层的求解残差，不加入物理连续性评估的原始势能。
- 求解器返回成功可能由梯度容差或相对能量变化容差触发；同时检查日志中的真实梯度及求解残差，不能把 success 当成所有梯度均达到 gtol。

## 验证与首轮观察

10 项自动测试覆盖膜能/弯曲能/地面屏障/代理能的梯度、刚体运动不变性、内力与力矩平衡、延拓的仿射精确性、积分模板恒速行为、自由落体、固定点和地面接触，以及竖直旗帜和空间变化外力。

2026-11-05 的四层悬挂实验中，三种渐进方法均跑完，未出现优化器失败。细层连续性 RMS 峰值约为 Semi 9.79 mm、Full 6.45 mm、VelPro 2.93 mm；相应粗细一致性 RMS 峰值约为 18.7、24.1、79.9 mm。VelPro 在该配置下改善连续性，但无惩罚时跨层漂移更大，不能宣称所有指标均优于另两种方法。

在同一悬挂配置下给 VelPro 加入权重 50 的一致性惩罚后，一致性 RMS 峰值由 79.9 mm 降至 63.5 mm，细节残差峰值由 19.0 mm 降至 18.2 mm，连续性峰值由 2.93 mm 变为 2.95 mm。这只是一个权重样本，不是完整参数扫描。

三个配置均通过保存结果的物理不变量检查：轨迹有限、固定点不动、地面间隙为正、共享预览一致、求解器无失败，以及独立积分的指标归一化控制。落地算例的最小地面间隙约 11.4 mm，屏障接触保持可行。

这些是本实现、小网格和当前材料下的结果，未复现论文的网格、场景和全部数值细节；四层实验也不足以验证作者关于深层级失稳的全部结论。

## 文件组织

新增的旗帜 4/10 层实验报告见 [docs/FLAG_EXPERIMENT.md](docs/FLAG_EXPERIMENT.md)。已观察到 Semi-diagonal 随层数增加的明显异常放大，但当前缺少论文的 10% 应变限制，不能宣称严格复现原场景。

- `cloth/mesh.py`：网格、质量和延拓。
- `cloth/energy.py`：薄壳与地面能量和解析梯度。
- `cloth/simulation.py`：积分模板、隐式求解、场景运行和日志。
- `cloth/report.py`：定量分析和动画。
- `configs/`：可重复运行的场景。
- `tests/`：数值正确性验证。
- `docs/REPRODUCTION_PLAN.md`：当前布料路线与后续里程碑。

参考：本地 `papers/` 中的两篇论文；官方项目页 https://pcs-sim.github.io/pd/ 和 https://pcs-sim.github.io/pd%2B%2B/ 。
