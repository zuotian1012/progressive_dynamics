# 安全边展开与垂直初始化

2026-11-11 更新：[继承顶点身份](INHERITED_VERTEX_IDENTITY.md)已改为区分未拆分粗顶点与复用旧槽的新端点；旧策略可显式重放。该文提供最新渐进摩擦对照与尚存失败，本页下方保留此前快照。

更新：2026-11-10。本页记录本机 CPU 参考实现的范围和验证，不把初始化通过等同于完整叠衣或性能复现。

**后续验证**：已补充同一细层问题的初值质量对照，以及 17→25 顶点实体夹爪 60 步有摩擦测试，总计 55 项测试通过。也发现了继承顶点固定后应变修正不可行的输入，渐进零摩擦对照尚未完成。详见 [REFINEMENT_DYNAMICS.md](REFINEMENT_DYNAMICS.md)。本页下方的 52 项测试及旧初始化数字保留为前一阶段快照。

## 论文与实现

依据 [PSQ 2023 §4.4、附录 B Algorithms 2–7](https://pcs-sim.github.io/psq/psq-combined.pdf)，应变修正另依据 [PCS 2022 §4.3、Algorithm 1](https://pcs-sim.github.io/pcs-main.pdf)。本地原文为 `tmp/pdfs/psq.txt` 和 `tmp/pdfs/pcs-main.txt`。

| 论文步骤 | 文件 | 实现 |
| --- | --- | --- |
| 反向边展开所需数据 | `native/intrinsic_builder/expansion_history.h` | 导出每次成功折叠的顶点身份、前后有向面、静止位置及联合 UV；不使用最近点匹配 |
| Algorithms 6–7 分组 | `progressive/expansion_history.py` | 保持重叠补丁的依赖次序，贪心选取顶点集合互不相交的补丁；这一独立性条件比仅排除邻接展开更保守 |
| Algorithm 2 拓扑更新 | 同上 | 在原始细层顶点编号中恢复拓扑，检查最终有向面、静止位置和所有局部 PSQ 静止重构 |
| Algorithm 5 安全扰动 | `progressive/refinement_contact.py` | 先沿目标方向分开两个端点；失败后固定第一端点方向，对第二端点做有种子的球面分层采样；失败预算耗尽则拒绝 |
| Algorithm 4 逐对 CCD | 同上 | 近距离对保留比例 0.999，其余 0.1；IPC Toolkit 的 `conservative_rescaling` 必须传 `1-r` |
| Algorithm 3 继续展开 | 同上 | 前进至允许步长，冻结限制该步长的碰撞对顶点，其余顶点继续前进；CPU 路径重新构建候选集 |
| §4.4.3 局部松弛 | `progressive/refinement_relaxation.py` | 仅在当前颜色补丁范围内优化，逐阶段提高接触屏障刚度，直到恢复间距和非退化性 |
| 松弛中的代理能 | `ExpansionHistory.dynamic_prolongation` | 经剩余折叠的联合 UV 逐次追踪最细顶点，构建动态中间网格到目标细层的内在 U、锚定面和非线性 PSQ 偏移；弹性能在目标细层计算 |
| §4.4.5 应变修正 | `strain_relaxation` | 暂时固定继承顶点；维护只增不减的可行面集合，使用惯性目标项、接触屏障、可行面的应变屏障及不可行面的 ARAP 项 |
| 接入动力学 | `progressive/run.py --warm-start vertical` | 只替换渐进细层 Newton 的初始猜测，保持原有积分预测、惯性历史、摩擦滞后状态和实体夹爪接触；Direct 仍独立求解 |

ARAP 项为 `kappa_a * sum_f sum_i (sigma_fi - 1)^2`；实现了解析梯度与精确谱 Hessian，并对局部 Hessian 做半正定投影。可行面在线搜索中不能重新违反应变限制。这里对继承顶点的冻结只发生在初始化的应变修正阶段，不会向衣服添加永久抓取约束。

拓扑插入后新顶点重合，不能直接对它运行 CCD，也通常还不满足最终厚度。实现先在小扰动后的非退化网格上用零厚度 IPC 完成几何展开和局部松弛，之后以实际 `dmin` 再检查，并进行应变修正；只有全部通过才交给动力学求解器。

## 明确记录的参数与实现边界

- 初始安全间距取碰撞距离的保守下界，默认上限 1 mm；没有近距离候选时返回的是下界，不是无限距离。
- 后续增加独立的 `minimum_gap`（附录 Algorithm 3 的 epsilon）参数；不指定时仍逐批恢复原间距。显式参数必须严格大于物理厚度，相关选择与失败证据见上述后续验证。
- 扰动幅度为间距的 0.01 倍；默认随机种子 527，最多额外采样 2048 次。论文的 `d0 << g` 在本实现中明确取 `d0 < 0.1*g`，不是声称恢复了作者未公开的配置。
- 松弛每阶段最多 100 次 Newton 检查，接触刚度最多经历 7 个阶段、每阶段乘 10；ARAP 刚度默认 1000。达到预算但仍不可行必须报错，不能作为预览成功。
- 分组内几何可以同时前进，但碰撞对和线性代数由 CPU 参考路径处理；不是论文 GPU/并行性能实现。
- **当前动力学入口限定两层垂直初始化**，这样本次展开的目标细层就是全局最细层。独立展开模块可处理任意一个已导出层级对，但三个及以上动力学层级还需要接入跨多个层级的全局最细代理；入口会拒绝，而不是静默改用不同能量。水平初始化仍支持多层。
- 采样不保证任意几何都存在安全方向，预算失败时保存诊断。也不自动修复静止网格厚度不足的问题。
- `replay_progressive_step.py` 当前只重建水平初态，明确拒绝垂直运行，避免误把不同初始猜测当作同问题重放。垂直主程序的正常断点续算已经验证。

## 薄三角形步长问题及修复

真实衣服第一批展开曾在局部松弛中停滞。已保存 `shirt_safe_expansion_rest` 与 `shirt_safe_expansion_diagnostic`，没有覆盖。诊断显示：接触 CCD 允许完整步长，而原通用 Weyl 范数界给出约 `0.000767` 的步长；限制三角形在该方向上实际保持非退化。这是过于保守的充分条件造成的停滞。

为局部初始化路径新增了整段三角形面积检查：线性顶点轨迹的叉积为二次多项式，面积平方为四次多项式；检查端点及导数的实根，发现接近退化的区间时二分并回退。它允许薄三角形安全展开，同时能截断“两个端点都不退化、但中间发生塌缩”的路径。数值根检查是本项目的稳健性措施，不冒充作者源码或严格区间算术证明。常规动力学求解器保持此前实现。

## 验证记录

- 单元与回归测试涵盖原有求解器，以及导出历史、分组独立性、朝向、动态内在映射、随机回退、采样预算、逐对距离比例、限制碰撞对冻结、薄三角形内部塌缩、ARAP Hessian、应变集合单调性和继承顶点保持。
- 动态内在映射在初始层与原生作者查询输出逐项比对；每批恢复后的 PSQ 静止重构误差小于 `1e-12 m`。
- 垂直运行的断点恢复与连续运行状态逐位一致；折叠历史摘要不一致时拒绝续算。
- 小型自接触案例将两片衣料放入**同一个**可变形碰撞网格。直接延拓发生自交；安全展开经过碰撞截断、松弛和应变修正后通过厚度与无自交检查。
- 真实衣服资产 `output/shirt_expansion_2level` 为 533→4090 顶点，导出 3557 次折叠、81 批，最大批次 110 次展开；前后有向拓扑与静止几何重构检查通过。

`shirt_safe_expansion_area` 是**切换到共享细层松弛前**的历史验证：81 批完成，最大主拉伸 1.096715，间距下界 1 mm；相对直接延拓的位移 RMS 为 0.1953 mm、最大 5.8716 mm。它证明了面积步长修正的作用，但不替代最终共享细层版本的验证。

`shirt_vertical_two_steps` 是此前局部自身弹性能版本的原夹爪诊断，已主动停止并保留 `INTERRUPTED.json`；仅有粗层两步被接受，不能作为细层抓取或最终垂直算法通过的证据。

最终共享细层版本的实验和代码摘要汇总在 [`output/safe_expansion_final_audit.json`](../output/safe_expansion_final_audit.json)：

| 最终验证 | 结果 |
| --- | --- |
| `vertical_proxy_final_tests.log` | 52 项测试通过，包含新增的 9 项展开/续算测试 |
| `shirt_safe_expansion_proxy` | 3557 次展开、81 批全部完成，无自交，间距下界 1 mm，最大主拉伸 1.093415；53 次应变修正更新 |
| `shirt_dynamic_map_check.json` | 真实衣服动态 U 与原生作者查询的最大权重差 `1.54e-13`，静止重构误差 `7.12e-20 m` |
| `safe_expansion_proxy_contact_final` | 直接延拓自交，安全初始化无自交；最终最小间距约 0.200117 mm，大于要求的 0.2 mm，最大主拉伸 1.098383 |
| `vertical_proxy_integration_final` | 两层小网格，Semi / Full / VelPro / Direct 各 4 步完成，所有物理步收敛；渐进细层逐步记录安全初始化 |

真实衣服初始化计算约 245.68 s。相对直接延拓，其位移 RMS 为 0.2371 mm、最大 7.9418 mm。自接触压力案例的位移 RMS 为 18.416 mm、最大 69.170 mm，存在明显局部偏离：这里验证的是可行初始化，不是细层平衡质量或精确形状保持。后续物理求解的收敛与质量仍须另外验证。图中展示完整高度范围，没有裁掉偏离最大的顶点。

审计确认所有最终初始化的物理模块摘要与当前实现一致；真实衣服静态验证启动后，`run.py` 增加了两层入口限制和中途停止处理，这一非初始化模块差异也已记录。最终小网格动力学实验的全部模块摘要与当前版本一致。所有初始化耗时都应计入渐进方法总耗时，不能从性能统计中排除；以上单次结果不用于加速结论。

## 复现入口

```powershell
# 先按 native/intrinsic_builder/README.md 重新构建带历史导出的原生预处理器。
.venv/Scripts/python.exe -m progressive.prepare_intrinsic --targets 1000 --output output/my_shirt_expansion_assets
.venv/Scripts/python.exe scripts/validate_safe_expansion.py --assets output/my_shirt_expansion_assets --case rest --output output/my_shirt_expansion_check

# 小算例动力学接入；资产包含 expansion_0_1.json。
.venv/Scripts/python.exe -m progressive.run --scene intrinsic --assets output/expansion_fixture_assets --levels 2 --steps 4 --methods semi full velpro direct --warm-start vertical --output output/my_vertical_check

.venv/Scripts/python.exe -m unittest discover -s tests -v
```

所有输出要求新目录。真实夹爪场景可选择 `--scene shirt --warm-start vertical`，但完整动态抓取、折叠、更多层级和统计性能仍须分别验收。
