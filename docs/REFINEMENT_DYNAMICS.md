# 安全初始化后的动力学与实体夹爪短测

2026-11-10。本轮补上安全展开之后的求解质量检查，并接入小型实体夹爪。**这些是局部验证，仍不是原衣服完整折叠、任意粗化层级或论文加速性能的复现。**

**2026-11-11 后续**：[继承顶点身份修正](INHERITED_VERTEX_IDENTITY.md)区分存储槽复用与拓扑身份，17→25 顶点渐进有/无摩擦均完成 60 步。下文的失败、55 项测试与数值保留为前一阶段记录；强粗化仍有后续失败，不作完整复现结论。

## 同一细层问题、不同求解初值

入口：`scripts/validate_refinement_dynamics.py`。两条路径使用完全相同的细网格、上一帧、零上一帧速度、0.01 s 步长、零重力、接触与摩擦状态。分别从可行的上一帧位置、安全展开位置启动；没有衣服固定点。自交的直接延拓位置不作为求解初值。两个求解器分别建立材料状态，避免自适应刚度串用；最终能量还在相同刚度下重新计算。

这里比较的是同一个增量问题的初值敏感性，不是跨时间积分方法的轨迹比较，也不是把静态初值实验冒充动态叠衣。

| 案例 | 初值 RMS 差异 | 收敛结果 RMS 差异 | 收敛结果最大差异 | 水平 / 安全初值 Newton 检查次数 |
| --- | --- | --- | --- | --- |
| 同一衣料网格的两片自接触 | 19.1635 mm | 0.002868 mm | 0.010422 mm | 2 / 14 |
| 4090 顶点真实衣服静态输入 | 0.237109 mm | 0.009297 mm | 0.384535 mm | 1 / 8 |

两例都达到既有 `1e-3 m/s` RMS Newton 收敛标准，最终无自交且满足 0.2 mm 厚度。最终自适应刚度都为 1000 Pa，能量差分别为约 `-1.19e-9 J` 和 `5.85e-7 J`。这表明本轮输入中的初始化扰动在物理求解后显著减小；不是所有输入都收敛至相同局部极小值的证明。真实衣服最大差异仍有 0.385 mm，不能写成完全相同。

原始记录：`output/refinement_contact_dynamics`、`output/refinement_shirt_dynamics`。它们引用此前保存的初始化状态及摘要；本轮后续的最小间距选项和失败预检查不改变这里的常规细层动力学内核。

![同一细层问题的初值对照](../output/refinement_contact_dynamics/comparison.png)

## 对齐 PSQ 的两个间距量

[PSQ 2023 附录 B Algorithm 3](https://pcs-sim.github.io/psq/psq-combined.pdf) 第 3、6、10 行使用测得的 `g0` 进行扰动及比例 CCD；第 17 行另用最小间距 `epsilon` 判断是否需要局部松弛。正文 §4.4.3 则以恢复 gap 描述这一过程。此前实现将两者合并，要求每批恢复到其原始间距，作为一种更严格的策略。

现在 `ExpansionOptions.minimum_gap` / `--expansion-min-gap` 显式提供附录中的最小间距参数；不指定时仍使用此前逐批恢复 `g0` 的策略。每批记录 `gap_before` 与 `required_gap`。该阈值必须严格大于物理厚度 `dmin`，而最终初始化仍单独检查实际厚度、自交和应变。

本轮夹爪取 `epsilon=0.21 mm`、`dmin=0.20 mm`；这是明确记录的本地数值选择，不声称是作者公开参数。它只控制几何初始化的松弛触发和结束条件，不改变物理步的接触模型。此修改解决了部分狭窄夹爪案例的间距恢复失败，但不能解决下面的应变可行性问题。

## 实体夹爪接入结果

新增 `--scene intrinsic-grip`，读取原生自参数化与折叠历史资产，复用原实体夹爪的关闭/上提轨迹。衣服无绑定、无固定点；每个细层渐进物理步都经过安全展开。仿真 60 步、共 0.6 s；夹爪上提约 16.667 mm。

| 完整实验 | 层级 | 上部材料平均竖直位移 | 结果 |
| --- | --- | --- | --- |
| VelPro，摩擦系数 1 | 17→25 顶点 | 粗 16.636 mm，细 16.647 mm | 两层共 120 个物理步收敛；细层 60 次初始化全部接受 |
| Direct，摩擦系数 1 | 同样的两层 | 粗 16.637 mm，细 16.650 mm | 全部收敛 |
| Direct，摩擦系数 0 | 同样的两层 | 粗 -1776.165 mm，细 -1724.079 mm | 全部收敛；衣料下落 |

上部材料固定定义为初始 `z > 0.06 m` 的顶点，不每帧重新选点。渐进细层的最大主拉伸为 1.005279，低于 1.10。Direct 有/无摩擦对照验证这个小夹爪模型依赖摩擦；**渐进垂直分支的零摩擦对照未完成，不能将 Direct 的通过替代它。**

最终成功输出为 `output/vertical_grip_mild_mu1_final`，Direct 零摩擦为 `output/intrinsic_grip_direct_mu0`。汇总和内核摘要核验见 `output/refinement_dynamics_audit/audit.json`。

![粗细网格与独立细层实际轨迹](../output/refinement_dynamics_audit/grasp.gif)

这组 CPU 单次测量中，渐进粗层约 0.794 s，细层约 19.084 s（包含约 4.568 s 安全初始化），Direct 细层约 0.868 s。**当前这条小网格逐步垂直初始化路径更慢，不能据此声称加速。**运行中还并发进行了小型验证，数字只作诊断；后续性能实验必须隔离运行并统计重复测量。

## 失败证据与尚未解决的条件

所有失败目录保留，不把未接受帧写成成功轨迹。

- `vertical_grip_mu1`：旧的逐批恢复原间距策略在细层第一步局部松弛失败。
- `vertical_grip_strain_diagnostic` / `vertical_grip_strong_infeasible`：10→25 顶点强粗化在应变修正前已有固定继承边违反拉伸上限。部分固定端点边长比约为 1.493，大于 1.10。只优化新顶点不能修复两个固定端点间的距离。这是当前层级/展开状态与后续固定策略不相容的证据，不能用增加迭代次数解决，也不能据此认定原论文一定有错。
- `vertical_grip_mild_mu0_final`：17→25 顶点零摩擦的细层第一步接受，第二步应变修正达到 100 次预算，最大主拉伸仍约 1.127494。这里没有固定边超限证书，不能把它与上一项断言成完全相同的原因。
- `vertical_grip_light_mu0`：更轻的粗化也出现固定继承边超限，说明仅减少折叠数量不保证可行。

新增固定端点边长证书检查：主拉伸上限必然约束每条材料边的长度比；若一条边两端都固定且已超限，立即报告不可行，保存几何与端点索引。这个检查是充分的失败证据，不是所有不可行情形的完整判定。其余迭代预算失败保存主拉伸、逐迭代记录与失败几何。

下一项算法工作应检查**任意边折叠产生的继承顶点身份/静止位置、局部松弛后的继承位置，以及 PCS 固定旧点应变修正的可行性条件**。需要重新对照 PSQ/PCS 论文及可用源码，区分映射实现问题和论文在当前层级构造下的适用条件。不能为通过实验直接解除固定策略后仍声称严格复现。

## 验证与重跑

`output/refinement_dynamics_tests.log`：55 项测试通过，新增覆盖最小间距与物理厚度的分离、同一细层问题的最终结果对照，以及固定边不可行证书。已有水平/垂直断点续算测试仍通过。

```powershell
.venv/Scripts/python.exe scripts/validate_refinement_dynamics.py --source output/safe_expansion_proxy_contact_final --output output/my_contact_dynamics
.venv/Scripts/python.exe scripts/validate_refinement_dynamics.py --source output/shirt_safe_expansion_proxy --output output/my_shirt_dynamics
.venv/Scripts/python.exe -m progressive.run --scene intrinsic-grip --assets output/intrinsic_grip_mild_assets --levels 2 --steps 60 --methods velpro direct --warm-start vertical --expansion-min-gap .00021 --friction 1 --output output/my_vertical_grip
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

资产是本项目生成的 8 cm 方形布片，经原生预处理导出内在映射和展开历史；其原始输入为 `output/intrinsic_grip_input.obj`。它不是原衬衫资产。3 层及以上垂直路径、完整衬衫抓取/折叠和论文性能仍未验收。
