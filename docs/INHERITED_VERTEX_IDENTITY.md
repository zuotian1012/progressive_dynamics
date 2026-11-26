# 边展开后的顶点身份与应变修正

更新：2026-11-11。本轮针对此前固定继承边超限、渐进零摩擦对照失败的问题，追踪实际几何并修正顶点身份判定。17→25 顶点的有摩擦、零摩擦对照现均完成；10→25 顶点的强粗化仍存在后续应变修正预算失败。

## 论文依据与解释边界

[PSQ 2023 §4.4.1–4.4.2](https://pcs-sim.github.io/psq/psq-combined.pdf)描述将一个折叠顶点反向展开为两个端点，并分别扰动、推进这两个端点；§4.4.5 固定先前粗层中仍继承的顶点，对新插入顶点进行应变修正。[PCS 2022 §4.3](https://pcs-sim.github.io/pcs-main.pdf)的对应规则是固定细分前的偶顶点、优化新产生的奇顶点。两篇论文的拓扑操作不同，不能仅凭数组索引相同就把两种继承关系等同。

本项目使用的原生边折叠代码，在合并两个顶点时复用其中一个数组槽；反向展开同样复用该槽存放一个子端点。旧实现将 `coarse_ids` 全部固定，把存储槽复用当成了顶点身份继承。

当前按以下身份解释处理：

```text
粗层父顶点 k（占用数组槽 i）
             │ 反向边展开：父顶点退出，两个子端点产生
             ├─ 新端点 i（复用原数组槽）
             └─ 新端点 j（恢复另一数组槽）

只有从未参与拆分、直接存续到目标细层的粗顶点保留继承身份。
```

这是依据论文拓扑描述对本仓库数据结构的解释与修正，**不是声称已取得作者安全展开分支的源码并逐行确认**。访问的 PSQ/PCS 项目页提供论文，没有列出这一分支的源码下载；论文也没有明确给出“复用旧槽时如何生成固定掩码”的代码。旧策略保留作对照，便于复核解释。不能仅因为新规则通过实验就宣称完整论文复现。

## 实现变化

- `ExpansionHistory.unsplit_coarse_ids` 保存未被拆分的粗顶点身份；`coarse_ids` 继续用于初始几何装载和编号对应，不再兼任固定约束集合。
- `ExpansionOptions.fixed_policy` 默认 `unsplit-lineage`；`coarse-indices` 保留旧策略。主入口为 `--expansion-fixed-policy`，实际选择写入实验和逐步初始化日志。
- 应变修正只固定存续粗顶点和障碍物。拆分产生的两个端点均可优化，包括复用旧编号的端点。修正结束逐位检查真正固定的顶点没有移动。
- 固定端点超限的提前失败检查、单调可行面集合、CCD、0.2 mm 厚度和 1.10 应变上限保持生效。
- 垂直断点续算拒绝混用两种固定策略；旧输出缺少该参数时按旧策略解释。水平路径不使用这个参数。

这不是解除物理抓取绑定：衣服的物理步从来没有这类绑定；此掩码只用于一次初始化中的应变修正。

## 同输入对照与失败归因

`scripts/diagnose_inherited_strain.py` 可按保存的粗层位置、实体夹爪姿态和时间步复现初始化，保存每批扰动、比例 CCD、局部松弛、最终应变修正的完整几何。诊断中的中间“继承边长度比”参照目标细网格静止边长；在该细边尚未恢复时，它只是追踪量，不能当成中间网格的材料应变。

| 同输入诊断 | 旧规则 | 新规则 |
| --- | --- | --- |
| 10→25 顶点，第 1 个细层时间步 | 固定 10 个粗层数组槽；4 条固定边超限，最大边长比约 1.493 | 其中 9 个父顶点已被拆分，只保留顶点 24；14 次应变更新后通过，最大主拉伸 1.058189 |
| 17→25 顶点，零摩擦第 2 个细层时间步 | 固定 17 个粗层数组槽；100 次应变更新后仍未通过 | 7 个父顶点已被拆分，保留 10 个存续顶点；1 次应变更新后通过，最大主拉伸 1.003066 |

记录分别为 `output/inherited_strong_stages`、`output/inherited_strong_lineage`、`output/inherited_mu0_lineage`。前两种失败输入的直接延拓目标本身可行；强粗化第一步的局部松弛没有实际更新，说明其固定边问题不能归因于局部松弛把继承顶点推远。

## 完整小型夹爪对照

两组参数、资产、网格摘要完全相同，只改摩擦系数和输出目录。均为 `unsplit-lineage`、两层内在层级、60 步、步长 0.01 s；最小展开间距 0.21 mm，实际接触厚度 0.20 mm。每组包含 VelPro 和 Direct 各两层。

| 结果 | 摩擦系数 1 | 摩擦系数 0 |
| --- | --- | --- |
| VelPro 粗层上部材料平均位移 | +16.636 mm | -1775.874 mm |
| VelPro 细层上部材料平均位移 | +16.642 mm | -1775.775 mm |
| Direct 细层上部材料平均位移 | +16.650 mm | -1724.079 mm |
| 夹爪上提 | 16.667 mm | 16.667 mm |
| 每组物理步收敛 | 4 条轨迹各 60 步全部收敛 | 4 条轨迹各 60 步全部收敛 |
| VelPro 细层安全初始化 | 60 次全部接受 | 60 次全部接受 |

“上部材料”固定为初始 `z>0.06 m` 的顶点，不随运动重新选取。两组的渐进细层最大主拉伸分别为 1.005659 和 1.003723；初始化的最小间距下界分别为约 0.210970 mm 和 0.289199 mm。

本轮终于完成了**渐进分支自身**的零摩擦对照；此前只有 Direct 完整通过。这支持当前小夹爪靠接触摩擦抓取的解释，不是原衬衫全流程的验证。零摩擦下 VelPro 与 Direct 细层位移相差约 51.7 mm，仍需区分时间积分差异与跨层误差，不能写成全部动力学轨迹相同。

输出：`output/lineage_grip_mu1`、`output/lineage_grip_mu0`；审计及动画：`output/lineage_grip_report`。审计核验成功轨迹的物理内核摘要与当前实现相同。

![有摩擦与零摩擦的真实仿真轨迹](../output/lineage_grip_report/grip_controls.gif)

动画为 0.2 倍速；两行竖直尺度不同，下排展示完整下落距离。单次 CPU 测量中，渐进细层有摩擦约 8.58 s、零摩擦约 4.20 s，均包含初始化；不是性能统计或论文加速结论。

## 仍未通过的案例

`output/lineage_grip_strong_mu1`：10→25 顶点，粗层 60 步完成，细层接受前 3 步，**第 4 步**应变修正耗尽 100 次更新，主拉伸仍为 1.528522。这次不再是固定继承边超限证书，而是修正迭代仍未恢复可行性。

`output/lineage_strong_step3_stages` 保存该步诊断。其直接延拓目标满足接触厚度、无自交和应变约束，最大主拉伸约 1.002080；局部松弛也没有实际更新。尚需进一步核对逐批目标位置、碰撞截断产生的偏离，以及应变修正路径。当前没有增加预算掩盖失败，也没有以可行目标直接替代展开流程来宣称强粗化已经通过。

## 真实衣服与自接触复测

真实衣服 `output/lineage_shirt_initialization` 重新执行了全部 3557 次展开、81 批，**没有跳过展开或直接返回可行目标**。最终无自交，间距下界 1 mm，最大主拉伸 1.000679；应变修正 13 次更新。533 个初始粗层存储槽中只有 1 个对应未拆分顶点，其余均通过真实边展开产生新端点。

相对原直接延拓目标，最终 RMS 位移差为 **0.000183 mm**、最大差约 **0.011692 mm**；相比旧固定槽规则的 0.237109 mm RMS 和 7.941785 mm 最大偏差明显减小。新初始化耗时约 237.78 s，仍是 CPU 单次诊断数据。

`output/lineage_shirt_dynamics` 的同一细层增量问题对照，两种初值均在第 1 次 Newton 检查满足既有收敛标准，没有进行位置更新；最终能量差约 `5.35e-10 J`。这里不能把“第 1 次检查即通过”表述成又消除了剩余误差。

同一网格的两片自接触压力案例 `output/lineage_contact_initialization` 也通过，最终间距约 0.200014 mm，最大主拉伸 1.099898。随后 `lineage_contact_dynamics` 中两种初值均收敛，结果 RMS 差约 0.002505 mm、最大差约 0.005701 mm。

## 回归覆盖与重跑

新增独立原生导出的小夹爪历史和失败时间步输入，测试旧规则确实失败、新规则通过、存续顶点与所有障碍物逐位保持不动、复用编号的新端点能够修正。原有自接触、ARAP Hessian、CCD、同问题细层质量、水平/垂直断点测试仍保留；续算另覆盖固定策略不一致时拒绝。

最终 `output/lineage_final_tests.log`：**56 项测试通过**。完整实验索引、初始化/常规动力学内核摘要检查及限制保存在 `output/lineage_final_audit.json`。

```powershell
.venv/Scripts/python.exe scripts/diagnose_inherited_strain.py --experiment output/vertical_grip_strong_infeasible --step 0 --fixed-policy coarse-indices --output output/my_old_identity
.venv/Scripts/python.exe scripts/diagnose_inherited_strain.py --experiment output/vertical_grip_strong_infeasible --step 0 --fixed-policy unsplit-lineage --output output/my_new_identity
.venv/Scripts/python.exe -m progressive.run --scene intrinsic-grip --assets output/intrinsic_grip_mild_assets --levels 2 --steps 60 --methods velpro direct --warm-start vertical --expansion-min-gap .00021 --friction 0 --output output/my_lineage_mu0
.venv/Scripts/python.exe scripts/report_lineage_validation.py --positive output/lineage_grip_mu1 --negative output/lineage_grip_mu0 --output output/my_lineage_report
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

这轮解决的是顶点身份与小型渐进摩擦对照。强粗化鲁棒性、三层以上垂直路径、完整衬衫抓取/折叠和论文性能仍未完成。
