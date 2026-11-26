# 后续应用：论文算法用于 Scene527 叠衣服

2026-11-11 更新：用户决定先验证论文中的布料兔子场景，见 [当前复现主线](BUNNY_REPRODUCTION.md)。原折衣长实验已通过 STOP 标记停在接受检查点：粗层 1045/1540 步，细层未开始；监测已暂停，旧结果保留。下文保留原应用计划，不作为当前复现验收依据。

2026-11-11：用户明确要求复现原论文算法，并应用到现有叠衣服 demo。

主线采用 PD++ 的 VelPro，配套 PD 的共享最细层弹性能、PSQ 非线性位置延拓与内在速度映射，以及 C-IPC 接触、摩擦、应变限制和 Newton 求解。Semi-diagonal / Full-diagonal 保留作算法对照。实际入口是 `progressive.run --scene shirt`。

## 当前路径

PD 2024 §4.6 和 Algorithm 1 第 12 行允许水平或垂直 warm start。正文还说明，除展示垂直与混合初始化的旗帜例子外，其余动画采用逐层水平 warm start。因此先用每层自身上一帧初始化 Newton，是论文内的有效应用路径；不把每个时间步的安全边展开当成折衣所必需的前置条件。

采用 `output/shirt_paper_2level`：533→4090 顶点，1000→8000 三角面，保留原始最细衣服拓扑，具有非零 PSQ 偏移。使用原 Scene527 双夹爪碰撞几何、桌面和关节控制轨迹。衣服通过实体接触和摩擦运动。

第一里程碑是完整第一折：60 个沉降步，740 个原控制帧，每帧 2 子步，步长 1/120 s，每层 1540 步。之后再扩展后续折叠、扫平和下压阶段。完整第一折也不等于原 4373 帧全流程。

## 已启动的真实场景运行

```powershell
.venv/Scripts/python.exe -m progressive.run --scene shirt --assets output/shirt_paper_2level --levels 2 --frames 740 --settle-steps 60 --methods velpro --warm-start horizontal --max-iterations 200 --threads 1 --checkpoint-every 10 --output output/pdpp_shirt_first_fold
```

这是从初态启动的当前模型运行，没有继承旧模型轨迹。状态以输出目录中的 `experiment.json` 和各层 `run.json` 为准；本文不是完成证明。`completed_steps` 之外的预分配数组内容不能作为仿真结果。

求解器按层运行：先算完整粗层预览，再算细层。遇到未接受的求解步会停止并保存错误；不会把预算用尽当成收敛。可在输出目录创建 `STOP` 文件，使进程在下一时间步边界保存并停止。续算应写入新的输出目录，并核验材料、控制、网格和内核摘要。

初期真实场景剖析保存在 `output/pdpp_shirt_step10_profile`。该输入含 16629 个碰撞顶点、33472 个碰撞三角形。五次 Newton 迭代剖析主要耗时在接触集合构建和 CCD。这是局部诊断，且与主运行同时执行，不用于性能比值结论。

## 验收顺序

1. **算法与数值正确性**：逐项公式有论文依据，现有回归通过。粗层预览停止与 Newton 收敛分开记录；最细层必须收敛。新失败先用同一增量问题回放定位。
2. **真实抓取**：粗细层均完成闭爪、抬升，跟踪闭爪时选定的同一衣料材料点与实际夹爪姿态，检查滑移和接触反力；匹配的零摩擦对照验证抓取机制。
3. **真实第一折**：粗细层从初态连续完成转移、释放、落下。检查保存轨迹、自交、厚度和应变，输出可视化。包围盒变小不能单独作为折叠成功的证据。
4. **对照与一致性**：在相同材料、步长、接触参数和容差下运行 Direct 最细层，比较时间连续性、跨层一致性与可见褶皱。必要的一致性权重按 PD++ §8.2.3 记录并做对照。
5. **后续全流程与性能**：第一折通过后扩展原动作序列。性能测量隔离运行并重复，分别报告粗层预览、已有预览后的细化、从零开始总成本。CPU 结果不替代 GPU 性能复现。

`cloth/` 的早期实验、`shirt_demo/` 的原生 GPU 视频、安全展开的小布片通过结果都可作为参考，但不能替代上述真实衣服验收。

## 依据

- PD 2024：https://pcs-sim.github.io/pd/ ，本地提取文本 `tmp/pdfs/paper2.txt`，§4.6 / Algorithm 1。
- PD++ 2025：https://pcs-sim.github.io/pd%2B%2B/ ，本地提取文本 `tmp/pdfs/pdpp-readable.txt`，VelPro 与 §8.2.3。
- 当前公式对应与边界：`docs/PAPER_ALIGNMENT.md`。
- 安全展开的独立后续问题：`docs/INHERITED_VERTEX_IDENTITY.md`。强粗化修正、多层垂直初始化仍待完成，但不阻塞上述水平路径的应用验证。
