# L0 旗帜：只固定左边的两个端点

结果目录：`output/flag_L0_two_pins_wind150_20261122`。
对照目录：`output/flag_L0_seed1mm_wind150_20261122`。

按用户描述，将整条左边固定改为仅固定两端，未在本次实验中重新核对论文的完整场景参数。固定的是 L0 顶点 0 和 661，坐标分别为 `(0, 0, 0.25)`、`(0, 0, 1.25)` m。原左边其余 138 个顶点释放，没有软位置约束。图中的旗杆沿用之前的视觉标记，没有新增实体杆碰撞体。

简化阶段保留边界顶点与物理上固定边界是两回事：沿用原来的边界保留层级，以便只比较约束变化。L0 仍为 911 个顶点、1,044 个三角形，PSQ 弹性代理为原始 35,000 个细顶点。网格、PSQ 算子和 bundle 哈希与对照完全一致。

其余设置沿用：初始面外扰动约 1 mm、零初始速度、平面材料静止形状、0.04 s 步长、200 步、8 s。风载沿 +x，为自由顶点上的 `f_i = m_i [12 + 7.5 sin(2π × 1.3 t), 0, 0]`。材料、重力、接触参数、求解器配置和运行库相同。

固定点上的显式风载沿用置零规则；释放的 138 个顶点现在与其他自由点一样承受体力，因此自由点总风力比整边固定时增加约 2.26%。并排实验保持的是相同空间体力加速度，不能声称两次自由点合力完全相同。比较脚本独立重算面积集总质量，并验证两组每步合力与其自由点集合一致。

| 指标 | 整边固定 | 两端固定 |
|---|---:|---:|
| 固定顶点数 | 140 | 2 |
| Newton 收敛步数 | 200/200 | 200/200 |
| 原左边中间顶点最大位移 | 0 | 0.158077 m |
| 全程最大面外位移 | 0.435157 m | 0.550697 m |
| 面外位移峰值时刻 | 5.40 s | 3.88 s |
| 最后 2 s 自由边平均高度 | 0.283653 m | 0.288132 m |
| 最大主伸长 | 1.0992977 | 1.0993916 |

两端固定版的两个粗层固定点以及 PSQ 映射后对应的两个细层端点误差均为 0。其余左边顶点的移动是释放约束后的预期行为。全部 201 个保存状态未检测到 L0 自交，未使用未收敛预览帧。

整体下垂程度接近，靠杆侧边可以弯曲、鼓出，局部变形和摆动过程有所变化。这仍是预设体力模型的动力学演示，不是空气流动或气动颤振验证。

## 输出与复现

- `report/comparison.gif` / `comparison.mp4`：同一时刻、相机和空间比例；左侧红线表示整边固定，右侧两个红点表示两端固定。
- `report/flag_L0_parallel.mp4`：两端固定版单独动画。
- `report/audit.json` / `comparison.json` / `metrics.csv`：验证和指标。
- `L0/run.json`：固定索引、坐标、求解器配置和运行库哈希。
- `L0/positions.npy` / `steps.json` / `applied_wind.json`：位置、原生求解统计、逐帧外力。

```powershell
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_L0_two_pins_new --reuse-assets output/flag_L0_seed1mm_wind150_20261122 --seed 0.001 --wind-scale 1.5 --pin-mode corners
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py --root output/flag_L0_two_pins_new
```

本次固定目录的对照动画可用 `.venv/Scripts/python.exe scripts/compare_flag_seed.py --pins` 重新生成。
