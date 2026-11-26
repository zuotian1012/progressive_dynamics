# L0 旗帜：平行于初始平面的风载

结果目录：`output/flag_L0_parallel_wind_20261121`。

## 设置

- 原始旗帜宽 1.5 m、高 1 m，位于 xz 平面；左边固定，+x 指向自由端。
- 从原始 35,000 顶点网格实际简化，保留全部边界，得到 L0：911 顶点、1,044 三角形。旗杆侧原始 140 个顶点全部保留并固定。
- 只求解 L0 自由度；35,000 顶点网格用于 PSQ 弹性能计算，并非独立细层动力学。
- 从完全平面、静止状态开始，没有人为施加面外扰动。
- 自由顶点外力为 `f_i(t) = m_i [8 + 5 sin(2π × 1.3 t), 0, 0]`，固定顶点的显式风力为零。无起风缓启动；重力 `[0, 0, -9.81] m/s²`。
- 风载是预设的质量比例体力，并非根据相对气流、法向和迎风面积计算的空气动力学。世界坐标 +x 方向保持不变，不会跟随变形后的局部平面转动。
- E = 30,000 Pa，ν = 0.3，密度 200 kg/m³，全厚度 0.4 mm，主伸长上限 1.1。
- 时间步 0.04 s，200 步，8 s。Newton 速度容差 0.001，最大 1,500 次迭代，线性容差 1e-10；禁止预览提前接受。
- 启用原生自接触和摩擦（μ = 0.3），接触距离 1 mm。求解器将请求的接触刚度 1e9 自动限制为 66,890,778.10893504；初始化日志保留该信息。

## 验证与观察

- 200/200 步按原生 Newton 收敛条件完成，没有使用 non-descent 或 small-step 预览终止。
- 粗网格固定边误差为 0；PSQ 映射后的原始细网格固定边最大误差 4.885e-15 m。
- 全部 201 个保存状态的 L0 网格未检测到自交。此项是保存状态的检查，不是额外的连续时间几何证明。
- 最大主伸长 1.099306519，低于 1.1。
- 面外位移峰值 0.556319 m。第一步已经出现约 9.03 μm 的面外偏移，之后增长。外力和初始条件均未设置面外分量，破坏平面对称性的具体数值来源尚未单独定位，不能把该结果解释为已验证的气动颤振。
- 动画使用实际 L0 顶点位置，位移不放大；提供斜视、正视、风载和面外位移曲线。颜色统一，面部明暗来自实际三角形法向。

## 文件与复现

- `report/flag_L0_parallel.mp4`：25 fps 完整动画。
- `report/flag_L0_parallel.gif`：每隔一个时间步的轻量预览。
- `report/audit.json`、`report/metrics.csv`：检查结果与逐帧指标。
- `L0/positions.npy`、`L0/steps.json`、`L0/applied_wind.json`：真实位置、求解器统计及实际施加载荷。
- `scene.json`、`source_snapshot.py`：场景参数、原始网格哈希和运行代码快照。

准备新目录后执行仿真：

```powershell
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --prepare --output output/flag_parallel_new_run
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_parallel_new_run
```

当前结果的审计和渲染：

```powershell
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py
```

首次启动因辅助脚本间接导入另一套 native 模块而失败，尚未推进任何时间步；已移除不必要导入后重新启动。失败日志和空目录保留，正式结果来自修正后的完整单次仿真。
