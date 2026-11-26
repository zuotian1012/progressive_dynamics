# 两端固定旗帜：继续增大风载

本次目录：`output/flag_L0_two_pins_wind300_20261122`。
对照目录：`output/flag_L0_two_pins_wind150_20261122`。

为观察更展开的旗帜，将上次风载再乘以 2，即最初风载的 3 倍：

`f_i(t) = m_i [24 + 15 sin(2π × 1.3 t), 0, 0]`

自由顶点的外力加速度范围为 9–39 m/s²，始终沿世界坐标 +x。此倍率表示预设体力强度，不表示空气速度倍率。

固定条件仍为旗杆侧上下两个角点（顶点 0、661），中间的左边顶点可自由移动。保留相同约 1 mm 初始面外扰动、零初始速度、材料静止形状、质量、重力、材料和接触参数。步长 0.04 s，总时长 8 s，共 200 步。只仿真 L0，911 个动态顶点，35,000 个细顶点用于 PSQ 弹性代理。

比较脚本核验两次初始位置相同，网格、算子、bundle 的哈希相同，运行库及求解器配置相同，每一帧的风载加速度和合力均为原来的 2 倍。渲染不放大位移，使用相同相机和空间比例。

输出包含 `report/comparison.gif`、`report/comparison.mp4`、`report/audit.json`、`report/comparison.json`。展开程度用自由边平均 x 坐标和平均高度描述；平均高度相对于初始值 0.75 m 的降低量可反映下垂。面外位移另行统计，不将其直接作为展开程度。

| 指标 | 上一轮风载 | 再加倍 |
|---|---:|---:|
| 最后 2 秒自由边平均 x | 1.350679 m | 1.492876 m |
| 最后 2 秒自由边平均 z | 0.288132 m | 0.519539 m |
| 最后 2 秒平均下垂量（0.75 − z） | 0.461868 m | 0.230461 m |
| 8 秒时自由边平均 x | 1.329987 m | 1.539516 m |
| 8 秒时自由边平均 z | 0.239907 m | 0.590251 m |
| 最大主伸长 | 1.0993916 | 1.0995873 |

最后 2 秒统计使用 6–8 秒共 51 个保存时刻。平均下垂量减少约 50%，向 +x 的伸展更充分；旗帜仍随周期风载起伏，并未始终保持完全水平。

200 步全部按原生 Newton 收敛条件完成，没有接受未收敛预览帧。两个粗层固定点和对应 PSQ 细层端点误差均为 0，201 个保存状态未检测到 L0 自交；此项检查不等于额外的连续时间无自交证明。

复现（输出应使用新的目录）：

```powershell
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_L0_wind300_new --reuse-assets output/flag_L0_two_pins_wind150_20261122 --seed 0.001 --wind-scale 3 --pin-mode corners
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py --root output/flag_L0_wind300_new
.venv/Scripts/python.exe scripts/compare_flag_seed.py --wind --baseline-root output/flag_L0_two_pins_wind150_20261122 --variant-root output/flag_L0_wind300_new --metric free_edge_height
```

旗杆仍仅为视觉标记，未加入实体杆碰撞；风仍采用预设体力，没有求解空气流动。
