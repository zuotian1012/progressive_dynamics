# L0 旗帜：风载提高 50%

结果目录：`output/flag_L0_seed1mm_wind150_20261122`。
对照目录：`output/flag_L0_seed1mm_20261122`。

本次只将预设风载乘以 1.5：自由顶点受力为
`f_i(t) = m_i [12 + 7.5 sin(2π × 1.3 t), 0, 0]`，对应加速度范围 4.5–19.5 m/s²。
原对照为 `m_i [8 + 5 sin(2π × 1.3 t), 0, 0]`，范围 3–13 m/s²。
这是体力强度的比例变化，不是空气速度增加 50%；当前未求解空气动力学。

沿用同一约 1 mm 面外初始扰动、零初始速度、140 个旗杆固定顶点、材料静止形状、质量、重力、材料参数、接触参数和求解设置。实际求解 L0 的 911 个顶点，35,000 个细顶点用于弹性能代理。两次均为 0.04 s 步长、200 步、8 s。

比较脚本验证两次初始位置逐项相等，网格、PSQ 算子和 bundle 的 SHA-256 相同，原生运行库哈希和求解器配置相同；每一帧的加速度及合力均为对照的 1.5 倍，时刻一致。

| 指标 | 原风载 | 风载 ×1.5 |
|---|---:|---:|
| Newton 收敛步数 | 200/200 | 200/200 |
| 全程最大面外位移 | 0.565960 m | 0.435157 m |
| 最大面外位移对应时刻 | 1.68 s | 5.40 s |
| 8 s 时自由边平均 x | 1.128386 m | 1.332528 m |
| 8 s 时自由边平均 z | 0.038862 m | 0.241947 m |
| 最后 2 s 自由边平均 z（51 个保存时刻） | 0.039171 m | 0.283653 m |
| 最大主伸长 | 1.0993001 | 1.0992977 |

增强风载后旗布整体更展开、下垂减轻。最后 2 秒自由边平均高度增加约 24.4 cm，但全程面外位移峰值较小；不能用风力增大直接推断面外摆幅必然增大。

增强版固定边最大误差为 0；PSQ 细网格固定边误差为 4.885e-15 m；全部 201 个保存状态未检测到 L0 自交。这不是连续时间无自交的额外证明，也不是气动颤振或算法普遍鲁棒性的验证。

## 查看与复现

- `report/comparison.gif` / `comparison.mp4`：原风载与增强风载，同一时刻、相机、空间比例。
- `report/flag_L0_parallel.mp4`：增强版单独动画。
- `report/audit.json` / `comparison.json` / `metrics.csv`：检查结果和指标。
- `L0/positions.npy` / `steps.json` / `applied_wind.json`：实际仿真位置、求解统计、逐帧载荷。

```powershell
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_L0_wind150_new --reuse-assets output/flag_L0_seed1mm_20261122 --seed 0.001 --wind-scale 1.5
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py --root output/flag_L0_wind150_new
```

本次固定目录的对照动画可用 `.venv/Scripts/python.exe scripts/compare_flag_seed.py --wind` 重新生成。
