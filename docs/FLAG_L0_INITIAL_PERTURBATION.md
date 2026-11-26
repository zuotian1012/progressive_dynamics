# L0 旗帜：微小初始面外扰动

结果目录：`output/flag_L0_seed1mm_20261122`。对照目录：`output/flag_L0_parallel_wind_20261121`。

只改变初始位置，在原有 xz 平面旗帜上施加：

`y0 = 0.001 (x / 1.5)^2 sin(2π (z - 0.25) / 1.0)`，坐标和位移单位均为米。

这是一段沿高度平滑变化、离旗杆越远幅度越大的波形。旗杆 x=0 处位移为零，解析包络最大幅度为 1 mm；网格采样实际最大值为 0.999936 mm。没有额外面外风力，初始速度为零。

原生求解器先以平面网格初始化材料，再通过状态接口写入初始位置与零速度，并回读核验。因此改变的是初始变形状态，材料静止形状、质量、PSQ 映射和弹性参数均保持不变。扰动也带来微小的初始弹性能，并非无应力的曲面材料。

复用了对照组的层级资产。比较脚本核对 L0 网格、PSQ 算子和二进制 bundle 的 SHA-256 完全一致，同时验证求解器配置、运行库哈希和每步施加的风力完全相同。两次均为 911 个动态顶点、1,044 个三角形、35,000 个弹性代理顶点、0.04 秒步长和 8 秒时长。

| 指标 | 平面初始状态 | 1 mm 扰动初始状态 |
|---|---:|---:|
| 收敛步数 | 200/200 | 200/200 |
| 最大面外位移 | 0.556319 m | 0.565960 m |
| 峰值时刻 | 1.64 s | 1.68 s |
| 8 s 时最大面外位移 | 0.287161 m | 0.288467 m |
| 最大主伸长 | 1.0993065 | 1.0993001 |
| L0 固定边最大误差 | 0 | 0 |
| PSQ 细网格固定边最大误差 | 4.885e-15 m | 4.885e-15 m |
| 保存帧 L0 自交 | 未检测到 | 未检测到 |

末帧两组对应顶点的位置差 RMS 为 2.996 mm。整体下垂和摆动幅度接近，局部形状和峰值时间略有变化。这一次对照表明该具体初始扰动没有大幅改变整体响应；单个扰动案例不足以证明普遍鲁棒性，也没有定位原平面实验中对称性被打破的数值来源。

风载仍是沿世界坐标 +x 的质量比例体力，强度为 `8 + 5 sin(2π × 1.3 t)` m/s²，重力沿 -z。此实验没有求解空气流动，不能据此认定为真实气动颤振。

## 查看与复现

- `report/comparison.gif`、`report/comparison.mp4`：两组同一时刻、同一相机和位移比例的比较。
- `report/flag_L0_parallel.mp4`：扰动版单独动画。
- `report/audit.json`、`report/comparison.json`、`report/metrics.csv`：验证和指标。
- `L0/positions.npy`、`L0/steps.json`、`L0/applied_wind.json`：位置、原生求解器统计和实际载荷。

```powershell
.venv/Scripts/python.exe scripts/run_flag_parallel_wind.py --output output/flag_L0_seed1mm_new --reuse-assets output/flag_L0_parallel_wind_20261121 --seed 0.001
.venv/Scripts/python.exe scripts/report_flag_parallel_wind.py --root output/flag_L0_seed1mm_new
```

本次比较动画可用 `scripts/compare_flag_seed.py` 重新生成；该脚本固定使用上述两个已完成目录。
