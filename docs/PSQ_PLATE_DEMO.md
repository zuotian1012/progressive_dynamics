# PSQ 薄板压屈与剪切验证

这组实验对应 PSQ 2023 Fig.16 的薄板测试类型。它采用真实弹性平衡求解，
内部顶点没有预设轨迹；没有惯性、重力或人为放大面外位移。它是本项目的
可重复受控算例，不声称恢复了作者未公开的网格、夹持条件或加载脚本。
这是较小规模的验证版：论文 Fig.16 的层级约为 1.6K→24K 顶点，
本次为 145→625 顶点。不能将本次粗层偏差直接推广为论文同规模实验的结论。

结果目录：`output/psq_plate_boundary_fixed/report/`。主入口：
`scripts/demo_psq_plate.py`、`scripts/run_psq_plate_suite.py`、
`scripts/report_psq_plate.py`。

## 材料、边界和加载

| 项目 | 设置 |
|---|---|
| 原始薄板 | 0.3 m × 0.3 m 平板，内部顶点轻微扰动的三角网格 |
| 各层 | L0：145 点 / 192 面；L1：241 点 / 384 面；L2：625 点 / 1152 面 |
| 材料 | E=70 GPa，ν=0.33，厚度=0.318 mm；Neo-Hookean 膜能与离散铰链弯曲能 |
| 参数依据 | PSQ 补充材料 Table 1 的 Fig.16 行；尺寸与加载方式为本实验设置 |
| 边界 | x 两侧边的 xyz 位置受控，y 两侧边自由；没有额外约束边界斜率 |
| 压屈 | 两侧对称压缩，总缩短 18 mm，24 个载荷增量 |
| 剪切 | x 两侧沿 y 反向移动，相对位移 36 mm，24 个载荷增量 |
| 初始分支 | 首次求解初猜加入至多 30 μm 的正弦面外扰动；静止形状仍完全平坦 |
| 收敛 | 自由点力残差无穷范数 < 0.001 N，且完整 Newton 更新无穷范数 < 30 nm |

密度沿用论文参数，但该实验没有质量或惯性项，密度不影响平衡形状。
材料为弹性模型，不含塑性。当前载荷采样不用于识别精确的临界屈曲载荷。

## 层级和映射

从原始细网格实际执行概率二次误差边折叠。新增可选 `--lock-boundary`
禁止折叠任何涉及边界顶点的边；每层保留相同的 96 个材料边界点，坐标差
仅为 OBJ 序列化的浮点舍入。原始最细层位置与拓扑保持不变。

仿真使用平面共有材料坐标上的精确重心映射作为 PSQ 的平面特例。独立对照
同次折叠历史输出的原生逐次自参数化权重，相邻层最大差分别为
3.49e-14、3.74e-14。这不是最近点投影。两套映射均恢复平面静止形状，
PSQ 残差为数值零。因此这组实验不重复验证非零法向/切向残差的搬运。

原生映射和折叠历史在 `assets/`，独立平面映射在 `planar_maps/`。
`boundary_mesh_audit.json` 记录边界保持误差及最小三角形角度。
细化只在已独立确认无自交的非接触状态下直接初始化；没有调用安全边展开。

## 五组对照的含义

- Direct L0：在 L0 自身网格上计算弹性能，随载荷连续求解。
- PSQ L0：在 L0 自由度上，通过延拓计算 L2 弹性能。
- PSQ L1：从同一载荷下的 PSQ L0 平衡解延拓，继续在 L1 求平衡。
- PSQ L2：从同一载荷下的 PSQ L1 解初始化，在原始细网格上继续求平衡。
- Direct L2：从独立的细网格加载历史求解，不读取粗层结果。

求解采用投影 Newton、Armijo 线搜索和膜退化步长限制。Hessian 使用
投影细层 Hessian 的 UᵀHU 形式，这是显式记录的求解器选择，不是论文
默认粗层 Hessian 预条件器的逐行复现；不据此宣称论文加速比。

对于高刚度材料，原始能量公式在接近平衡时出现浮点相消。demo 内使用
数学等价的 log1p/级数表达式计算 Neo-Hookean 能量，梯度与 Hessian 沿用
已有解析实现。`*_energy_check.json` 保存与原公式及方向有限差分的核对。
没有通过放宽力或位移容差使失败算例通过。

## 如何阅读结果

压屈外形接近不等于力学响应接近：末态 Direct L2 反力约 18.73 N，
PSQ L2 约 18.73 N，但 PSQ L0 约 206.30 N，PSQ L1 约 56.27 N。
PSQ L0 膜能约 1.434 J，Direct L2 约 0.00329 J，说明粗层代理的受限
变形空间仍存在明显偏硬现象。这是观测结果，不把它包装成粗层精度验证通过，
也不能仅凭本算例就把全部偏差归因于某一个实现模块。

最终两种载荷各 5 组、每组 24 个加载点全部满足相同收敛标准，所有保存状态
及其到共同细网格的映射均未检出自交；10 个末态的精确约束 Hessian
最小特征值均为正。

压屈的 PSQ L2 与独立 Direct L2 末态材料点 RMS 为 0.0000924 mm。
剪切的对应 RMS 为 3.82015 mm，两个末态分别具有不同的稳定褶皱分布：
PSQ L2 / Direct L2 能量为 2089.86948 / 2090.17799 J，约差 0.0148%；
反力为 122.10469 / 122.13418 kN，约差 0.0241%。不做镜像或形状配准来
消除这个点对点差异。这一结果支持不同初始化可以进入不同局部稳定分支，
不支持“所有层的形状都必须完全一样”的验收标准。

这组测试验证了当前加载路径的数值收敛和末态局部稳定性；粗层反力精度
仍有明显不足。它不是完整 PSQ 鲁棒性或原作者性能实验的验收结论。

`audit.json` 给出每组的全部载荷收敛检查、边界误差、采样状态自交检查、
末态精确约束 Hessian 最小特征值、材料点 RMS、反力与能量组成。
RMS 比较时各层映射到同一原始细网格，按静止面积加权。
Hessian 检查只针对末态，不是所有载荷点稳定性的完整证明。

预检查和未通过的试运行目录保留；最终报告只读取 `_verified` 后缀的
五组完整结果，不将迭代上限退出或未收敛帧计入成功实验。

## 复跑

先按 `native/intrinsic_builder/README.md` 编译到独立的
`output/intrinsic_plate_build`，避免覆盖历史二进制。下面的输出目录必须不存在：

```powershell
.venv/Scripts/python.exe scripts/demo_psq_plate.py --prepare --output output/my_plate_test
.venv/Scripts/python.exe scripts/run_psq_plate_suite.py --case compression --output output/my_plate_test
.venv/Scripts/python.exe scripts/run_psq_plate_suite.py --case shear --output output/my_plate_test
.venv/Scripts/python.exe scripts/report_psq_plate.py --output output/my_plate_test
```

论文来源：https://pcs-sim.github.io/psq/psq-combined.pdf
