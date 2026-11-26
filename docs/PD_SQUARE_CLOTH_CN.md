# Progressive Dynamics 方布复现

2026-11-14 · Scene527 · libuipc 0.0.28 · RTX 4090

在 Scene527 上接入了四级方布、细网格弹性代理和跨层速度预测。两项数值自检通过，四级均完成软抓取首折。现有衬衫轨迹在方布上未通过纯摩擦抓取。共形网格的膜内能量对照未验证锁死改善，软抓取动作段的跨层 RMS 为 102–169 mm。方法背景见 [SIGGRAPH 2024 论文项目页](https://pcs-sim.github.io/pd/)。

## 实现和生效配置

| 阶段 | 实现与修改入口 |
|---|---|
| 场景 | `main()` 加载 OBJ 和双 X5 轨迹，`IPCCoupler._add_fem_entities_to_ipc()` 注册布料。方布边长 0.64 m，生成中心为 (0.667, 0.015, 0.93) m。 [场景入口](../dependencies/scene527-0.0.28/src/run_genesis_ipc.py) · [接入补丁](../patches/pd-cloth-scene527.patch) |
| 网格与延拓 | 每边分别为 8、16、32、64 个单元，各层有 81、289、1,089、4,225 个顶点。使用静止构型的 2D 重心坐标构造直达与相邻 CSR 矩阵，按顶点 XYZ 展开。 [生成脚本](../tools/assets/generate_pd_square_cloth.py) · [网格清单](../outputs/pd_square_cloth/manifest.json) |
| 粗层弹性 | $l<L$ 时，`FiniteElementElastics::Impl::assemble()` 和 `compute_energy()` 使用同一代理 $\Psi_L(P_lx_l)$。CUDA 计算细层拉伸、剪切、弯曲后，以 $P_l^\top g_L$ 和 $P_l^\top H_{L,\mathrm{proj}}P_l$ 装配粗层。惯性、夹爪约束及 IPC 碰撞在当前层计算。 [CUDA 实现及积分器补丁](../patches/pd-cloth-libuipc-0.0.28.patch) |
| 时间步进 | $l=0$ 用 BDF1。细层通过 `--pd-cloth-integrator progressive` 启用 $x_l^t+h(v_l^t+P_l^{l-1}\Delta v_{l-1}^{t+1})$。驱动程序按 IPC 子步生成上一层的延拓增量，记录每步结束的求解器位置和速度。 [运行脚本](../tools/pd/run_square_fold.py) · [预测 kernel](../dependencies/libuipc/src/backends/cuda/finite_element/bdf/fem_bdf1_time_integrator.cu) |

实验使用原生材料参数，拉伸模量输入 9,800、剪切模量输入 4、弯曲参数 10⁻⁷、密度 800 kg/m³、碰撞半厚度 0.1 mm，IPC 激活距离 1.5 mm。每次运行使用 seed 0，沉降 60 个子步，再执行原轨迹的前 600 个动作帧。动作频率 60 Hz，每帧两个 IPC 子步。软抓取每只手选一个邻近顶点，强度参数 100，同样启用 IPC 接触。 [实际执行命令](../outputs/pd_square_fold_soft/pd_L0/command.json) · [场景记录](../outputs/pd_square_fold_soft/pd_L0/fold.json)

## 数值验收

- **静止构型**。7 个延拓矩阵重建误差最大为 1.11 × 10⁻¹⁶ m，各行权重和严格等于 1。 [自检代码](../tests/test_pd_square_cloth.py) · [矩阵清单](../outputs/pd_square_cloth/manifest.json)
- **代理导数**。三个粗层各取两组随机扰动，每组做 12 个方向的中心差分。梯度有限差分最大绝对误差 1.85 × 10⁻¹⁴，投影前 $Hv$ 的有限差分误差 3.28 × 10⁻¹²。原生细层 CUDA 海森经 CPU 回传后，与粗层 GPU 投影装配的逐项最大差为 3.28 × 10⁻¹⁴。误差采用求解器的 $h^2$ 缩放。 [自检代码](../tests/check_pd_cloth_proxy.py) · [六组原始结果](../outputs/pd_proxy_build/self_check_all.json)
- **时间索引**。两步 CUDA 解析测试的位置误差至多 2.22 × 10⁻¹⁶ m，速度误差至多 2.13 × 10⁻¹⁴ m/s。完整软抓取实验的三份速度增量输入与离线 $P\Delta v$ 逐项相等。 [解析测试](../tests/check_pd_cloth_integrator.py) · [测试日志](../outputs/pd_proxy_build/step4_integrator.log)

## 首折结果

| 运行 | 动作耗时 | 相对上一层延拓轨迹的动作段 RMS |
|---|---:|---:|
| 原生 8×8 | 64.50 s | — |
| PD 8×8 | 72.95 s | — |
| PD 16×16 | 71.46 s | 168.7 mm |
| PD 32×32 | 97.71 s | 102.2 mm |
| PD 64×64 | 228.22 s | 132.1 mm |

统计覆盖软抓取首折的 600 个动作帧。耗时取动作循环墙钟时间，RMS 按同一子步的细层顶点与上一层延拓顶点计算。五组均记录两次附着、两次释放，Newton 触及迭代上限的子步为 0。不同分辨率按同一 TCP 邻近规则选点，实际受约束的材料点随分辨率变化。 [逐层结果与误差](../outputs/pd_square_fold_soft/comparison.json) · [8×8 状态序列](../outputs/pd_square_fold_soft/pd_L0/state.npz)

纯摩擦对照中，两只夹爪在原生和代理两组均判定抓取失败，方布最大抬升约 0.72 mm。软抓取下，原生和代理最大抬升分别为 135.1、134.3 mm，释放后的顶点高度 P95–P05 分别为 65.6、65.3 mm。这个高度差描述整体折后形态，局部折痕厚度需要专用测量。 [纯摩擦记录](../outputs/pd_square_fold_friction/comparison.json) · [粗层关键帧对照](../outputs/pd_square_fold_soft/coarse_fold_comparison.png)

细网格是粗网格的共形细分。分片线性延拓在每个粗三角形内产生恒定变形梯度，同一均匀膜内材料在子三角形上的积分之和等于该粗三角形积分。四组扰动的原生与代理膜内能量最大差为 5.55 × 10⁻¹⁶ J，因此这组对照无法辨别膜锁死改善。后续若要检验锁死，需先选择能使两种膜内离散产生可测差异的配对，并在相同材料点施加约束。若要定位跨层误差，先固定各层的材料空间抓取点，再比较速度预测开关。 [膜内能量核对](../outputs/pd_square_fold_soft/membrane_equivalence.json)

## 纯摩擦与布料中部抓取

保持方布尺寸和材料，沿用原轨迹时，8×8 和 16×16 的两手均未通过抓取，32×32 的左手通过、右手失败。随后在 32×32 将两手的 TCP 向布内移动，X 方向各移 100 mm，Y 方向各向中心移 55 mm。闭爪阶段左右手分别下调 2、6 mm，张爪推迟 37 帧。原生与代理组使用同一轨迹，均通过 IPC 摩擦带动布片。 [分辨率回放](../outputs/pd_friction_resolution/lift_comparison.mp4) · [中部抓取轨迹](../outputs/pd_friction_resolution/center_2_6_100_55.npz)

从第 270 帧到第 547 帧闭爪保持段，固定材料片区相对手腕坐标的最大漂移如下。两组左手在开始张爪后发生明显回弹，现有诊断将逐渐张爪纳入事件，直到开度达到 60% 才结束。两组右手均通过该诊断。 [逐帧统计](../outputs/pd_friction_resolution/comparison.json) · [原生运行命令](../outputs/pd_friction_resolution/center_32_native/command.json) · [代理运行命令](../outputs/pd_friction_resolution/center_32_proxy/command.json)

| 32×32 中部抓取 | 左手最大漂移 | 右手最大漂移 | 触及 Newton 上限的子步 |
|---|---:|---:|---:|
| 原生粗层弹性 | 4.4 mm | 1.5 mm | 3 |
| PD 代理弹性 | 4.3 mm | 1.9 mm | 4 |

[中部纯摩擦抓取并排回放](../outputs/pd_friction_resolution/center_comparison.mp4)左侧为原生弹性，右侧为 PD 代理。

## 复现与文件入口

使用本地 Scene527 v0.0.28 部署、项目 Python 环境和配套 CUDA 后端。libuipc 源码基线为 `9c5710ee4b716502e3757e7312360e5ee68cb79f`，两份补丁保存工作区的 PD 增量。配置或重建 CUDA 后端见 [CMake 入口](../tools/pd/CMakeLists.txt)，网格与代理资产由 [网格生成](../tools/assets/generate_pd_square_cloth.py) 和 [代理打包](../tools/assets/prepare_pd_cloth_proxy.py) 脚本生成。

```bash
envs/genesis-ipc/bin/python -B tools/pd/run_square_fold.py \
  --output outputs/pd_square_fold_soft_rerun --frames 600 --virtual-grasp
envs/genesis-ipc/bin/python -B tools/pd/run_square_fold.py \
  --output outputs/pd_square_fold_friction_rerun --frames 600 --last-level 0
```

每个运行子目录内的 `command.json`、`fold.json`、`simulation.log` 和 `state.npz` 对应执行命令、场景指标、求解日志和原生 IPC 状态。历史 55k 原衬衫运行另见 [Scene527 复现指南](SCENE527_REPRODUCTION_CN.md)。

### D 组齿轮披覆的日常调试配置

层级从粗到细排序。日常调试将最后三分之一的层设为 50 Hz、PCG 相对残差阈值 1e-3，其余层用 100 Hz、1e-5。层数不能被 3 整除时，降频层数向下取整，至少保留一个降频层。当前 D 组入口支持三层，对应 100/100/50 Hz。使用对角预条件器，Newton 上限 100，各层位移阈值均为 0.01 mm。严格验证用全层 100 Hz、PCG 1e-5。

从仓库根目录运行下列命令可使用日常配置。指定现有实验目录时可复用相同命令的已完成阶段。更换输入资产或后端构建后请使用新的输出目录。

```bash
envs/genesis-ipc/bin/python -B -m tools.pd.benchmark_d_pcg outputs/pd_gear_drape_50k_sweep/D outputs/pd_D_frequency daily
```

[分层频率与预条件器实测](../outputs/pd_D_frequency/REPORT.md) · [同频 PCG 精度对照](../outputs/pd_D_pcg_speed/REPORT.md)
