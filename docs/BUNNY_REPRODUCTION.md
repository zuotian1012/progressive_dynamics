# PD++ 布料落到兔子：先验证算法，再应用折衣

## 重跑无惩罚的 bunny 实验

本轮排查使用仓库重建的布料、兔子和无摩擦地面。布料自接触、布料与兔子、布料与地面的摩擦系数均为 0。两个积分器共用一个 L0，速度延拓均用内在线性矩阵 U。原版 PD 的惯性预测从父层延拓位置出发，VelPro 从本层已接受的位置出发。两种积分器都保留法向接触和应变限制，层间一致性权重均为 0。

对装配顺序做复核时，使用 `bunny_ordered_fem` 构建。它以固定顺序累加 PSQ 材料梯度、FEM 和接触的梯度与矩阵项，以及接触能量，然后用双精度稀疏 LDLT 求解。此前固定第 39 帧位置和速度重算，GPU 的原子累加会让同一个 Newton 系统的右端项发生变化。修复后，两次独立重放的全部 20 组矩阵、右端项和方向文件逐字节相同，最终顶点逐位相同。[PSQ 梯度累加](../native/libuipc/bunny_gradient_gather.inl) · [FEM 装配](../native/libuipc/bunny_ordered_assembly.inl) · [逐次比较](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/replay_comparison.json)

从静止独立计算两次 L0，覆盖 4 秒及布料滑落后的地面接触。固定装配顺序后，两个包含 401 帧的 `positions.npy` 的 SHA-256 同为 `c3dbad3ad90af6c223d023471b18787ade784673778c8a4621b43a8cd97a5ead`。每步的 Newton 次数也一致。此前 GPU 迭代求解路线的两次重复间，质量加权材料点 RMS 偏差峰值为 173.70 mm。在已用稀疏 LDLT 和有序 PSQ 梯度的对照中，补上 FEM 与接触的固定累加顺序前峰值为 181.87 mm，补上后为 0。[固定 PSQ 累加顺序的对照](../output/bunny_velpro_diagnosis/frictionless/ordered_psq/repeatability.json) · [完整修复的重复性度量](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/repeatability.json)

下面的命令重现 0.7 mm 激活距离的零摩擦实验。输出目录应为新目录。L0 和两种 L1 各跑 4 秒，再以这两份 L1 的前 25 个状态为父轨迹，从静止逐层细化至 L7，测量 0.24 秒内的首次披覆。

```powershell
$py = '.venv/Scripts/python.exe'
& $py scripts/build_flag_l3_fix.py --variant bunny_ordered_pcg
& $py scripts/run_bunny_ground_l0.py --scene bunny --assets output/bunny_paper_assets --output ZERO_REF_L0 --contact-distance .0007 --frozen-offset --friction 0 --steps 400 --max-newton 1500 --refined-probe --adaptive-search 2>&1 | Out-File ZERO_REF_L0.log -Encoding utf8
& $py scripts/audit_paper_run.py ZERO_REF_L0
& $py scripts/audit_paper_non_descent.py ZERO_REF_L0
& $py scripts/audit_bunny_ground.py ZERO_REF_L0
foreach ($method in 'semi','velpro') {
    & $py scripts/continue_bunny_ground.py --lower ZERO_REF_L0 --method $method --output "FOUR_SECOND_$method" --last 1
    if ($LASTEXITCODE) { throw "Bunny $method failed" }
    & $py scripts/measure_bunny_chain.py "FOUR_SECOND_$method" --movie
    & $py scripts/continue_bunny_ground.py --lower "FOUR_SECOND_$method/L1" --method $method --output "CONTACT_$method" --steps 24 --last 7 --native-package output/flag_L3_fix_20261118/bunny_ordered_pcg_runtime/src
    if ($LASTEXITCODE) { throw "Bunny $method refinement failed" }
    & $py scripts/measure_bunny_chain.py "CONTACT_$method" --movie
}
& $py scripts/compare_bunny_integrators.py FOUR_SECOND_semi FOUR_SECOND_velpro FOUR_SECOND_comparison
& $py scripts/compare_bunny_integrators.py CONTACT_semi CONTACT_velpro CONTACT_comparison
```

### 0.7 mm 激活距离的 4 秒结果

两次从静止生成的直接分解 L0 都完成了 400 步。两个 401 帧位置文件的 SHA-256 同为 `cf80b0b9404f5eeeaf3bf2b7c7ddd6a73a9247b2d9d2587419aacc14fda2f2a9`，逐帧最大顶点差为零。[重复性检查](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/rest_clearance_full/repeatability.json)

L0 和两个 L1 使用有序装配与稀疏直接分解，每条轨迹的 401 个保存状态均通过检查。将各层延拓到同一张 9 万顶点网格后，原版 PD 的材料点 RMS 峰值为 2.12 mm（0.40 s），末帧为 0.37 mm，最低轮廓交并比为 99.07%。VelPro 的 RMS 峰值为 17.54 mm（4.00 s），最低轮廓交并比为 93.43%（1.16 s）。平铺时的额外排斥消除后，接触阶段的层间差异仍然存在。[同步视频](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/rest_clearance_full/paired_L1/comparison.mp4) · [最低轮廓重叠帧](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/rest_clearance_full/paired_L1/minimum_overlap.png) · [两条分支度量](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/rest_clearance_full/paired_L1/metrics.csv)

对较高分辨率可在同一套有序装配上使用原生 GPU PCG。构建 `bunny_ordered_pcg` 只替换最后执行线性求解的翻译单元，继续沿用相同的能量、梯度、对称矩阵和接受判据。固定 0.7 mm 实验的第 39 帧重算下一步，第一组 Newton 矩阵及右端项与直接分解版逐字节相同，两个方向相对差为 4.52e-6，单步末态最大顶点差为 0.0284 mm。[同一起点的两种线性解](../output/bunny_velpro_diagnosis/frictionless/rest_clearance_contact_24/linear_check/comparison.json)

将该组 L0 的前 25 个状态直接延拓到 L7，11 帧与兔子相交，首见于 0.14 s。另有 8 帧出现三角面超过 6.8% 的拉伸上限。0.24 s 时超限面积占布料静止面积的 0.136%。在这些位置，细层必须调整部分材料点才能满足自身的接触和应变条件。[延拓后的逐帧几何检查](../output/bunny_velpro_diagnosis/frictionless/rest_clearance_contact_24/L0_on_finest_feasibility.json)

### 0.7 mm 激活距离的八层首次披覆

从静止到 0.24 s 的每层材料点 RMS 峰值与同一相机下的最低可见轮廓交并比如下，均以共同 L0 为参考。位置误差覆盖 25 个状态，轮廓来自 7 张同步帧。两个 L7 的 24 步全部达到 Newton 收敛判据。

| 层级 | 顶点 | 原版 PD RMS 峰值 | 最低轮廓交并比 | VelPro RMS 峰值 | 最低轮廓交并比 |
|---|---:|---:|---:|---:|---:|
| L0 | 1,443 | 0.00 mm | 100.00% | 0.00 mm | 100.00% |
| L1 | 2,663 | 0.80 mm | 99.72% | 2.06 mm | 99.49% |
| L2 | 4,834 | 1.67 mm | 99.59% | 4.07 mm | 98.85% |
| L3 | 8,608 | 2.67 mm | 99.24% | 6.10 mm | 98.45% |
| L4 | 15,651 | 3.70 mm | 99.15% | 8.06 mm | 98.05% |
| L5 | 28,185 | 5.19 mm | 98.70% | 9.69 mm | 97.73% |
| L6 | 50,268 | 6.40 mm | 98.17% | 11.01 mm | 97.46% |
| L7 | 90,000 | 8.86 mm | 98.09% | 12.10 mm | 97.23% |

八层在兔子上的整体披覆位置接近，细层在兔耳侧面和靠近前缘处出现更密的褶皱。[八层同帧图](../output/bunny_velpro_diagnosis/frictionless/rest_clearance_contact_24/paired/final.png) · [同步视频](../output/bunny_velpro_diagnosis/frictionless/rest_clearance_contact_24/paired/comparison.mp4) · [逐层度量](../output/bunny_velpro_diagnosis/frictionless/rest_clearance_contact_24/paired/metrics.csv)

### 保留旧 1 mm 激活距离的 4 秒对照

下表沿用原重建场景的 IPC 激活距离 1 mm。L0 和两个 L1 均完成 400 步，401 个保存状态均通过几何检查。使用同一相机每 0.04 s 比较一次可见布料轮廓。材料点误差在每个保存状态计算，按静止面积加权，两张网格都用各自的 PSQ 算子延拓到相同的 9 万材料点。

| 与共同 L0 比较 | 原版 PD L1 | VelPro L1 |
|---|---:|---:|
| 材料点 RMS 峰值 | 2.23 mm（0.40 s） | 19.34 mm（1.91 s） |
| 在该峰值帧去除质心平移后的 RMS | 2.18 mm | 18.45 mm |
| 4 秒末帧 RMS | 0.45 mm | 8.53 mm |
| 最低可见轮廓交并比 | 99.04%（0.96 s） | 93.87%（1.96 s） |

两种方法都在 0.34 s 开始触地。VelPro 的最大轮廓差出现在滑离兔子后，布边和折叠的位置有可见差异。峰值帧的质心偏移为 5.79 mm，去掉质心平移后仍有 18.45 mm 的材料点偏差。[同步视频](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/paired_L1/comparison.mp4) · [轮廓差异最大的一帧](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/paired_L1/minimum_overlap.png) · [逐帧 VelPro 数据](../output/bunny_velpro_diagnosis/frictionless/ordered_fem/velpro/level_errors.csv)

### 平铺布料的初始自接触

在没有碰到兔子和地面时，L5、L6 的原生日志已分别报告 20、18 个点－边候选。用独立 IPC Toolkit 对初始网格检查，得到 7、6 个去重后的活动点－边对。每一对的三个顶点属于同一个三角形。两层的最小活动距离分别为 0.811 mm、0.872 mm。原参数将 0.070 mm 的实体间隙与 1.000 mm 的激活距离相加，因此某些平铺三角形在静止构形中就受到了排斥力。L0–L4 和 L7 的独立检查均为零。[逐层几何记录及顶点编号](../output/bunny_velpro_diagnosis/frictionless/rest_contacts_1mm.json)

重建 bunny 的新 L0 默认使用 0.700 mm 激活距离，所有细层继承同一数值。重放历史轨迹时自动读取原有宽度。此时八层初始网格上的活动自接触数均为零。这个选项调整法向屏障开始作用的距离，影响后续真实接触；布料厚度、CCD、摩擦和一致性权重各自独立设置。对比旧数据时使用 `--contact-distance .001`。[0.700 mm 的八层检查](../output/bunny_velpro_diagnosis/frictionless/rest_contacts_0_7mm.json)

用同一份有序装配 DLL 从静止再做原生单步检验。原 1 mm 设置下，L5 和 L6 对自由落体解析位置的最大顶点偏差分别为 0.02723 和 0.02672 mm。改为 0.7 mm 后，两种积分器在全部八层的第一步都通过无相交、应变和能量检查，最大单顶点偏差为 5.21e-8 mm。两种积分器从同一静止状态起步，第一步的预测相同。[逐层原生候选数和解析误差](../output/bunny_velpro_diagnosis/frictionless/rest_clearance_one_step/first_step_geometry.json)

### 先前用于定位问题的对照

保留迭代线性求解器时，两种方法都完成了从静止到第 24 步的 8 层实验。末帧处于首次披覆，所有层都尚未触地。与 L0 比较，L7 的共同网格 RMS 为原版 PD 8.75 mm 和 VelPro 12.14 mm，同帧可见轮廓交并比分别为 98.11% 和 97.19%。两组的接触区域位置接近，较细网格上布边的褶皱逐渐增多。[八层同步图](../output/bunny_velpro_diagnosis/frictionless/first_contact_24/paired_L0_L7/final.png) · [各层数值](../output/bunny_velpro_diagnosis/frictionless/first_contact_24/paired_L0_L7/metrics.csv)

先生成共同的 L0，再分别运行原版 PD 的 semi-diagonal 和 PD++ 的 VelPro。以下命令使用同一 CUDA 构建、资产和初始状态。步长为 0.01 秒，观察窗口为 4 秒。两条分支的一致性势能权重均为零。请选择全新的输出目录。

```powershell
.venv/Scripts/python.exe scripts/build_flag_l3_fix.py --variant bunny
.venv/Scripts/python.exe scripts/build_flag_l3_fix.py --variant bunny_energy
.venv/Scripts/python.exe scripts/build_flag_l3_fix.py --variant bunny_frozen
.venv/Scripts/python.exe scripts/run_bunny_ground_l0.py --scene bunny --assets output/bunny_paper_assets --output NEW_L0 --contact-distance .001 --exact-offset --refined-probe --adaptive-search *> NEW_L0.log
.venv/Scripts/python.exe scripts/audit_paper_run.py NEW_L0
.venv/Scripts/python.exe scripts/audit_paper_non_descent.py NEW_L0
.venv/Scripts/python.exe scripts/audit_bunny_ground.py NEW_L0
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --method semi --lower NEW_L0 --output NEW_SEMI --max-newton 1500 --last 7
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --method velpro --lower NEW_L0 --output NEW_VELPRO --max-newton 1500 --last 7
.venv/Scripts/python.exe scripts/measure_bunny_chain.py NEW_SEMI --movie
.venv/Scripts/python.exe scripts/measure_bunny_chain.py NEW_VELPRO --movie
```

全零摩擦对照从静止重新生成 L0。在上面的 L0 命令加上 `--friction 0` 并使用新目录，两个细化命令的 `--lower` 都指向这个零摩擦 L0。细层自动继承摩擦系数。该选项将布料与兔子、布料自接触和地面摩擦都设为零，并关闭 CUDA 摩擦系统。法向 IPC 接触、应变限制、时间步、惯性项和弹性项沿用同一配置。审计直接读取原生后端的 `backend/systems.json`，核验两种法向接触系统均已启用，并核验摩擦系统的实际状态。

要检查 PD 2024 §4.6 指定的冻结偏移梯度，请使用下面的 `--frozen-offset` 命令。每一层继承父层的梯度模式、原生求解器和法向接触刚度设置。下文比较摩擦系数 0 与 0.3 的 L1 对照使用完整 PSQ 偏移梯度。两种梯度使用相同的弹性能和零一致性权重。

要同时使用 C-IPC 的自由顶点 RMS 停止判据，在上述构建后依次构建 `bunny_paper` 和 `bunny_precision`。以下命令开启零摩擦和冻结偏移梯度。每个粗层沿碰撞检测允许的方向回溯；若全部顶点坐标已舍入回迭代起点，记录一次数值停滞，保留该粗层预览。最细层要求正常收敛。日志和审计分别统计负能量变化、非下降方向、数值停滞和正常收敛。[粗层回溯实现](../native/libuipc/bunny_representable_search.inl)

```powershell
.venv/Scripts/python.exe scripts/build_flag_l3_fix.py --variant bunny_paper
.venv/Scripts/python.exe scripts/build_flag_l3_fix.py --variant bunny_precision
.venv/Scripts/python.exe scripts/run_bunny_ground_l0.py --scene bunny --assets output/bunny_paper_assets --output ZERO_L0 --contact-distance .001 --native-package output/flag_L3_fix_20261118/bunny_precision_runtime/src --frozen-offset --friction 0 --steps 32 --max-newton 1500 --refined-probe --adaptive-search 2>&1 | Out-File ZERO_L0.log -Encoding utf8
.venv/Scripts/python.exe scripts/audit_paper_run.py ZERO_L0
.venv/Scripts/python.exe scripts/audit_paper_non_descent.py ZERO_L0
.venv/Scripts/python.exe scripts/audit_bunny_ground.py ZERO_L0
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --lower ZERO_L0 --method semi --output ZERO_SEMI --last 7 --max-newton 1500
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --lower ZERO_L0 --method velpro --output ZERO_VELPRO --last 7 --max-newton 1500
.venv/Scripts/python.exe scripts/measure_bunny_chain.py ZERO_SEMI --movie
.venv/Scripts/python.exe scripts/measure_bunny_chain.py ZERO_VELPRO --movie
```

`--steps 32` 覆盖从静止到布料首次披到兔子上的 0.32 秒。所有层继承相同观测窗，使用全分辨率网格和原时间步。把步数改为 400 即可覆盖 4 秒。两次实验各用独立的输出目录和日志。

在冻结偏移、零摩擦、零一致性权重的 0.32 秒配对实验中，两种积分器共用同一条 L0。把每层延拓到同一张 9 万顶点网格后，前 0.13 秒 L0 至 L3 的材料点位置 RMS 差均小于 0.00001 mm。第 0.14 秒开始接触兔子，0.32 秒时原版 PD 的 L1/L2/L3 相对 L0 分别相差 0.77/2.15/3.23 mm，VelPro 分别相差 3.01/5.95/8.98 mm。同一时刻 L3 与 L0 的可见布料轮廓交并比分别为 98.64% 和 97.84%。所有层在这个窗口内均未触地。[原版和 VelPro 同相机对照](../output/bunny_velpro_diagnosis/frictionless/contact_window/paired_L0_L3/final.png) · [逐层 RMS 与轮廓](../output/bunny_velpro_diagnosis/frictionless/contact_window/paired_L0_L3/metrics.csv)

用同一实验的 L0 实际形状直接延拓到最细网格，33 个保存状态中有 18 个与兔子相交，首次出现在第 15 步。16 个状态有细网格三角面超过 1.068 的主拉伸上限。实际保存的粗层形状通过了该层的无相交和应变检查。这说明细层求解在这些帧必须调整部分材料点以保持可行。[逐帧延拓检查](../output/bunny_velpro_diagnosis/frictionless/contact_window/L0_on_finest_feasibility.json)

法向接触刚度也做了配对检查。输入刚度 1e9 经原生后端逐层限制，在 L0 和 L3 分别实际使用 2.903e7 和 1.132e7。在另一组 L0 加入 `--contact-stiffness 1427067.9156648605`，各层继承这个值，原生日志证实 L0 至 L3 全部落在允许区间。两组均从静止生成自己的 L0，每种刚度下的两个积分器共用父轨迹；相同层使用相同构建和物理参数。按各层区间限制刚度时，L3 相对 L0 的峰值 RMS 为原版 PD 3.24 mm、VelPro 9.02 mm。固定实际刚度后分别为 3.15 mm 和 9.34 mm。[两种刚度设置的原生有效值与误差](../output/bunny_velpro_diagnosis/frictionless/contact_window/stiffness_comparison/metrics.csv)

### 冻结偏移的 4 秒实验

L0 和两种 L1 均完成 400 步，401 个保存状态逐帧通过几何检查。牛顿方向使用论文的冻结偏移近似，启用自由顶点 RMS 收敛判据和上面的回溯检查，所有摩擦系数和一致性权重为零。两个 L1 读取同一份 L0 轨迹，材料点误差按面片静止面积分配的质量加权。轮廓数值来自 101 张同步的同相机渲染帧。

| 方法 | L1 上相对父层的 RMS 峰值 | 延拓到共同 9 万顶点后相对 L0 的 RMS 峰值 | 最低轮廓交并比 | 最后接触兔子 |
|---|---:|---:|---:|---:|
| L0 | 0 | 0 | 100% | 1.31 s |
| 原版 PD | 2.66 mm | 2.10 mm | 98.99% | 1.32 s |
| VelPro | 15.78 mm | 15.42 mm | 94.26% | 1.25 s |

L0 的 3,710 次 Newton 迭代中，84 次在通常的回溯次数用完后找到了严格下降步，另有 2 次在试探坐标舍入回起点时保留粗层预览。原版与 VelPro 的 L1 在 4 秒时相对 L0 的共同网格 RMS 分别为 0.77 和 13.29 mm。[两种积分器同步视频](../output/bunny_velpro_diagnosis/frictionless/paper_full/paired_L1/comparison.mp4) · [4 秒度量](../output/bunny_velpro_diagnosis/frictionless/paper_full/paired_L1/metrics.csv) · [逐帧布－兔接触](../output/bunny_velpro_diagnosis/frictionless/paper_full/contact_times.json)

在同一构建、配置和初始条件下再独立运行一次 L0，两次的质量加权 RMS 偏差从 0.32 秒的 0.00154 mm 增至 0.40 秒的 1.22 mm，峰值为 173.70 mm（3.63 秒）。地面从 0.34 秒开始接触。即使去掉质心平移，4 秒末帧仍相差 147.65 mm。该对照只涉及一个网格。[L0 重复实验](../output/bunny_velpro_diagnosis/frictionless/paper_full_repeat/repeatability.json)

为检验线性求解精度，在 `bunny_precision` 基础上另建 `--variant bunny_direct`，用 CPU 上 Eigen 的稀疏 LDLT 分解求解 CUDA 实际装配的同一 Newton 矩阵和右端项。分解逐次核验正主元和线性残差。第 35 步重放的四个方向又用 SciPy 稀疏 LU 独立求解，最大坐标差低于 3e-12 mm。两次独立的直接分解 L0 各完成 400 步，0.32 秒时相差 3.8e-7 mm，至 0.40 秒仍相差 1.12 mm，4 秒相差 416.94 mm。这一对照说明改用高精度直接分解也不能消除本重建场景的长时间重复差异。单对轨迹的终态大小仅适用于这次运行。[直接分解](../native/libuipc/bunny_sparse_ldlt.inl) · [四个 Newton 方向的独立复核](../output/bunny_velpro_diagnosis/frictionless/sparse_direct/direct_solve_check.json) · [直接分解的同网格重复实验](../output/bunny_velpro_diagnosis/frictionless/sparse_direct/repeatability.json)

旧后端的两个能量接受分支会让上升步进入轨迹。一个发生在 Newton 已收敛时，另一个发生在 Newton 仍未收敛、8 次回溯用完且预览早停证据不足时。零摩擦冻结梯度实验的第 47 帧记录了第二种情况，最后试探点的能量上升了 6.64e-18，后端随后继续了 Newton 迭代。该次 400 步运行未通过逐次能量审计。修复后，普通回溯用完时继续沿已通过碰撞检测的方向缩小步长，逐次记录能量。若仍无法下降，就恢复当前 Newton 迭代起点并报告失败。对冻结梯度的粗层，还在原点计算完整目标的方向导数以核验是否满足论文的非下降早停条件。此处只检验既有方向，求解 Newton 方向时仍冻结偏移。[回溯实现](../native/libuipc/bunny_exhausted_search.inl) · [非下降方向检验](../native/libuipc/bunny_preview_derivative.inl) · [逐次回溯审计](../scripts/audit_flag_fine_search.py)

同一层的跨摩擦对照可以用 `scripts/report_bunny_friction_control.py --zero ZERO_SEMI ZERO_VELPRO --baseline NEW_SEMI NEW_VELPRO --output COMPARISON` 重新生成。四个输入目录均须先运行层级度量。每个摩擦设置使用自己的 L0 作为参考；视频六个面板使用同一相机和物理时刻，`metrics.csv` 保存相邻层和共同最细网格材料点的偏差。

两个积分器的首次细化各跑 400 步，结果如下。位置误差在实际 L1 网格上按静止面积加权。轮廓在 101 个同步帧比较可见布料像素，1 表示重合。

| 方法 | 布料与兔子、布料自接触 μ | L1 相对 L0 的 RMS 峰值 | 最低轮廓交并比 | L0 / L1 最后接触兔子的时刻 |
|---|---:|---:|---:|---:|
| 原版 PD | 0 | 2.81 mm | 99.59% | 1.05 / 1.05 s |
| 原版 PD | 0.3 | 2.90 mm | 99.57% | 1.75 / 1.75 s |
| VelPro | 0 | 14.76 mm | 95.22% | 1.05 / 1.04 s |
| VelPro | 0.3 | 12.16 mm | 96.41% | 1.75 / 1.75 s |

全零摩擦时两种 L1 的轮廓都接近共同 L0，VelPro 的材料点偏差仍然存在。原版 PD 的最大差出现在撞击时，零摩擦 VelPro 的最大差出现在 4 秒末帧。每个摩擦组内都固定同一条父轨迹。[两行三列同步动画](../output/bunny_velpro_diagnosis/frictionless/paired_L1/comparison.mp4) · [0.40 秒披覆形状](../output/bunny_velpro_diagnosis/frictionless/paired_L1/contact_0_40s.png) · [4 秒末帧](../output/bunny_velpro_diagnosis/frictionless/paired_L1/peak.png) · [逐方法度量](../output/bunny_velpro_diagnosis/frictionless/paired_L1/metrics.csv)

Newton 线搜索的旧实现先移动顶点，再判断搜索方向是否足够小。如果方向已经满足收敛容差，试探点会直接成为该步的结果。日志中某些收敛步的试探能量超过出发点的 13 倍。修复后，收敛步也须满足能量有限且不增加。若回溯用完预算，则恢复已经满足收敛判据的出发点，复算其能量。运行日志记录实际测试的能量，审计会逐项核验。[收敛步处理](../native/libuipc/psq_converged_trial.inl)

固定同一条 L0 轨迹、PCG 比值 1e-10 和所有物理参数，仅替换线搜索目标文件。在第 13 步，原版 PD 的 L1 相对自由落体解析解的最大顶点误差从 0.39248 mm 降到 0.000381 mm，VelPro 从 2.99993 mm 降到 0.001146 mm。[两种积分器逐步实测](../output/bunny_velpro_diagnosis/freefall_line_search.json)

在完整 4 秒的首次细化中，三条轨迹第一次接触兔子都在 0.14 秒，最后一次都在 1.75 秒。对应布料位置相对 L0 的面积加权 RMS 峰值，原版 PD 为 2.90 mm，VelPro 为 12.16 mm。同一相机每 4 步采样一次，共 101 帧，计算可见布料轮廓交集与并集的面积比。与 L0 的最低重叠率分别为 99.57% 和 96.41%。[披覆、滑动与脱离的同步画面](../output/bunny_velpro_diagnosis/energy_L1_comparison.png)

层间差异在布料接触兔子后就能测到。按半平面接触的激活距离检查，L0 和两种 L1 都在 0.34 秒第一次进入地面接触范围。此之前的 0.32 秒，原版 PD 与 L0 的面积加权 RMS 差为 1.03 mm，VelPro 为 3.28 mm。[原版 PD 逐帧数据](../output/bunny_velpro_diagnosis/energy_semi_L1/level_errors.csv) · [VelPro 逐帧数据](../output/bunny_velpro_diagnosis/energy_velpro_L1/level_errors.csv)

固定同一条 L0，只把 L1 的 Newton 最大坐标速度增量容差从 0.05 收紧到 0.001 m/s。VelPro 的 4 秒峰值从 12.16 变为 12.33 mm，两次都出现在 0.86 秒。原版 PD 在两次都接受的前 95 步内，峰值从 2.90 变为 2.96 mm，两次都在 0.39 秒。原版 PD 的紧容差对照在第 96 步达到 1500 次 Newton 迭代上限，该帧已拒绝。[VelPro 完整对照](../output/bunny_velpro_diagnosis/energy_newton_velpro_comparison/summary.json) · [原版 PD 共同时间窗](../output/bunny_velpro_diagnosis/energy_newton_semi_prefix.json)

每条层级轨迹的 `level_errors.csv` 提供相邻层的 PSQ 位置差、按论文式（13）投回父层的误差，以及各层用实际求解器的直接 PSQ 算子映射到同一张 90K 网格后的误差（`common_finest_L0_rms_mm`）。`silhouettes.csv` 使用动画中经遮挡处理的可见布料像素，每行保留交集和并集的像素数。数值误差覆盖每一保存帧，轮廓只能衡量指定相机下的外形。

将 L0 的 401 个保存状态通过它实际使用的直接 PSQ 算子映射到最细层，对映射结果独立检查几何可行性。其中 148 帧与兔子相交，165 帧有细三角形超过主拉伸上限 1.068。在 0.42 秒有 360 个细三角形超限，占布料静止面积 0.20%。这些数字衡量直接延拓的参考形状，求解器实际接受的 L0 状态通过了本层检查。按 PD 2024 式（1），每一层在本层网格上计算接触与应变限制，直接延拓仅用于计算细网格弹性。这个几何检查说明某些材料点必须调整位置才能得到可行的细层解，尚不能由此推出整张布的偏差幅度。[逐帧可行性](../output/bunny_velpro_diagnosis/energy_coarse_on_finest_feasibility.json) · [检查代码](../scripts/check_bunny_prolongation_feasibility.py)

### 同网格重复性

将零摩擦 L0 的 PCG 平方残差比从 1e-10 收紧至 1e-18，保留最大坐标 Newton 容差 0.05 m/s，再从静止独立跑两次。每次均完成 400 步并通过审计。本表每行代表一对独立重复。

| PCG 平方比值 | 0.50 s 的两次轨迹 RMS 差 | 全程峰值 | 4 s 的 RMS 差 | 4 s 对齐质心后的 RMS 差 |
|---:|---:|---:|---:|---:|
| 1e-10 | 7.46 mm | 185.27 mm | 183.90 mm | 164.76 mm |
| 1e-18 | 1.27 mm | 33.08 mm | 33.08 mm | 3.88 mm |

收紧后的两次轨迹在最后一帧具有更接近的褶皱形状，仍有水平位置差异。[四条轨迹同步动画](../output/bunny_velpro_diagnosis/frictionless/paired_repeat_precision/comparison.mp4) · [收紧后的逐帧数据](../output/bunny_velpro_diagnosis/frictionless/pcg_precise_repeatability.json)

再从同一条 1e-18 的 L0 分别跑两种 L1，全程保留这一 PCG 设置。原版 PD 和 VelPro 对父层的面积加权 RMS 峰值分别为 2.79 mm 和 14.10 mm。对照 1e-10 的结果分别为 2.81 mm 和 14.76 mm。每组内部固定父轨迹，两种精度之间的父轨迹各自从静止求解。[原版 PD](../output/bunny_velpro_diagnosis/frictionless/pcg_precise_semi_to_L1/level_errors.json) · [VelPro](../output/bunny_velpro_diagnosis/frictionless/pcg_precise_velpro_to_L1/level_errors.json)

在零摩擦条件下，还可以固定某一帧的顶点位置和 BDF1 速度，单独重算下一步。对下表的四组对照都使用第一条零摩擦 L0 的第 18 帧。原生读回验证初始位置和速度完全相同，每次独立求解第 19 步并验证接受能量。此时布料与兔子接触，尚未碰到地面。

| PCG 平方残差比容差 | Newton 最大坐标速度增量容差（m/s） | 两次末态 RMS 差（mm） | 单顶点最大差（mm） |
|---:|---:|---:|---:|
| 1e-10 | 0.05 | 0.06282 | 1.6042 |
| 1e-18 | 0.05 | 0.0000765 | 0.00250 |
| 1e-10 | 0.001 | 0.0000416 | 0.00093 |
| 1e-18 | 0.001 | 0.0000868 | 0.00228 |

同一起点的基线重算中，有 20 个粗顶点相差超过 0.1 mm，最大差位于自由边的顶点 810。该顶点在两次重算之一的终点距兔面约 105 mm。以同时收紧两种容差的第一条单步结果作数值参考，仅收紧 PCG 时两条结果各自都与参考相差约 1.48 mm（最大顶点差），仅收紧 Newton 时为 0.00070–0.00163 mm。两次重算彼此接近，与接近更严格的参考解，是两个单独测量的量。[与紧容差参考解的差值](../output/bunny_velpro_diagnosis/frictionless/frame19_vs_tight_reference.json)

另一个 CUDA 构建使用 C-IPC 的自由顶点 RMS 速度增量，并要求连续三次小于等于 0.001 m/s。固定相同的第 18 帧和 PCG 容差 1e-10，两次重算第 19 步分别用了 83 和 94 次 Newton 迭代，末态质量加权 RMS 差为 0.00283 mm，单顶点最大差为 0.06158 mm。[停止判据实现](../native/libuipc/bunny_rms_translation_checker.cu) · [单步实测](../output/bunny_velpro_diagnosis/frictionless/frame19_rms_repeatability.json)

对零摩擦共同 L0 的全部 401 个已保存状态另做几何检查。用该次求解的 PSQ 算子把粗布直接映射到 90K 网格后，65 帧与兔子相交，首次在第 15 帧。384 帧有细三角形超过拉伸上限，首次在第 17 帧，任一帧的超限静止面积最多占 0.221%。这些是粗解映射到细网格后的结果。L0 实际接受的几何已单独通过本层审计。[零摩擦逐帧几何检查](../output/bunny_velpro_diagnosis/frictionless/L0_on_finest_feasibility.json)

导出同一起点首轮 Newton 线性系统的另两次对照中，矩阵文件逐字节相同，右端项相对差为 2.29e-16。对两份系统用 CPU 稀疏直接法求解，解的相对差为 2.75e-15；原生 PCG 方向的相对差为 4.52e-7。离线使用实际三维对角块重算预条件残差，两个时间步解分别包含 23 和 35 次 Newton 迭代，所有线性求解都达到 1e-10 的平方比值判据。在单步内收紧任一容差明显改善了重复性。可在 L0 命令上加 `--restart ZERO_L0 --restart-frame 18 --steps 1` 重算，`--dump-linear-system` 导出每次迭代的矩阵、右端项和方向。[同一起点的单步对照](../output/bunny_velpro_diagnosis/frictionless/frame19_replay.json) · [首轮矩阵与方向](../output/bunny_velpro_diagnosis/frictionless/frame19_first_newton.json) · [真实 PCG 残差](../output/bunny_velpro_diagnosis/frictionless/frame19_pcg_residuals.json)

把所有摩擦系数都设为 0 后，独立重跑两次 L0，每次都通过 400 步审计。法向接触仍然启用，CUDA 实际加载的系统中没有单纯形或半平面摩擦模型。两次都在第 14 步接触兔子，最后一次接触分别在第 105 和 96 步。第 50 步对应顶点的 RMS 差为 7.46 mm，第 329 步达到 185.27 mm，末帧为 183.90 mm。将两次末帧的质心重合后，形状 RMS 差仍有 164.76 mm。由此可知，当前重建场景在关闭摩擦、完全不调用层间传递时也能产生轨迹分叉。这组数字衡量同网格重复性，跨层误差另用同一父轨迹计算。[零摩擦重复性](../output/bunny_velpro_diagnosis/frictionless/repeatability.json) · [实际接触模型](../output/bunny_velpro_diagnosis/frictionless/L0/continuation_audit.json)

用完全相同的初态、资产、物理配置、求解容差和 CUDA DLL 独立运行两次 L0。两次都完成 400 步，第 14 步接触兔子。两条轨迹在第 13 步的对应点 RMS 差约为 1.1e-9 mm。到第 18 步，Newton 迭代 8 的单位步长在一次运行中使能量升高，另一次降低，前者回溯到半步。这是两份试探日志中第一次一侧接受而另一侧回溯的位置。第 50 步 RMS 差为 16.4 mm，最后接触分别在第 175 和 178 步。离开兔子后，无摩擦地面保留了两次运行不同的水平速度。到 4 秒 RMS 差为 487.4 mm，扣除质心平移后的 RMS 差仍为 135.0 mm。这个对照没有使用层间传递；上文两个细化分支共同读取同一份已保存的 L0。可用 `scripts/compare_bunny_repeats.py FIRST REPEAT --output RESULT.json` 检查相同配置的两次运行。[逐帧 RMS 与质心](../output/bunny_velpro_diagnosis/energy_repeatability.json) · [重复运行接触时间](../output/bunny_velpro_diagnosis/energy_repeat_contacts.json)

## 接触前自由落体的解析校验

水平布从静止释放，初始高度 0.60 m，兔子最高点 0.50 m。前 13 个时间步的解析 BDF1 解与兔子之间均留有大于接触激活距离和厚度之和的间隙。每个顶点在第 `n` 步应平移 `g * dt² * n * (n+1)/2`。这个检查同时覆盖惯性、弹性和求解误差，且无需选取图像特征。

在同一新版 DLL、同一 L0 网格和完整 PSQ 梯度下，只改变求解容差得到以下第 13 步误差。原生 fused PCG 的比值为 `(rᵀ M⁻¹ r)/(r₀ᵀ M⁻¹ r₀)`，其中 M 为预条件器。原先的 0.001 对应预条件残差范数比约 3.16%。bunny 新建 L0 时默认使用 1e-10，细层继承所属 L0 的设置。历史轨迹按其实际使用的比值审计。

| Newton 速度增量容差（m/s） | PCG 比值 | 质量加权 RMS（mm） | 最大顶点误差（mm） |
|---:|---:|---:|---:|
| 0.05 | 1e-3 | 1.2864 | 19.4821 |
| 0.001 | 1e-3 | 0.01110 | 0.7460 |
| 0.05 | 1e-10 | 0.0000233 | 0.00143 |
| 0.001 | 1e-10 | 0.0000233 | 0.00143 |

用 `scripts/measure_bunny_freefall.py RUN` 可对实际保存的状态重复这一校验。下文摩擦接口的历史配对实验使用 1e-3，各项对照固定自己的输入轨迹和容差。[逐帧误差与容差对照](../output/bunny_velpro_diagnosis/freefall_tolerances.json)

对修复后的 VelPro 八层逐顶点复核，第 13 步相对解析解最大的单点误差出现在 L4，为 0.384 mm，质量加权 RMS 为 0.00397 mm。该帧 15,651 个顶点中有 7 个超过布料厚度 0.07 mm。固定 L3 输入并把 L4 的 Newton 最大坐标速度增量容差从 0.05 收紧到 0.001 m/s，单点最大误差降到 0.00228 mm。直接运行 `run_bunny_ground_fine.py` 时可用 `--newton-velocity-tol` 做这一配对测试，常规续算默认继承父层容差。[八层自由落体](../output/bunny_velpro_diagnosis/energy_velpro_freefall.json) · [同一 L4 的三种容差](../output/bunny_velpro_diagnosis/energy_L4_tolerance_freefall.json)

## 历史对照：在同一个粗预览上隔离摩擦接口

两种细层积分器都通过 `FiniteElementStateAccessorFeature` 写入下一步的预测速度。历史 CUDA 接口在只写速度时也调用 `request_attribute_update()`，触发 `require_discard_friction()`，使接触求解丢失上一帧的摩擦候选。修复将失效操作移入实际写入位置的分支。可通过 `scripts/run_bunny_ground_l0.py --replay-velocity-from <L0>` 在同一网格、同一完整代理能量上回放上一帧速度，隔离网格延拓的影响。

下述对照使用相同网格、材料、重力、摩擦、地面、时间步和求解容差，关闭一致性惩罚。历史粗预览在线搜索时重新计算 PSQ 的法向和切向偏移，却在通常的弹性梯度中冻结这些偏移。400 步中有 353 步以当前方向不下降结束。这样生成的 L0 最后仍披在兔子上。

原生后端已经实现了偏移的完整导数。现在 L0 和细层入口都能用 `--exact-offset` 对完整的代理弹性能求导。该选项改变梯度计算，保留原有能量；它是对论文冻结导数近似的数值对照。修正后的 L0 有 400 次 Newton 收敛，布滑离兔子，随后在无摩擦地面上移动。这个 L0 单独计算，尚未使用任何细层时间积分器。

```powershell
.venv/Scripts/python.exe scripts/build_flag_l3_fix.py --variant bunny
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --method velpro --lower output/bunny_velpro_diagnosis/exact_gradient_L0 --output HIST_VELPRO --native-package output/flag_L3_fix_20261118/bunny_runtime/src --max-newton 1500 --last 7
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --method semi --lower output/bunny_velpro_diagnosis/exact_gradient_L0 --output HIST_SEMI --native-package output/flag_L3_fix_20261118/bunny_runtime/src --max-newton 1500 --last 7
```

后两条命令使用同一条已通过检查的 L0 轨迹。L0 不写入外部速度。细层使用修复摩擦的 CUDA 库，各层牛顿迭代上限为 1500，收敛容差与原运行相同。续算程序从父层继承弹性梯度模式。VelPro 速度传递仍默认采用论文的冻结偏移近似 U；`--velocity-map psq-jacobian` 仅用于独立测试完整位置延拓的方向导数。在同一旧 L0 下，仅改变速度传递后，L1 末帧 RMS 由 122.07 mm 变为 121.67 mm。在配对的 CUDA 构建中，使用完整 PSQ 弹性梯度且仍丢弃摩擦候选时，保持 U 速度传递，L1 末帧 RMS 为 11.83 mm。

上面的配对实验固定使用已有的 `exact_gradient_L0`。该轨迹由支持完整梯度的历史 `rescue_runtime` 生成。本文开头的复现命令从初始状态生成同一新版 CUDA 包下的完整层级。

[两种积分器和同一 L0 的同步对照](../output/bunny_velpro_diagnosis/L1_paired/comparison.mp4) · [仅改变速度传递的对照](../output/bunny_velpro_diagnosis/velocity_comparison/summary.json)

恒等网格回放中，使用仅速度接口目标文件不同的两个 CUDA 构建，400 步的最大 RMS 偏差从 24.43 mm 下降到 0.64 mm，末帧从 12.10 mm 下降到 0.14 mm。对照构建命令为 `scripts/build_flag_l3_fix.py --variant bunny_control`，应在构建 bunny 后执行。对真实 L0→L1，在历史粗轨迹上只修复摩擦使 VelPro 末帧偏差从 121.11 mm 下降到 33.07 mm。在完整 PSQ 梯度生成的粗轨迹上修复摩擦后，VelPro 的末帧偏差为 7.36 mm，原版 semi 为 0.013 mm。L0、VelPro L1、semi L1 的全部布料持续进入离地 1 cm 内的时刻依次为 2.06 s、2.07 s、2.06 s。全程偏差和末帧偏差应分别查看。

[摩擦修复前后的恒等回放](../output/bunny_velpro_diagnosis/identity_comparison.json) · [四组 VelPro 对照](../output/bunny_velpro_diagnosis/solver_comparison.png) · [修复后的 L1 同步动画](../output/bunny_velpro_diagnosis/contact_fixed_L1/comparison.mp4)

### 仅修正摩擦接口与弹性梯度时的八层记录

共同的初始布料从 1,443 个顶点加密到 90,000 个顶点，两种方法共用同一条 L0 轨迹，一致性权重均为零。测量逐帧对齐时钟和材料坐标，以静止布料面积分配顶点质量，再计算位置差向量长度的加权均方根。图像使用同一相机。实验采用本地重建的资产、材料与无摩擦地面，布－兔和自接触摩擦系数均为 0.3。原生接触后端按照网格的平均顶点质量限制接触屏障刚度，日志记录的有效值从 L0 的 2.90×10⁷ 变为 L7 的 1.43×10⁶。该系数属于接触求解。

| 方法与已测范围 | 相对 L0 的峰值 RMS | 4 秒时相对 L0 的 RMS | 最大顶点速度 |
|---|---:|---:|---:|
| VelPro，L7 | 88.09 mm（1.80 s） | 64.85 mm | 5.91 m/s |
| 原版 PD semi，L6 | 32.07 mm（2.13 s） | 0.30 mm | 21.77 m/s |

VelPro 各层都经历落下、披覆、滑离兔子并落地。各层均完成 400 步 Newton 收敛，每层的 401 个保存状态均通过静态相交检查。最细层滑离时刻和褶皱仍有可见差别，4 秒时 99.69% 布料质量在离地 1 cm 内。原版 PD 在 L1 的全程峰值只有 2.72 mm，到 L6 时冲击阶段的速度尖峰已经明显。L7 在一次连续运行的前 210 步中出现 31.03 m/s 的局部速度，时刻为 0.47 s。原论文团队也在 PD++ 的八层 bunny 对照中报告了 semi 的非预期变形与帧间跳动，见图 11、12。

[两种积分器同帧对照](../output/bunny_velpro_diagnosis/methods_comparison.png) · [VelPro 八层动画](../output/bunny_velpro_diagnosis/friction_velpro/comparison.mp4) · [原版 PD 前七层动画](../output/bunny_velpro_diagnosis/friction_semi_serial/comparison.mp4) · [PD++ 论文](../papers/3731202.pdf)

按接触激活距离 1 mm 和两表面厚度之和 0.07 mm，独立检查每个保存状态的布－兔点面、边边等接触对。VelPro 的八层首次进入激活距离都在 0.14 s。L0 至 L7 最后一次检测到布－兔接触的时刻依次为 2.03、2.03、2.02、1.96、1.88、1.83、1.76、1.72 s。最细层提前 0.31 s 脱离，单靠比较 4 秒末帧会漏掉这一差异。[逐帧接触记录](../output/bunny_velpro_diagnosis/contact_times.json)

原版 PD 的首次串行 L7 在接受 210 步后遇到 Windows GPU 驱动重置。使用 MAS 预条件器诊断时，PCG 另一次在第 43 步报告负搜索曲率。使用补偿求和的诊断构建在第 30 步导出矩阵；用 80 位十进制精度重算，归一化搜索方向的二次型为 −0.0931。该诊断构建在有界对角修正的第 4 次尝试时主动停止。使用原 DLL 从 2.08 s 无活动摩擦的位置重算，首个重叠帧的最大顶点偏差为 2.43 mm，超过预先设定的 0.50 mm 复核阈值，故未拼接两段轨迹。

[原版 PD 全八层 2.10 秒动画](../output/bunny_velpro_diagnosis/friction_semi_serial/accepted_prefix_210/comparison.mp4) · [L7 的前 210 步审计](../output/bunny_velpro_diagnosis/friction_semi_serial/L7/audit.json) · [独立矩阵复核](../output/bunny_velpro_diagnosis/semi_L7_stable/L7/matrix_analysis.json) · [重叠帧检验](../output/bunny_velpro_diagnosis/semi_L7_contact_free/L7/run.json)

## 带地面 bunny 的原版 PD 对照

原生 CUDA 入口用 `--method semi` 运行 Zhang 等（2024）式（5）。在细层第 n 步，以相邻粗层位置的 PSQ 延拓为起点，加上粗层上一时间步位移的线性延拓。BDF1 通过速度接口接收这个惯性预测，接触求解从细层上一帧的可行位置开始。该模式使用零一致性权重，并检查输入轨迹来自同一方法的无惩罚父层。历史实验可通过 `--method velpro` 复跑。

使用已通过检查的 L0、同一套网格和求解设置，从仓库根目录运行以下命令。每层模拟 400 步（4 秒），`--last` 指定最后一层，输出选择新目录。

```powershell
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --method semi --lower output/bunny_ground_L0_20261118/full --output output/bunny_pd2024_repeat --native-package output/flag_L3_fix_20261118/rescue_runtime/src --last 7
```

[同一 L0 下的无惩罚 L1 逐帧对照](../output/bunny_pd2024_w0/L1_comparison/comparison.mp4)使用相同相机和时钟。[逐层误差](../output/bunny_pd2024_w0/level_errors.json)按静止布料面积对顶点位置差加权，单位为毫米。`parent` 比较相邻层，`L0` 比较逐级延拓至当前层的粗预览。`floor_pct` 是末帧高度小于 1 厘米的布料质量比例。

### 八层 full-diagonal 对照

`--method full` 使用 PD++（2025）第 5.4.1 节的式（7）和式（8）。L1 与原版 PD 共用同一公式，可以直接复用合格的零惩罚 L1。L2 起，粗层当前帧与再粗一层的上一帧构成对角位移。位置采用非线性 PSQ 延拓，位移采用冻结偏移的线性部分。运行记录包含两级输入轨迹的 SHA-256。

```powershell
.venv/Scripts/python.exe scripts/continue_bunny_ground.py --method full --lower output/bunny_pd2024_w0/L1 --output output/bunny_full_repeat --last 7
.venv/Scripts/python.exe scripts/measure_bunny_chain.py output/bunny_full_repeat --movie
```

本地重建场景中，八层 full-diagonal 已完成每层 400 步。L0 复用已审计的粗预览，L1 复用式（8）与 semi 相同的无惩罚轨迹。L7 的 9 万个顶点参与动态求解，400 步均达到 Newton 收敛判据。各层共 3208 个保存状态通过静态相交检查；粗层另有按非下降判据接受的预览步。4 秒时各层布料最高点介于 0.5015 m 与 0.5128 m 之间。对照用的原版 semi 在八层设置下出现动态偏差，L6 第 115 步用满 300 次迭代而被拒绝。

[八层同步动画](../output/bunny_full_w0/comparison.mp4) · [末帧同视角](../output/bunny_full_w0/comparison_400.png) · [逐层质量加权 RMS](../output/bunny_full_w0/level_errors.json) · [semi 第 114 帧](../output/bunny_pd2024_w0/comparison_114.png)


2026-11-11：用户选择论文 Cloth Drop Over Bunny，替代 Scene527 作为首个复现验收场景。叠衣实验保留在 `output/pdpp_shirt_first_fold`，已安全停在粗层 1045/1540 步，监测暂停。

## 论文依据和当前输入边界

[PD++ 正文图 11–12](https://pcs-sim.github.io/pd%2B%2B/PD%2B%2B_main.pdf#page=12) 给出：布料厚度 0.07 mm、拉伸上限 6.8%（最大主拉伸 1.068）、摩擦系数 0.3、时间步 0.01 s、8 层、约 1.5K→90K 顶点。比较目标是 Semi-diagonal 的不连续跳动、Full-diagonal 的残留抖动以及 VelPro 的连续细节；布料最终从低摩擦兔子上滑落。

已查看项目页、作者主页和项目网站仓库公开入口，尚未取得可确认属于该图的原始场景包。没有把替代输入称为作者原始输入：

| 输入 | 当前设置及来源 |
|---|---|
| 兔子 | 仓库内 C-IPC `bunny3K.obj`，3135 顶点/6266 面；保持拓扑，Y-up 旋转成 Z-up，高度 0.5 m；非已认证 PD++ 模型 |
| 布料 | 新建 1 m 正方形，中心 XY=(0.18,0)，高于兔子最高点 0.1 m，零初速，无固定点、无地面 |
| 材料 | E=30000 Pa、ν=0.3、ρ=200 kg/m³；是公开记录的本地假设，未找回作者值 |
| 接触 | dmin=0.00007 m、dhat=0.001 m、刚度 10000；为本地假设 |
| 层级 | 概率 QEM + 内在联合参数化，保存相邻映射、最细代理锚点、非零 PSQ 偏移 |
| 初始化 | 水平 warm start；当前层 Hessian；非线性 PSQ line search；RMS 容差 1e-3 |

`configs/bunny_debug.json` 和资产 manifest 区分论文明确参数与本地假设。仍缺作者网格/摆放/材料/接触参数及动画帧和物理时间对应，不能凭调试场景通过就宣布原论文复现。

## 执行阶段和验收

1. **短程接触检查**：8 层顶点数 60→82→113→155→218→300→422→625，每层 25 步。检查自由落体、首次接触、每层 Newton 与预览早停、应变上限和全部接受状态的几何。不是论文规模。
2. **同条件比较**：固定同一输入和容差比较 Semi、Full、VelPro、Direct，先覆盖同一短程，再扩大到滑落阶段。方法失败保留失败原因和最后接受步，不为某方法单独调参数。Direct 在各层使用本层弹性能；正式精度/耗时对照重点是同一最细网格。不能用粗层不同弹性代理的耗时声称加速。
3. **规模和原始输入对齐**：用户已明确要求执行完整论文规模实验；已生成约 1.5K→90K 的 8 层资产，正式配置见下节。`prepare_bunny --profile paper-size` 仅匹配端点规模，不代表已取得作者资产。完整跑完不等于原论文复现成功。
4. **验收**：展示真实轨迹，同一时刻/视角的比较；报告全部层级接受步数、严格收敛与预览早停、最大主拉伸、接触可行性、连续性和一致性。性能需相同最细网格 Direct 对照、隔离运行、重复计时及单列预处理/层级总成本。通过后再返回叠衣。

粗层预览早停仅按论文允许的分支单列，不算 Newton 收敛。最细层和 Direct 必须严格收敛。不修改物理参数或放宽容差来掩盖失败。

## 命令

从项目根目录使用 `.venv\Scripts\python.exe`：

```powershell
.venv\Scripts\python.exe -m progressive.prepare_bunny --output output/bunny_debug_assets
.venv\Scripts\python.exe -m progressive.run --config configs/bunny_debug.json --steps 25 --methods velpro --output output/bunny_contact_pilot
.venv\Scripts\python.exe -m progressive.audit_bunny output/bunny_contact_pilot
.venv\Scripts\python.exe scripts/audit_progressive_runs.py output/bunny_contact_pilot
.venv\Scripts\python.exe -m progressive.report output/bunny_contact_pilot
```

输出/资产目录非空即拒绝覆盖；已有运行不要重复执行上述创建命令。JSON 是默认值，显式 CLI 参数优先，实际参数和配置内容/哈希均入 experiment.json。续跑检查 bunny、网格、映射与锚点哈希。

失败时对审计和报告添加 `--accepted-prefix`，报告另存并显式标注未完成；只有粗层时再添加报告 `--all-levels`。审计只读取 `completed_steps` 内的轨迹，包括初态，不读取预分配尾部。几何复查逐帧验证交叉、厚度和重新计算的主拉伸；不把离散检查冒充独立的连续 CCD 证明。

新增 2 项集成测试覆盖自由落体无 pin、固定兔子无驱动自由度、JSON 参数覆盖、Poisson 参数传递、续跑资产变化拒绝，以及接受前缀之外 NaN 尾部不得进入审计。旧 56 项回归和新增 2 项测试均已通过。

## 已取得的短测结果

`output/bunny_contact_pilot`：VelPro，8 层分别完成 25/25 步（物理时间 0.25 s），200 个求解全部达到设定 Newton 判据，预览早停 0。208 个接受状态（含各层初态）通过逐帧几何和重算应变审计。全层最大主拉伸 1.052381 < 1.068；最细层峰值 1.049400。累计求解 342.00 s，其中最粗层 216.94 s、最细层 24.45 s。最粗层最大单步 166 次 Newton，最细层最大 31 次。

真实轨迹显示布料自由落下后与兔子耳部接触并开始披覆，没有覆盖完整滑落过程。动画：`output/bunny_contact_pilot/report/comparison.gif`，审计：`audit_bunny.json`、`audit_solver.json`。初始设置和 0.10、0.14、0.20、0.25 s 采样见 `report/trajectory_samples.png`。

最细层连续性 RMS 图有约 109 mm 峰值；对应求解残差 RMS 约 219 mm，而 Newton 速度增量 RMS 为 0.000915 m/s，满足 0.001 m/s 判据。Newton 增量收敛不代表所有力残差度量接近零；这张绝对误差图不能单独证明 VelPro 连续性优于其他方法。需要匹配 Direct 残差、归一化指标和真实轨迹比较。

这些计时包含与测试/诊断重叠的时段，只能用来评估调试成本，不作性能或加速声明。`output/bunny_contact_controls` 的 Semi、Full、Direct 也已全部完成 25 步和几何审计，各层均 Newton 收敛、无预览早停；Semi/Full 共用粗层，且该粗层与 VelPro 粗层轨迹逐元素相同。最细层最大主拉伸：Semi 1.059903、Full 1.049653、Direct 1.051168。四方法短程动画位于 `output/bunny_contact_pilot/report_comparison/comparison.gif`。

## 完整论文规模实验（用户明确授权）

正式配置：`configs/bunny_paper_scale.json`。正式输出：`output/bunny_paper_scale`。由 `progressive.bunny_suite` 串行运行四种方法，某一种方法失败也会保留失败并继续其余对照；每个方法输出独立完整实验。全套完成以 `suite.json` 的 `finished`、`completed` 和各 job 状态共同判断，单一层或方法完成不算全套完成。

| 项目 | 设置 |
|---|---|
| 顶点数 | 1443、2663、4834、8608、15651、28185、50268、90000 |
| 最细面数 | 178802 |
| 方法顺序 | VelPro、Full、Semi、Direct；全部使用同一组资产、材料和求解设置 |
| 时间 | dt=0.01 s，每层 400 步，共 4 s；**本地明确选定的完整实验时段，非已确认作者原始时长** |
| Newton | RMS 速度增量阈值 1e-4 m/s；所有方法统一 1000 次迭代上限；不自动放宽阈值 |
| 阈值依据 | 论文 §8.2.2 定量连续性比较采用紧阈值 1e-4；此处对 8 层实验也统一使用该阈值，并不声称找回作者图 11 的完整配置 |
| 并行 | 单个方法运行，IPC/BLAS 8 线程；不并行跑多个方法争抢内存 |
| 检查点 | 每个接受步落盘，失败日志独立保留；不读预分配尾部 |
| 计时 | 各层物理计算、单方法墙钟、审计和渲染分开记录；本轮每方法一次，不宣称统计加速 |

资产位于 `output/bunny_paper_assets`。原生网格构建 10.88 s（不含 Python 历史验证和其他预处理），8 层全部通过无交叉、厚度可行和 PSQ 初态重建检查。初态重建最大误差 < 7e-18 m。存储空间检查约 266 GB 可用，机器约 32 GB RAM。最细层单独使用真实 90K 网格做 2 步资源预检（`output/bunny_paper_finest_probe`），已完成且审计通过：两步各 2 次 Newton、累计求解 144.63 s、最大主拉伸约 1.0，无活动接触。预检不代替正式 400 步实验，也不据自由落体速度估算接触阶段耗时。

正式套件已于 2026-11-11 22:13:26（Asia/Shanghai）在隐藏后台启动，入口和日志见 `output/bunny_paper_scale_launcher.json`、`output/bunny_paper_scale_launcher.log`。已核对套件与 VelPro 子进程完整命令行，VelPro 最粗层已开始接受时间步；Full/Semi/Direct 在队列中等待。原有监测已切换到此正式套件，正常运行保持安静，失败或全套结束时交付；不会再以短测完成触发正式实验的完成通知。

```powershell
.venv\Scripts\python.exe -m progressive.bunny_suite --config configs/bunny_paper_scale.json --output output/bunny_paper_scale
```

套件保存输入及实现哈希，每个方法开始前复核；若中途代码或输入变化则停止启动后续方法，避免混合不同实验。各方法结束自动进行求解日志审计和全部接受状态几何审计，四方法均通过才生成最细层同步对照报告。失败和未完成结果不冒充完整对照。

4 s 结束后另行检查实际接触、披覆和是否滑落。若未滑落，报告现有重建输入未复现该现象；不能通过改变摆放、摩擦或材料来把新结果冒充这一轮。如果需要延长观测，应使用新输出、相同物理参数及明确的延长时段，保留本轮原始记录。


## 正式实验失败记录：VelPro 公共粗层

2026-11-12 02:44（Asia/Shanghai）监测确认：VelPro 任务已失败，Full 正在运行，其余排队中。失败发生在 L0（1443 顶点）的第 57 次时间步求解（日志 step=56），达到 1000 次 Newton 上限。最后接受状态为 56/400 步、物理时间 0.56 s；L1–L7 均未启动。该层是渐进方法共用的粗层积分，尚不能将此失败归因于 VelPro 跨层速度更新。

接受的 56 步中，22 步 Newton 收敛、34 步为论文允许但并非 Newton 收敛的预览早停。最大主拉伸 1.06272354 < 1.068，含初态的 57 帧通过几何/厚度/应变审计。失败迭代最终 RMS 速度增量约 0.00336341 m/s（阈值 0.0001），梯度无穷范数约 12.6826；最后线搜索 alpha 约 1.49e-8、回溯 26 次。这说明求解停滞，不能据此确认更深的根因。

接受步骤累计求解 2491.55 s（41.53 min），失败步骤额外约 11050.97 s（184.18 min），累计计算 13542.52 s，任务墙钟 13568.85 s（3.77 h）；审计另外 4.38 s。动画标题的 compute 包含失败步骤耗时，但帧只含接受状态。

部分动画 `output/bunny_paper_scale/velpro/report_prefix/comparison.gif` 已通过 --accepted-prefix --all-levels 生成并检查，显式标注 INCOMPLETE。轨迹显示落下与披覆，尚未证明完整滑落；不是 90K 最细层结果。审计在 `output/bunny_paper_scale/velpro/audit_bunny_prefix.json`。已通知此次失败，其他对照继续，未修改运行参数、代码或容差。


## 2026-11-12 13:06 监测：旧套件中断

套件与 Full 的完整命令行进程均已不存在，但 suite.json 仍为 running。Full 最后接受 L0 400/400、L1 381/400，L1 检查点最后写入 12:21:20（Asia/Shanghai）。Full/launcher 日志没有异常栈或退出码，无法确定退出原因。Semi/Direct 未开始。未重启或修改原实验。已生成 full/report_prefix/comparison.gif（--accepted-prefix --all-levels），同步展示各层前381步，标注 INCOMPLETE；不是90K结果。最终全轨迹固定视角下物体极小，不能据此确认成功披覆滑落。原生C-IPC90K新实验独立正常运行，当前优先交付该实验。
