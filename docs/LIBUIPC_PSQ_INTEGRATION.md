# libuipc 接入试验（2026-11-13）

2026-11-16新增：[粗层非下降停止接入](BUNNY_COARSE_PREVIEW.md)。Bunny L0短测完成25步，其中18步严格收敛、7步预览停止、0步强制接受；关闭预览仍在第18步严格失败。完整多层迁移与最细层验证仍未完成。以下保留历史阶段记录。

2026-11-15新增：用户明确要求的“未收敛仍接受更新”独立实验已完成40步，其中19步严格收敛、21步强制接受；见 `LIBUIPC_FORCED_UPDATES.md`。它不改变下文严格验收未通过的结论。

## 当前结论

2026-11-14 更新：**PSQ 已接入独立源码版 libuipc 的原生 Newton/线搜索，并加入当前层硬拉伸屏障；恒等验证通过，但真实短接触验证只完成17/25步，第18步线搜索失败。完整迁移验收尚未通过，不能称为论文复现成功。** 未修改已安装 libuipc、旧 Python 求解器、实验资产或历史结果。下方2026-11-13的构建/预检记录保留为历史；最新实现和结果见文末。

本机 pyuipc 0.0.28、RTX 3060 Ti 8 GB、驱动610.88通过原生doctor CUDA检查。已安装wheel使用CUDA12.8；新增的独立算子检查DLL使用本机CUDA12.6 / MSVC14.34编译（sm_86），不加载或替换libuipc后端，不代表二者ABI已集成。

## 唯一短接触运行

入口：`scripts/probe_libuipc_bunny.py`；正式短验证目录 `output/libuipc_bunny_L0_contact_v2`。使用原Bunny L0的1443个顶点和原固定兔子，25步、dt=.01，物理时间0.25秒。Newton、PCG、CCD和回溯全部由原生libuipc执行，Python只构造场景、调用advance/retrieve和保存接受状态。

- 半隐式提前终止关闭，strict_mode开启，固定每轴速度阈值1e-3（位移阈值1e-5），不是RMS；最多100次Newton、32次回溯。运行器最多40步、步骤之间检查180秒预算。
- 25/25步报告原生收敛，0步预览早停；单步最多50次Newton，最后一次成功判据的最大位移残差9.944662709284352e-6。第14步起出现接触候选，最终图可见布料已披覆耳部。
- 墙钟3.829787秒，初始化0.125085秒，advance+retrieve累计3.688474秒；不与90K原生C-IPC作加速比较。
- **最大主拉伸1.3626069819，超过原C-IPC限制1.068。** 这次使用原生NeoHookeanShell+DiscreteShellBending，没有硬拉伸屏障，故仅通过接入验证，不通过原物理模型验收。
- full thickness为7e-5，libuipc存储单侧半径3.5e-5，材料E=30000、nu=.3、rho=200。碰撞刚度由libuipc自适应处理，日志记录初值1e9被限制至29032287.397566944。与旧固定kappa不同，不能宣称同物理模型A/B。
- 保留布料自接触和布料/兔子接触，按原静态障碍物过滤语义关闭兔子内部自接触；没有关闭sanity检查。
- 两次前置接线失败保留：`libuipc_bunny_L0_probe`在初始化时因未过滤固定兔子内部接触而拒绝；`libuipc_bunny_L0_contact`完成原生第1步后因Python几何slot解包错误而停止，没有保存为接受轨迹。修正接口后进行了上述唯一25步运行。未调物理参数使失败通过。

产物：`contact.gif`、`final.png`、`metrics.png`、`audit.json`；完整原生日志在同名`.log`。报告脚本 `scripts/report_libuipc_probe.py` 只读取接受前缀。

## 已实现的 PSQ CUDA 部分

- `scripts/export_psq_libuipc.py` 导出L0/L1到90K的直接CSR映射、锚三角形、gamma、tangent、inv_dm；并保存类型、能量缩放与代理角色约定。导出后位置回读误差分别1.11e-16米和0。
- `native/libuipc/psq_kernels.cuh`：float64 CUDA计算 `P(x)=Ux+n gamma+R(F)t`，旋转使用3×2极分解的2×2 Gram矩阵闭式逆平方根；退化输入返回无效标记，不能作为接受状态。另实现冻结偏移梯度 `U.T @ g_fine`。
- `psq_operator_check.cu`仅为一次性host-buffer验证壳，不是新的Newton求解器，也不是已注册的libuipc插件。实际后端接入应直接使用device buffer及已有CUDA stream，不能每次迭代经Python搬运90K数据。
- 仅做一次GPU对照：使用上面25步运行最后的已接受L0形状，生成90K代理。相对原SVD实现位置最大误差3.33e-16米；固定种子合成向量的U.T映射最大误差3.20e-14。合成向量不代表物理力。GPU调用含初始化和传输约0.10秒；不是性能基准。结果在 `output/libuipc_psq_bridge_L0/cuda_check.json`。

## 原生后端接入位置与剩余工作

核对版本：上游v0.0.28源码快照 `9c5710ee4b716502e3757e7312360e5ee68cb79f`，已下载相关源文件并逐文件校验Git blob SHA；位于 `native/libuipc/reference`。它们是只读参照，不是构建完成的本地libuipc。此前git clone连接未完成，已终止该下载；不影响任何仿真。

1. `src/pybind/pyuipc/constitution/finite_element_constitution.cpp`仅绑定基类，没有Python能量/导数虚函数trampoline；不能把当前Python evaluate直接传给wheel的Newton。
2. 后端 `FiniteElementElastics` 的 `compute_energy` 和 `assemble` 分别通向原生线搜索及全局线性系统。应新增PSQ原生constitution/reporter，接入代理弹性（膜+弯曲）能量、U.T梯度和当前层近似弹性Hessian。
3. libuipc膜能量按 `dt^2 * E` 组装；PSQ新增能量和导数必须同样缩放。当前层质量、接触与摩擦由原生系统负责；90K代理只用于弹性，不能再注册为带质量/接触的物体。避免同时加入原生粗层弹性梯度和PSQ弹性梯度。
4. 在Newton组装位置计算当前P(x)与冻结偏移导数；在线搜索每个trial位置重新计算P。直接映射U有约110万至119万非零，不能假设每行只有3个非零来硬套普通三角形stencil。
5. 增加当前层C-IPC硬拉伸屏障及其可行步长限制。原生NeoHookeanShell没有该屏障，Baraff-Witkin的软过伸长放大也不能冒充它。
6. 原生 `engine/advance_ipc.cu` 使用能量不增的回溯、碰撞可行步和自身终止规则；PSQ近似方向仍可能对非线性能量不下降，不能认为换库必然消除停滞。后续接入需明确论文粗层预览终止与最细层严格收敛策略，不复制旧Python任意阈值。

以上是2026-11-13阶段计划；2026-11-14已实施接入并进行下述验证。在完整验收通过前，不启动400步或8层实验。

## 源码构建接续状态

用户已授权依次完成原生PSQ接入、硬拉伸屏障、恒等和短接触验证。已将上述固定提交完整下载到 `external/libuipc-psq-src`，与现有安装和旁边的用户libuipc仓库隔离。没有启动新增仿真。

`native/libuipc/prepare_native_build.py --configure --build` 是可重现的准备与构建入口；源代码构建仍不等于PSQ插件完成。其 `build_support` 保存XMake/CMake构建适配：使用SHA256核验的pybind11 v3.0.4头文件，避免依赖安装器误选Python3.10；显式选择Python3.11 SDK；下载仅在子进程中使用Windows已有代理，不更改全局代理设置。关闭examples、tests、benchmarks，只构建库和Python模块；保留上游禁用ccache约定。

初次构建在Python.h缺失时停止，已修正SDK路径并继续。当前构建日志为 `output/libuipc_psq_build_sdk.log`，执行会话34748。后续先核对该会话及完整xmake进程命令行，不重复启动相同构建。PSQ能量组装、硬屏障与两项动力学验证均尚未完成，不能用之前25步普通布料运行充当验收。

## 2026-11-14：原生接入与短验证结果

### 已接上的内容

- 独立源码 `external/libuipc-psq-src`，固定提交 `9c5710ee4b716502e3757e7312360e5ee68cb79f`，模块自报版本0.9.0；实际导入路径和四个核心DLL路径写入每个run.json，全部位于该源码的build/python/src/uipc/_native。没有回落到已安装wheel。
- `native/libuipc/psq_proxy.cu`在GPU常驻存储CSR、偏移、90K代理位置和梯度，调用原生NeoHookean膜能量/梯度与DiscreteShellBending能量/梯度，回传dt² U.T g。代理不增加质量、碰撞网格或独立求解自由度。
- 当前层膜/弯曲的弹性能和梯度被替换，Hessian仍用当前层原生近似；每次线搜索trial都重算P(x)。惯性、接触、摩擦、PCG、Newton和回溯仍走libuipc。
- `psq_strain.cuh`加入C-IPC形式的奇异值屏障及解析梯度/Hessian。阈值1.068、激活1.034、固定刚度1000，均从二进制bundle读取。基于Weyl界限制当前层拉伸和秩退化的步长，然后继续原生CCD。**尚无C-IPC自适应拉伸刚度策略；不能称为完整C-IPC移植。**
- 本原型限定单个均匀NeoHookean布料和额外固定FEM障碍物。按实际FEM顶点偏移核对坐标、拓扑、材料、厚度，不假定布料排在第一个。没有实现通用多物体、层级间状态传递、安全细化或论文预览早停策略。

### 本次修正的接线/数值问题

1. 构建脚本曾把Python包复制到多嵌套一层的目录，导致误导入旧wheel。已修正复制路径，并加入强制模块/DLL路径检查；依赖使用匹配构建的spdlog、Intel TBB和CUDA12.6。
2. libuipc按constitution重排FEM顶点，固定兔子在前，布料offset=3135。PSQ读取与梯度写回都已使用核对后的偏移。
3. 原生NH存的是逆静止度量 `(Dm.T Dm)^-1`，不是逆形变基 `Dm^-1`。拉伸屏障现在取该SPD矩阵的正平方根再形成F；PSQ延拓本身仍使用导出的inv_dm。
4. 90K三角形上，原生生成的NH多项式用绝对坐标会发生相消；第一次接触试验在自由落体第2步就因虚假弹性残差失败。代理现用每个三角形的局部原点调用同一原生能量/导数公式，利用平移不变性避免相消。没有修改材料、阈值、线搜索条件或加入完整偏移Jacobian。

### 仅有的两类验收

**恒等对照**：L0的1443顶点，U=I、零偏移，两步，初始面内拉伸1.045以激活屏障。原生参考与PSQ路径都严格收敛，最大坐标差2.1517565507167546e-11米（预设判据1e-9米）。参考 `identity_reference_v5`，最终代理 `identity_proxy_local`，结构化结果 `output/libuipc_psq_native/identity_comparison_local.json`。早期加载/初始化失败、矩阵约定失败记录均保留，不计成功仿真。

**真实短接触**：L0 1443自由布料顶点 → 90000弹性代理顶点；178802代理三角形、267605代理弯曲铰链；dt=.01，请求25步。最终目录 `output/libuipc_psq_native/L0_contact_local`：

| 项目 | 结果 |
|---|---:|
| 接受步数 | 17 / 25，未完成 |
| 原生严格收敛 | 17，预览早停0 |
| 有效物理时段 | 0.17秒 |
| 接受步最大Newton次数 | 27 |
| 首次接触候选 | 第14步 |
| 接受前缀最大主拉伸 | 1.0499659813657338 < 1.068 |
| 初始化 | 0.22154秒 |
| 17个接受步advance+retrieve累计 | 3.47347秒 |
| 整个子进程墙钟（含失败/退出） | 8.15002秒 |

真实动画显示布料下落到兔耳附近，出现初始接触；不能据此宣称完整披覆或滑落。动画和图表仅包含初态及17个接受状态，标题明确标注INCOMPLETE。没有90K独立动力学或多层全流程成功结论，也没有性能加速结论。

### 第18步究竟失败在哪里

原生日志：Frame=18、Newton=24（零基下标），最大位移增量4.094742198994534e-5米，大于原生阈值1e-5米，即最大轴速度增量约0.00409474 m/s > 0.001 m/s。32次回溯未满足能量不增，最后alpha=4.656612873077393e-10。E0=8.686953e-8，最终相对能量增加9.141e-16；最后几次已到浮点分辨率附近，但迭代方向仍不满足Newton阈值，不能按严格收敛接受。

CCD/CFL最终系数均为1；日志中的停止条件是线搜索失败，不是内存耗尽或此次观察到的拉伸越界。开发构建遇到严格异常会abort（Windows进程码3221226505），父运行器已记录失败并保留接受前缀。失败第18帧未进入动画。

这里仍有未解决的算法接口问题：冻结偏移的梯度与线搜索每次重算偏移的完整非线性能量并不保证构成下降方向；本场景切向静止残差最大约0.01633米，不能当作恒等/零偏移场景。**现有日志只证明了停滞位置，尚不能排除所有实现或模型问题，更不能仅凭此判定根因已完全查清。** 后续应围绕该迭代点核对搜索方向与能量变化，并单独实现/验收论文规定的粗层预览终止语义；不能把线搜索失败直接改名为Newton成功。

论文原文位置：本地 `papers/psq-combined.pdf` §4.5及Algorithm 1（文本 `tmp/pdfs/psq.txt`）区分粗层预览与最细层收敛。本轮没有为凑25步而启用早停、放宽容差或重跑完整Bunny。

### 重现入口与产物

以下命令需从项目根目录执行，输出必须使用新目录：

```powershell
.venv/Scripts/python.exe native/libuipc/prepare_native_build.py --with-psq --build
.venv/Scripts/python.exe scripts/run_psq_probe.py --bundle output/libuipc_psq_native/L0_to_90k.bin --steps 25 --output output/NEW_PROBE_DIRECTORY
.venv/Scripts/python.exe scripts/report_libuipc_probe.py output/NEW_PROBE_DIRECTORY
```

最终编译日志：`output/libuipc_psq_integration_build9.log`，build ok。
汇总：`output/libuipc_psq_native/integration_result.json`。
短接触审计/图像：`L0_contact_local/audit.json`、`contact.gif`、`metrics.png`、`final.png`。
失败原生日志：`output/libuipc_psq_native/L0_contact_local.log`。

### 失败点的针对性方向核对

仅额外回放到第18步一次（7.65秒进程墙钟），未扩大场景或时段，目录 `output/libuipc_psq_native/L0_direction_audit`。新增只在失败后显式启用的诊断开关，不改变接受规则；诊断试探点不计入接受轨迹。对第18步第24次Newton迭代，读取原生线性系统梯度与解向量，计算预测斜率g·p，并对同一方向用完整原生能量作正向差分：

| alpha | 原生组装梯度预测g·p | 完整能量实际差分斜率 |
|---:|---:|---:|
| 1e-3 | -7.73158455e-13 | +4.92584070e-15 |
| 1e-4 | -7.73158455e-13 | +4.85958684e-15 |
| 1e-5 | -7.73158455e-13 | +4.86249851e-15 |

三个尺度都显示方向不是完整非线性能量的局部下降方向；其中alpha=1e-3的能量增加约4.93e-18，明显大于末次回溯的舍入量级。这明确了本次停滞的直接机制，而不是只看到alpha小就归咎于浮点数。冻结偏移梯度与重算偏移能量的差别是首要解释；该检查本身没有逐项证明所有能量/导数或接触实现均无误，不能把它扩大为排除一切其他原因。

结构化方向证据：`output/libuipc_psq_native/direction_audit.json`。当前构建日志 `output/libuipc_psq_integration_build11.log`，仅比验收构建增加显式失败诊断；普通运行不启用它。最终汇总同时保留原验收DLL哈希及当前诊断构建哈希。
