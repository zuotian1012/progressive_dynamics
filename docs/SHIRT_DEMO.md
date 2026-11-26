# Scene527：本机夹爪折衣与渐进分辨率实验

本文记录的是早期原生 GPU 演示。新的独立论文求解器及其验证见 [PAPER_FRAMEWORK.md](PAPER_FRAMEWORK.md)；不能将本文的视频、耗时算作新算法的结果。

更新：2026-11-06。运行设备为 Windows、RTX 3060 Ti 8GB。

已经完成第一段动作的三组真实物理仿真、同步动画和可交互回放。夹爪通过实体碰撞和摩擦抓取，衣服没有固定点、软绑定或目标位置牵引。当前范围是原轨迹的前 740 帧，包含抓取、抬升、翻折、释放；不是完整 73 秒三折流程的复现。

## 直接查看

- [三组同步动画](../output/shirt_soft_fold/report/shirt_demo.gif)：从左到右为粗层、渐进细层、独立细层。动画使用统一物理时间，播放速度不是计算速度。
- [带网格的末帧对照](../output/shirt_soft_fold/report/comparison.png)。
- [完整指标](../output/shirt_soft_fold/comparison_metrics.json)。
- 交互回放可旋转视角、拖动时间轴、显示网格，并调整播放速度：

```powershell
.venv/Scripts/python -m shirt_demo.viewer output/shirt_soft_fold/coarse output/shirt_soft_fold/progressive_fine output/shirt_soft_fold/direct_fine
```

## 本次实测结果

三组使用相同动作、时间步和材料设置。每组为 1,540 个物理步：60 步初始沉降 + 740 个控制帧 × 2 子步；时间步 1/120 秒，总物理时间 12.833 秒。

| 运行 | 顶点 / 三角面 | 求解耗时 | 物理步/秒 | 中位步耗时 | P95 步耗时 |
|---|---:|---:|---:|---:|---:|
| 粗层预览 | 807 / 1,536 | 292.88 s | 5.26 | 118.19 ms | 615.27 ms |
| 渐进细化，使用已有粗层 | 3,152 / 6,144 | 360.54 s | 4.27 | 184.79 ms | 540.58 ms |
| 直接细网格 | 3,152 / 6,144 | 589.25 s | 2.61 | 250.40 ms | 1,191.06 ms |

粗层预览比直接细网格快 **2.01 倍**。已有粗层后，细化阶段比直接细网格快 **1.63 倍**。但是，从零开始的粗层 + 细化累计耗时是 **653.43 秒**，比直接细网格多约 **10.9%**。因此目前支持“更快获得预览”，尚不支持“单次最终结果总成本更低”。这些是每组一次运行的测量，不是重复实验的统计结论。

计时包括跨层速度计算、IPC 状态传递、原生推进和同步取回；不包括场景构建、机器人正运动学、磁盘记录或渲染。三组按顺序运行，避免互相竞争 GPU；这是求解阶段吞吐量，不是完整应用端到端耗时。

### 抓取与折叠观察

在闭合结束时，各手选择距离夹爪 TCP 最近的 8 个衣服材料顶点，跟踪到抬升/拉动阶段结束。粗层的左右测量区域分别抬升约 47.6 / 48.7 mm；渐进细层约 46.2 / 44.3 mm。对应夹爪局部坐标中的最大区域均值漂移：粗层约 5.5 / 1.4 mm，渐进细层约 4.6 / 6.6 mm。

这支持“布料随实体夹爪运动”的观察。该指标只是几何跟踪，尚未导出逐接触对的法向力/摩擦力，不能作为严格不滑移的证明。衣服最终仍有较大的隆起和褶皱，尚未完成多折整理及最终压平验收。三组最终平面包围盒面积约为各自沉降后面积的 55.7%、58.7%、57.4%；这是形状变化指标，不等于语义折叠覆盖率。

### 一致性没有显示全面优势

在相同 101 个时间样本上，将粗层位置线性延拓到细层，计算顶点均匀加权 RMS：渐进细层的时间均值约 **24.0 mm**，独立细层约 **17.8 mm**。因此本次渐进版本没有改善这个一致性指标。它也不是论文的质量范数连续性指标；不应将外观相似或更快求解解释为全面优于独立仿真。

## 复用了什么

参考分支固定在提交 `c29cc92f954fa4b803fb5cfe3155d6d9dbd5b088`：
[genesis-ipc-shirt-demo](https://github.com/happy1041/genesis-ipc-shirt-demo/tree/c29cc92f954fa4b803fb5cfe3155d6d9dbd5b088)。本地放在 `external/genesis-ipc-shirt-demo/`。

- 8,000 面短袖 OBJ，保持原坐标与单位约定；本次对其简化并构造嵌套细分网格。
- 双 X5 URDF 的运动学、8 个腕掌/夹片碰撞网格，以及累计关节命令。
- 桌子位置和尺寸、衣服初始姿态、摩擦接触对、第一段动作时间线。
- 正运动学第一帧的左右 TCP 与原动作计划目标匹配到 `1e-10 m` 检查容差。

没有使用原来的 55k checkpoint 冒充新拓扑状态，也没有将原参考视频作为新仿真结果。实际物理只包含原仓库参与 IPC 的 8 个腕掌/夹片刚体；完整机械臂的其余连杆外观未载入。手臂运动由原关节命令的正运动学生成，夹爪由软刚体姿态约束驱动，布料通过接触摩擦响应；不是完整机械臂力矩控制实验。

额外验证了原仓库 **13,767 → 55,068 面**确实是一次中点细分：顶点位置、三角形连接和绕序均匹配，静止形状映射误差为零。映射和原拓扑网格已保存到 `output/shirt_reference_hierarchy/`，可用于后续更大规模实验。8k 与 13k 之间没有假定存在相同关系。

本次小网格的实际层级为 1,536 / 6,144 / 24,576 面，简化器未达到请求的 500 面目标；记录的是实际返回数量。目前只运行前两级。小网格细层是简化衣服的细分，不是原 8k 网格的完全恢复。

## 算法与接触实现

采用本机已安装的 `pyuipc 0.0.28` 原生 CUDA 后端，不依赖 Genesis 或 Linux。细层的 VelPro-inspired 预测为：

```text
v_input(l,n) = P(l-1,l) [x(l-1,n) - x(l-1,n-1)] / h
x_tilde(l,n+1) = x(l,n) + h v_input(l,n) + h² gravity
```

第一步速度为零。细层保留自己当前的位置，由原生 IPC/BDF1 对弹性、自碰撞、桌面及夹爪接触、摩擦共同求解。并非把粗层坐标逐帧复制到细网格。每次仅用状态访问器替换预测所用速度；传回的位置是同一个细层的当前状态，不发生跨层投影或瞬移。

**这是借鉴 Progressive Dynamics++ / VelPro 的可运行工程原型，不是完整论文复现。** 当前使用线性中点延拓、各层自己的离散弹性能，没有 PSQ 非线性延拓、共享最细代理能或跨层一致性惩罚。粗网格膜锁定、接触差异和材料离散差异仍可能改变轨迹；论文的稳定性或性能结论不能直接套用。

材料使用当前原生版本的 Baraff–Witkin 膜能和离散壳弯曲。拉伸 Young 参数为 20,000、独立剪切 Young 参数为 4、Poisson 比 0.49、密度 800、单侧接触半径 0.0001 m、弯曲原始系数 `1e-5`，应变放大参数为 100。这里的 100 不是“10% 硬应变限制”。布—布、布—桌、布—夹爪摩擦分别为 2、1、√2。

原仓库依赖 `pyuipc 0.0.25`，与当前版本的材料参数语义不同；本次参数不声称与原 55k 成片等价。较硬的先行试验保存在 `output/shirt_first_fold/`，未混入正式比较。原生接触刚度会按场景质量调整，实际提示保存在每次运行日志中。

## 已做验证

- 本机 CUDA 世界初始化与步进通过。
- 速度注入的刚体平移/自由落体探针误差约 `1.7e-10 m`；这项测试不覆盖接触时的全部积分语义。
- 13 项现有及新增自动测试通过，覆盖原薄壳基线、细分仿射精确性、机器人正运动学及原生网格数据布局。
- 三组全部 1,540 步完成，保存状态均为有限值；原生严格模式启用，初始化几何检查通过。
- 交互回放器已实际初始化 OpenGL 并生成末帧截图；同步 GIF 已生成。
- 接入时修复了原生接口对非连续数组的误读：所有网格输入统一为连续行优先数据。

尚未完成独立逐帧自穿透扫描、接触力验收、三次以上重复计时或完整 4,373 帧流程。`world.is_valid()` 和有限状态检查不能替代上述物理质量验证。

## 重跑

当前目录的 `.venv` 使用已有本机科学计算环境，仅在隔离环境新增了网格简化依赖。新环境的版本列表见 `requirements-shirt.txt`。

```powershell
# 下载所需的少量资产；不下载全部视频、checkpoint 和渲染包
.venv/Scripts/python scripts/fetch_shirt_assets.py
.venv/Scripts/python scripts/probe_native_ipc.py
.venv/Scripts/python -m unittest discover -s tests -v
.venv/Scripts/python scripts/verify_shirt_hierarchy.py

# 三次运行须顺序执行；输出必须是新目录
.venv/Scripts/python -m shirt_demo.simulate --level 0 --frames 740 --output output/shirt_repeat/coarse
.venv/Scripts/python -m shirt_demo.simulate --level 1 --frames 740 --lower output/shirt_repeat/coarse --output output/shirt_repeat/progressive_fine
.venv/Scripts/python -m shirt_demo.simulate --level 1 --frames 740 --output output/shirt_repeat/direct_fine
.venv/Scripts/python scripts/analyze_shirt_runs.py --root output/shirt_repeat
```

首次重建小网格使用 `.venv/Scripts/python -m shirt_demo.assets --output output/shirt_assets`。更改资产时应使用新的目录，避免让旧轨迹对应的网格发生变化。每组输出有全物理步位置、实际刚体变换、碰撞网格、逐步计时和配置，图像可离线重建。

## 后续优先级

1. 在同一材料坐标采样点上完善抓取滑移、最终层叠覆盖率、连续性和应变指标；重复计时。
2. 引入可控的一致性项，并研究共享代理能或 PSQ 路线，解决当前细化后的一致性退化；不能只以单次速度提高作为成功标准。
3. 在本机逐级测试 24,576 面及原始 13k→55k 层级，记录显存峰值与失败情况。
4. 第一段通过明确质量验收后，再扩展原轨迹的后续折叠、扫褶与压平。不同拓扑必须从自己的初态建立完整状态链。

论文依据：[Progressive Dynamics 2024](https://pcs-sim.github.io/pd/) 与 [Progressive Dynamics++ 2025](https://pcs-sim.github.io/pd%2B%2B/)。外部仓库保留其 `NOTICE.md` 与原资产来源记录；本项目没有发布或再分发这些资产。
