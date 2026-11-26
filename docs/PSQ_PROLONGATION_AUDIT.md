# PSQ 非线性延拓核对（2026-11-12）

结论：现有 `progressive/prolongation.py` 已实现 PSQ §4.3–4.3.1 的核心构造，未发现需补写的公式或本次输入上的算子计算错误。本次不修改生产延拓、Newton、资产或物理参数；增加独立算子审计脚本和本记录。此结论不能扩大为完整 PSQ/PD++ 仿真已复现成功。

依据：[作者论文](https://pcs-sim.github.io/psq/psq-combined.pdf)，本地 `papers/psq-combined.pdf`。阅读第7–8页公式(4)–(9)、第10–11页§4.5及Algorithm 1；另渲染检查第7页脚注的矩形旋转因子。论文与代码对应如下。

| 项目 | 论文定义 | 当前实现与核对 |
|---|---|---|
| 静止形状残差 | d = fine_rest − U coarse_rest | 构造器计算delta；不是把偏移固定在世界坐标中 |
| 偏移分解 | gamma = n_rest^T d；d = n_rest gamma + (e, n_rest × e)t | gamma、tangent使用包含三角形的正交局部基 |
| 变形梯度 | F = Ds B_rest^-1，3×2 | frames构造2×2参考矩阵，offset计算F |
| 旋转因子 | R = U_svd[:, :2] V_svd^T | numpy薄SVD的left @ right等价；不把拉伸乘到切向残差上 |
| 法向 | 当前包含三角形单位法向 | cross(F列0,F列1)归一化；参考矩阵行列式为正，和当前边叉积方向一致 |
| 非线性延拓 | P(x) = Ux + n(x)gamma + R(F(x))t | offset与position实现式(5)、(7) |
| 直接延拓 | 组合线性U，再由当前粗层与最细层静止形状重新构造残差 | Hierarchy.to_finest实现式(8)、(9)；没有串联非线性P来替代 |
| Newton接口 | 每次迭代刷新偏移，求导时冻结；用U^T限制细层力 | solver.evaluate/solve与§4.5一致；不添加偏移的完整Jacobian |

层级输入也已沿调用路径核对：native/intrinsic_builder/main.cpp通过joint self-parameterization的query_fine_to_coarse导出相邻映射与锚三角形，并传播最细点的锚；prepare_intrinsic.py保存，run.prepare_scene加载。没有使用最近点投影替代此路径。本次检查了相邻映射每行的非零列都属于所记录锚三角形；没有重新运行或全面认证网格简化和joint-UV算法。

## 少量针对性验证

新增 `scripts/audit_psq_prolongation.py`。独立参照直接从输入边构造局部坐标，并使用 **F(F^T F)^(-1/2)** 计算极分解旋转，未调用生产frames/offset或复制其SVD路径。

1. 两三角形铰链弯曲、附加剪切、纯面内伸长。法向/切向残差均非零；检查旋转随形变更新、偏移不随拉伸缩放。
2. 实际bunny的全部8层，检查7个相邻算子及8个直接到最细层算子：静止形状恢复、刚体变换、非均匀弯曲加剪切与独立参照一致。
3. 直接线性映射与逐层线性应用一致；直接非线性延拓与逐层非线性延拓有约0.000490811米差异。这是论文§4.3.1明确指出的不同，不能修成相等。
4. 读取先前失败回放的L0诊断迭代点，只比较延拓结果，不重跑Newton。此点不是接受轨迹或新的仿真成果。

本次结果（米，最大坐标绝对误差）：

| 检查 | 误差 |
|---|---:|
| 静止形状恢复 | 6.94e-18 |
| 刚体变换 | 3.55e-15 |
| 弯曲/剪切下独立公式对照 | 1.11e-16 |
| 旧失败诊断点独立公式对照 | 1.11e-16 |
| 组合线性映射 | 6.66e-16 |
| U行和偏离1 | 6.66e-16（无量纲） |

算子审计核心耗时4.39秒，单线程；另运行现有两个测试：静止/刚体/拉伸偏移及冻结偏移代理能量的有限差分梯度，共0.015秒，两项通过。没有运行全套测试或长时间仿真。

复查命令：

```powershell
.venv/Scripts/python.exe scripts/audit_psq_prolongation.py --diagnostic-state output/bunny_failure_diagnosis/diagnostic_iterate.npy
.venv/Scripts/python.exe -m unittest tests.test_progressive.ProgressiveTests.test_psq_rest_rigid_and_tangent_rotation tests.test_progressive.ProgressiveTests.test_frozen_nonlinear_proxy_gradient
```

结构化结果：`output/psq_prolongation_audit.json`，含输入资产、审计脚本、生产延拓代码及诊断点SHA256。运行时会重写这个审计文件，不会写模拟目录。

## 范围与尚未解决的问题

- 生产代码有小于1e-13米残差的零偏移优化，以及退化三角形的数值保护。这些是实现级数值选择，不是论文额外公式；实际bunny各非恒等层均走非零偏移路径。不能声称任意尺度输入逐位精确等同作者实现。
- 冻结偏移后的近似梯度不保证对完整非线性能量下降，这是近似求解策略的性质。旧失败诊断点的延拓计算与独立参照吻合，进一步排除了该点出现明显算子算错的解释；并不证明所有求解和输入均正确。
- 不把完整偏移Jacobian加入生产求解器：这会改变论文所述的近似求解方法，也不能把它包装成补齐遗漏。
- 本次未验收安全细化/碰撞、材料模型、全套PD++动力学或原作者场景一致性。旧求解器的线搜索停滞和终止策略仍未修复，也未以放宽容差、扩大早停阈值或强行接受帧掩盖。
- 原生C-IPC运行不依赖此Python延拓模块，本次未干预它。
