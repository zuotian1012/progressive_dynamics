# 2023 PSQ 壳延拓接入

PD 新增了《Progressive Shell Quasistatics for Unstructured Meshes》§4.3 的壳延拓。离线工具使用 Liu 等人的公开 successive self-parameterization（SSP）实现和 QSlim 边折叠。相邻映射保留 SSP 查询对应的粗三角形；直达最细层组合相邻线性算子 U，并针对最终静止坐标重新计算偏移，用跨折叠的 SSP 面查询确定偏移参考面。几何不同的层级可以使用变长稀疏权重。当前 decimation 采用公开 SSP 工具的 QSlim 实现，论文实验采用自身的简化配置。[PSQ 2023](https://pcs-sim.github.io/psq/psq-combined.pdf) · [SSP 2021 公开实现](https://github.com/HTDerekLiu/surface_multigrid_code)

每个细点按 y = Ux + nγ + Rt 求值。γ 和 t 是静止构型下的法向与切向偏移，n 是变形粗面的法向，R 是其 3×2 形变梯度的薄极分解旋转部分。CUDA 每次计算粗层弹性能时在延拓后的细层几何求值。下文历史曲壳实验使用冻结偏移近似，按 Uᵀ 回传梯度。Scene527 修复版对法向和薄极旋转一并求导，使用 J(x)ᵀg，其中 J(x) = ∂y/∂x。两版均对试探位置求完整非线性弹性能量，均使用 UᵀH_projU 作为 Newton 搜索度量。修复版与历史二进制单独保存。[PD 2024](https://eriszhang.github.io/assets/pdf/progressive-dynamics-main.pdf)

相邻层的 paper_eq5 用完整位置延拓和线性速度延拓，中心为 P(x_c) + hUv_c。VelPro 使用本层位置和同一套 Uv_c。两支继续按本层上一步位置做横向热启动。PSQ §4.4 的安全边展开属于纵向热启动的独立算法，本次只接入 §4.3 的延拓。

## GPU 曲壳验证

同一块预弯曲非结构壳在固定齿轮上落下。层级为 42 → 94 → 195 个顶点，63 → 156 → 336 个三角形。粗预览、两条 PD 链和直接细网格组共 6 次运行，每次 60 步，步长 0.01 s。

42→195 的直达映射最多使用 8 个粗点。单用内禀 U 的静止重建最大差 52.35 mm，加入偏移后各映射的最大坐标差为 5.55 × 10⁻¹⁷ m。用非轴对齐的刚体旋转与平移验证了坐标一致性。第 25 步有代理的三次运行中，CPU 与 CUDA 延拓的最大坐标差均为 1.11 × 10⁻¹⁶ m。另核对了原版 PD 输入的相邻层位置及速度数据。

| 运行 | 达到 100 次 Newton 上限的子步 |
|---|---:|
| 42 顶点共用粗预览 | 15 / 60 |
| 94 顶点 paper_eq5 | 11 / 60 |
| 94 顶点 VelPro | 1 / 60 |
| 两组 195 顶点与直接细网格 | 每组 0 / 60 |

[曲壳数据与验证](../outputs/pd_psq/curved/validation.json) · [层级清单](../outputs/pd_psq/curved/hierarchy/manifest.json) · [运行入口](../tools/pd/run_psq_shell.py)

第 25 步的冻结偏移装配与独立细网格 CUDA 装配对比后，限制梯度最大逐项差为 1.94 × 10⁻¹⁸，投影限制海森最大逐项差为 1.73 × 10⁻¹⁶。原有平面代理三个粗层的投影海森回归误差最大为 3.07 × 10⁻¹⁴。 [曲壳导数对照](../outputs/pd_psq/curved/derivative_check.json) · [平面回归数据](../outputs/pd_psq/flat_regression.json) · [静止与变形延拓对照图](../outputs/pd_psq/curved/psq_geometry.png)

同一曲壳另跑了 26 → 43 → 63 → 95 → 142 → 195 顶点的六级链。共 12 组各完成 30 步。直达矩阵的最大单点权重数为 12，CPU 与 CUDA 延拓的最大坐标差为 1.67 × 10⁻¹⁶ m。共用粗预览、paper_eq5 的第三级细化及 VelPro 的第四级细化分别有 8、3、1 个子步达 Newton 上限。 [六级结果](../outputs/pd_psq/curved_six/validation.json) · [六级层级清单](../outputs/pd_psq/curved_six/hierarchy/manifest.json)

## 运行命令

独立离线工具依赖 SSP 的 adc49abe10a97200a3a5735bd370e98560091a37 和 libigl 子模块 70d2abaa53616af35c7857bf8f52c8a234f150ff。工作区已有带原许可证的依赖快照，位于 dependencies/psq_intrinsic，可按以下命令重建。PSQ 实验使用 libuipc 自带接触求解器与该工作区的 CUDA 补丁。

```bash
cmake -S tools/assets/psq_ssp -B outputs/pd_psq/ssp_build \
  -DPSQ_SSP_SOURCE="$PWD/dependencies/psq_intrinsic" \
  -DPSQ_EIGEN_INCLUDE=/path/to/eigen \
  -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/pd_psq/ssp_build -j 4
cmake --build outputs/pd_eq5_cuda -j 8
envs/genesis-ipc/bin/python -B -m tools.pd.run_psq_shell \
  --ssp outputs/pd_psq/ssp_build/psq_ssp --output outputs/pd_psq/curved
```

可通过 --mesh 和 --face-targets 指定自有三角网格和由细到粗的面数。单独导出实时延拓和弹性目标资产见 [prepare_psq_hierarchy.py](../tools/assets/prepare_psq_hierarchy.py)。
