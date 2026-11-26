# Scene527 原始 55k 衣服的 PSQ 层级

构造日期：2026-11-15。正式资产目录：`output/shirt55k_psq_ready/`。
本次仅完成静态层级资产，没有启动动力学或折衣仿真。

| 层 | 顶点 | 三角面 |
|---|---:|---:|
| L0 | 533 | 999 |
| L1 | 1,807 | 3,499 |
| L2 | 27,811 | 55,068 |

原始最细输入来自 `external/genesis-ipc-shirt-demo/reproduction/scene527_55k_73s/assets/cloth/short-shirt-55068f.obj`，SHA256 为 `92ad1f68995a2d8b827d9de1c896465f436b8384c38d311abc9eadbbec9cf9cd`，与原仓库冻结 bundle 匹配。正式目录的 `source_original_mm.obj` 是逐字节副本。

仿真网格沿用 Scene527 世界变换：`world = source[:, [0,2,1]] * [.001,.001,-.001] + [.667,.015,.93]`。保持原最细网格顶点编号、面顺序与绕序；最细世界坐标与上述变换后的源数据逐值一致。原始文件未修改。

## 构造与候选选择

使用概率二次误差简化与联合连续自参数化构造内在映射，PSQ 采用现有 `progressive.prolongation` 实现。不是最近点配对，也没有用简化网格细分替换原始55k几何。

第一次自55k依次简化到目标14000、3500、1000面，实际生成14000、3499、999面。其14k层存在静态自交，保留于 `output/shirt55k_psq_candidate`，不可将该候选整体用于仿真。另尝试8000/3500/1000及3500/1000路径，分别保留于 `shirt55k_psq_4level`、`shirt55k_psq_3level`；它们虽然没有自交，但部分层未通过0.2mm厚度检查。

正式资产选择第一条构造路径中通过检查的999、3499、55068面三层。跨过14k层时组合线性内在矩阵U，使用已有直达最细层锚定面重新计算非线性偏移；没有复合非线性P冒充直接延拓。完整折叠历史经过顶点身份重映射并合并，没有丢弃被跳过的拓扑操作。

因此3.5k到55k的分辨率跨度较大。这一选择优先保证当前交付网格的静态可行性，不声称该层级数量或间隔具有最佳性能。

## 文件接口

- `mesh_0.npz` 至 `mesh_2.npz`：`vertices`（米，世界坐标）及`faces`（零基索引）。相同网格另有OBJ。
- `map_0_1.npz`、`map_1_2.npz`：相邻层CSR内在矩阵，形状为目标顶点数×源顶点数，使用`scipy.sparse.load_npz`。
- `map_0_2.npz`：L0直达原始最细层；L1直达最细层即`map_1_2.npz`。
- `anchors_i_j.npy`：每个目标顶点在源层的锚定三角形编号。
- `psq_i_j.npz`：`anchors`、`gamma`、`tangent`、`inv_dm`，与上述U配套。
- `expansion_0_1.json`、`expansion_1_2.json`：完整可逆拓扑历史，分别1274、26004次边展开；已逐步验证静止形状和有向拓扑重建。
- `manifest.json`：来源、变换、构造参数、候选选择及静态可行性。
- `audit.json`：独立资产验证；`checksums.json`：正式目录文件SHA256。
- `hierarchy.png`：三个网格的俯视线框图。

可由现有`progressive.run`资产加载器读取，资产参数为`--assets output/shirt55k_psq_ready --levels 3`。这只是接口说明，本次没有运行仿真。现有垂直动力学驱动仍仅支持两层，不能因三层资产已构造就认为三层垂直动力学已实现。

## 验证结果和边界

- 源文件哈希匹配冻结bundle；最细拓扑与世界坐标精确保持。
- 各层无退化三角形、重复面或非流形边；均为一个连通分量，欧拉示性数均为-2。
- 独立重算各层自交与0.2mm自接触间距检查，全部通过。距离字段为null表示阈值内没有接触候选，不表示测得无限间距。
- 所有相邻/直达最细层矩阵维度正确、权重有限非负、行和误差不超过4.45e-16。
- PSQ静止恢复最大坐标误差不超过4.1e-20米，刚体旋转/平移协变误差不超过6.7e-16米。
- 现有PSQ静止恢复及切向旋转单元测试通过。

这些只证明静态网格及映射的完整性。未验证变形后安全展开、夹爪/桌面预检、摩擦抓取、应变修正、折衣动态或速度。合并历史包含曾产生自交中间网格的折叠路径，不能直接把原始展开位置当成碰撞安全初始化；必须通过现有安全展开过程单独验收。

## 重建（使用新的输出目录）

```powershell
.venv/Scripts/python.exe -m progressive.prepare_intrinsic --input external/genesis-ipc-shirt-demo/reproduction/scene527_55k_73s/assets/cloth/short-shirt-55068f.obj --scene527-transform --targets 14000 3500 1000 --output output/NEW_shirt55k_candidate --contact-clearance 0.0002
.venv/Scripts/python.exe scripts/select_intrinsic_levels.py output/NEW_shirt55k_candidate output/NEW_shirt55k_ready --levels 0 1 3
.venv/Scripts/python.exe scripts/audit_shirt_hierarchy.py output/NEW_shirt55k_ready
```

第三步补导出直达矩阵、PSQ系数、OBJ和预览，执行独立静态碰撞和映射验证。`audit.json`中的`passed`必须为true。文件中保存的构建耗时仅是原生简化阶段，不含Python历史验证、候选尝试和审计，不是整个预处理墙钟或动力学性能。
