# Scene527 · happy1041 实现对账

本仓库直接整理自 happy1041 本机 PD 工作区。根目录为 PD 工程，dependencies/libuipc/ 包含本机 libuipc v0.0.28 工作树源码，包括未提交的 CUDA 实现。两条首折链均以 55,068 面为弹性目标。控制 60 Hz、物理 120 Hz，先沉降 60 步，再执行 801 个动作帧。

| 内容 | 位置 |
|---|---|
| 网格生成、SSP/PSQ 绑定 | tools/assets/ |
| BDF1、VelPro、首折与五阶段编排 | tools/pd/ |
| Scene527 仿真、接触与 Newton 记录 | dependencies/scene527-0.0.28/ |
| 场景、材料、七阶段命令模板 | reproduction/scene527_55k_73s/ |
| 本地 libuipc 源码、CMake 配置和 Python 绑定 | dependencies/libuipc/ |
| 非线性 PSQ 能量、梯度、Hessian 搜索度量 | dependencies/libuipc/src/backends/cuda/finite_element/progressive_cloth.cu |
| Genesis 项目适配 | patches/、genesis_overlay/ |
| 子步、能量及有限差分核验 | tests/ |

## 两个完整首折视频

- [4,090 → 7,021 → 13,974 → 27,811 顶点](outputs/pd_scene527_shirt_first_fold/video/chain_4090.mp4)
- [7,021 → 13,974 → 27,811 顶点](outputs/pd_scene527_shirt_first_fold/video/chain_7021.mp4)

## 复现

Python 3.12、pyuipc/libuipc 0.0.28；原机包版本见 requirements/。直接从 dependencies/libuipc/ 按其 README 构建 CUDA 后端。工作树已包含 PSQ 梯度修复。上游可选子模块 libuipc-samples 和 scripts/SymEigen 在原机器未检出，地址见该目录的 .gitmodules。两段历史视频采用修复前的 Uᵀg；复跑视频版本时，在另一个源码副本中反向应用 patches/pd-psq-chain-rule.patch，再编译。修复模型采用 J(x)ᵀg，搜索度量为 UᵀH_SPD U。

从 [Scene527 原场景工程](https://github.com/happy1041/genesis-ipc-shirt-demo/tree/main/reproduction/scene527_55k_73s) 获取 assets、transport 和 controls 并放入本目录同名 reproduction 子目录。控制轨迹和 55k garment atlas 需要自行准备。依照原场景的 Genesis 补丁部署，再将 genesis_overlay 中两个适配文件覆盖到该 Genesis 树。准备好 Genesis、pyuipc 的 Python 环境，以及根据 docs/SSP_SOURCES.txt 和 tools/assets/psq_ssp/CMakeLists.txt 构建的 psq_ssp 后，运行：

```bash
SRC="$PWD/reproduction/scene527_55k_73s"
RUN="$PWD/local_run"
python -m tools.assets.prepare_scene527_shirt_hierarchy "$SRC" "$SSP_EXECUTABLE" "$RUN/hierarchy"
python -m tools.pd.prepare_scene527_first_fold_motion "$SRC" "$RUN/motion" --urdf "$SRC/assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
python -m tools.pd.run_scene527_pdpp_first_fold "$RUN" smoke --backend "$BACKEND_DIR"
python -m tools.pd.run_scene527_pdpp_first_fold "$RUN" chain4090 --backend "$BACKEND_DIR"
python -m tools.pd.run_scene527_pdpp_first_fold "$RUN" chain7021 --backend "$BACKEND_DIR"
```

BACKEND_DIR 指向本地已编译的 CUDA 后端目录。对账时先统一后端版本和顶点顺序；逐子步 state 行 0 为初态，正式首折源帧 f 末态的行号为 60+2(f+1)。视频通过 Git LFS 保存，可按需下载。
