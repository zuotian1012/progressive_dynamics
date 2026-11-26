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

本地参考机器为 RTX 4090，Python 3.12.3、Torch 2.8.0+cu128，驱动 580.178.04。requirements/scene527-py312-cu128.txt 基于旧 4090 环境清单，把 pyuipc 更新为本实验使用的 0.0.28；尚未在空环境整表安装验证。上游 Genesis 基准提交为 8b1dba2，应用 patches/genesis-world-8b1dba2-current.patch 后，将 genesis_overlay/genesis/ 合并至对应 Genesis 源码的 genesis/，再安装该源码。CUDA 后端需能为目标 GPU 编译；本机 CMake 使用架构 89。生成 psq_ssp 需要 C++17 工具链、CMake、Eigen 和固定版本的 SSP/libigl 源码。

下面的命令在交接仓库根目录执行，只取场景输入。机器需要 Git LFS。

```bash
ASSET_CLONE=$(mktemp -d)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 --branch handoff/scene527-ipc-demo-20261005 --sparse https://github.com/happy1041/genesis-ipc-shirt-demo.git "$ASSET_CLONE"
B=reproduction/scene527_55k_73s
GIT_LFS_SKIP_SMUDGE=1 git -C "$ASSET_CLONE" sparse-checkout set "$B/assets" "$B/transport" "$B/controls"
git -C "$ASSET_CLONE" lfs pull --include="$B/assets/**,$B/transport/**,$B/controls/**"
cp -a "$ASSET_CLONE/$B/assets" "$ASSET_CLONE/$B/transport" "$ASSET_CLONE/$B/controls" "$B/"
```

Python 3.12、pyuipc/libuipc 0.0.28；原机包版本见 requirements/。直接从 dependencies/libuipc/ 按其 README 构建 CUDA 后端。工作树已包含 PSQ 梯度修复。上游可选子模块 libuipc-samples 和 scripts/SymEigen 在原机器未检出，地址见该目录的 .gitmodules。两段历史视频采用修复前的 Uᵀg；复跑视频版本时，在另一个源码副本中反向应用 patches/pd-psq-chain-rule.patch，再编译。修复模型采用 J(x)ᵀg，搜索度量为 UᵀH_SPD U。

从 [Scene527 原场景工程](https://github.com/happy1041/genesis-ipc-shirt-demo/tree/c29cc92/reproduction/scene527_55k_73s) 获取 assets、transport 和 controls 并放入本目录同名 reproduction 子目录。控制轨迹和 55k garment atlas 需要自行准备。依照原场景的 Genesis 补丁部署，再将 genesis_overlay 中的适配文件覆盖到该 Genesis 树。准备好 Genesis、pyuipc 的 Python 环境，以及根据 docs/SSP_SOURCES.txt 和 tools/assets/psq_ssp/CMakeLists.txt 构建的 psq_ssp 后，运行：

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
