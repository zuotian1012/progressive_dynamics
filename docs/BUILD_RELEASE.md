# 完整源码构建与运行

## 原生源码的位置

`third_party/libuipc/` 是本地修改后的 libuipc 源码快照，上游参考 commit 为 `9c5710ee4b716502e3757e7312360e5ee68cb79f`。它不是未修改的上游版本。`native/release/backend_sources.json` 记录 1720 个源文件的 SHA-256 和来源；11 个最终覆盖文件与历史事件回退构建中的记录一致，原构建记录的 34 个来源哈希也已核对未变。

这次把历史目录中的最终源码放回正常编译位置：

| 功能 | 源码位置（相对 third_party/libuipc/src/backends/cuda） |
|---|---|
| 事件触发完整梯度回退 | `engine/advance_ipc.cu` |
| 非线性 PSQ 与确定性梯度汇总 | `finite_element/constitutions/psq_proxy.cu`、`psq_exact_offset.cuh` |
| FEM 有序梯度与矩阵装配 | `finite_element/fem_linear_subsystem.cu`、`linear_system/global_linear_system.cu` |
| 接触能量有序汇总 | `dytopo_effect_system/dytopo_effect_line_search_reporter.cu` |
| RMS 收敛检查 | `newton_tolerance/max_translation_checker.cu` |
| 支持头文件 | `pd_support/` |

运行时仍使用已有 PCG 实现。构建不复用旧 `.obj`，也不读取历史 `build_env.json` 或 `commands.json`。历史代码与脚本仍保留，便于追溯。

## 环境与命令

当前支持路径：Windows x64、Python 3.11、Visual Studio 2022 C++ 工具、CUDA Toolkit（本机验证 12.6）、xmake >= 3.0.5。Linux 构建尚未适配。编译需要主机内存与 CPU，但脚本不会启动仿真。

```powershell
python scripts/build_release_backend.py --jobs 2 --cuda-arch sm_86
```

请使用安装了本项目 Python 依赖的 Python 3.11 解释器。`--cuda PATH` 可显式指定工具包，否则读取 `CUDA_PATH` 或查找 `nvcc`。`--prepare-only` 仅校验并准备源码；`--configure-only` 只做到工具链和依赖配置。

- `build/configure.log`、`build/compile.log`、`build/dependencies.log`：阶段日志。
- `build/build_status.json`：实际命令与完成状态。
- `build/runtime/src`：可供运行脚本使用的 Python 包和原生库。
- `build/runtime/provenance.json`：二进制哈希、源码哈希与功能标识；构建完成后生成。

构建命令以低优先级运行，默认并发为 2；可用 `--jobs` 调整。源码校验清单用于固定交付版本，若主动修改 `third_party/libuipc`，应同步更新 `native/release/backend_sources.json` 中相应文件的 SHA-256。Git 已配置为保留这些源码的原始字节，避免换行转换导致校验失败。

xmake 根据本机工具链生成编译命令，可能联网安装第三方包。运行库打包通过本次 xmake 依赖图查找 DLL，避免读取旧工作区或 pip wheel 的原生库。

编译后可检查绑定与 DLL 加载，不启动仿真：

```powershell
python scripts/check_release_backend.py
```

本次解析到的依赖版本见 `release/native_dependency_versions.json`；Python 环境锁定版本见根目录 `requirements-release-lock.txt`。

## 连接现有场景入口

Bunny 使用明确的资产目录和新构建：

```powershell
python scripts/run_bunny_velpro_chain.py --event-fallback --native-package build/runtime/src --assets PATH_TO_PREPARED_BUNNY --steps 30 --max-level 0 --output output/bunny_release_L0
```

该命令需要事先准备完整的 Bunny 层级输入；`PATH_TO_PREPARED_BUNNY` 是需要替换的目录，不是随源码附带的资产。`--max-level 0` 允许只验收 L0。

Shirt 的 `scripts/run_shirt_psq.py` 接受 `--runtime-package build/runtime/src`，其余网格、PSQ bundle、机器人模型和夹爪轨迹参数见 `docs/SHIRT_EVENT_PCG.md`。该文档保留历史命令，不能把其中旧 output 路径当作本仓库自带输入。

运行库识别已支持相邻 `provenance.json`；DLL 哈希不匹配会被拒绝。加载器优先使用运行库自身依赖和本机 CUDA 工具包，不依赖原作者用户名或固定 xmake 缓存路径。

## 验证层次

1. 源码与引用完整性、主机编译及绑定导入。
2. Python PSQ、动力学、层级生成和审计规则测试。
3. 新编译 CUDA 二进制上的实际几何/收敛验证。

这些层次不互相替代。历史 Bunny / Shirt 数值结果不能直接作为新编译二进制的验证记录。本轮避免争用原工作区 GPU，未重新运行大型 CUDA 场景。
