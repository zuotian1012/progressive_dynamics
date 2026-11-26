# 原生 C-IPC 90K 兔子基准

用户于 2026-11-12 要求先使用 C-IPC 得到最细层结果。此次调用作者公开的原生 C++ 求解器，不调用 progressive 求解器或 ipctk 接触封装。

## 实现与构建

- 上游 https://github.com/ipc-sim/Codim-IPC ，commit `9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a`。
- Ubuntu 22.04 / WSL2；工作目录 `/opt/cipc-baseline`。
- 140 个 Library、Python、所需内嵌依赖文件逐文件验证 Git blob SHA，复制到 `/opt/cipc-baseline/source`。求解器源码未修改。
- 复现构建入口：`scripts/cipc_prepare_build.py`、`native/cipc/CMakeLists.txt`。详细文件 SHA 和依赖版本在 `native/cipc/source_manifest.json`。
- GCC 11，Release `-O2 -DNDEBUG`，CHOLMOD 5.10 + OpenBLAS，Kokkos 3.1.01，作者指定的 Liby99/Cabana fork。系统 pybind11 2.9.1 适配 Python 3.10。独立 CMake 只调整依赖/构建，不重写物理或求解算法。
- 原生模块 `/opt/cipc-baseline/build/JGSL.cpython-310-x86_64-linux-gnu.so`；实际二进制 SHA 记录于每次 run.json。

## 输入与参数

`scripts/cipc_export_bunny.py` 将 `output/bunny_paper_assets/mesh_7.npz` 与 `bunny.npz` 无变换导出到 `output/cipc_bunny_inputs`，保持原始 90,000 布料顶点、178,802 三角形和 3135 顶点固定兔子。OBJ 输入保留 17 位有效数字。

直接跑最细层，400 步 × 0.01 s = 4 s。8 线程；重力 (0,0,-9.81)；E=30000 Pa、nu=.3、density=200；厚度 .00007 m；主拉伸上限 1.068、激活 1.034、初始应变屏障刚度1000；摩擦 .3；Initialize_OIPC 激活距离参数 .001、接触 offset .00007。

接触刚度使用 C-IPC 原生 Initialize_OIPC 的质量相关初始化和原生更新，没有把旧 Python 的接触刚度10000直接移植。原生 RMS 速度增量容差1e-4，以最近三次残差的最大值判断收敛，不设置自写预览早停或额外迭代上限。每步原生函数返回后，适配器复核末三次残差、写完帧才更新 completed_steps。

**作者 PD++ 原始兔子场景仍未取得；材料和部分接触值仍是本地假设，4 s 是声明的观测时段。不能称为作者场景的完全复现，也不能把此次与旧 Python 结果说成全部求解细节相同的对照。**

## 运行与交付

正式输出：`output/cipc_bunny_native_90k`，2026-11-12 12:58:20（Asia/Shanghai）启动。

入口：Ubuntu 的 python3 执行 `scripts/cipc_run_bunny.py --inputs .../output/cipc_bunny_inputs --output .../output/cipc_bunny_native_90k --steps 400 --threads 8`。完整宿主启动命令在 `output/cipc_bunny_native_launcher.json`；标准输出与错误日志为同名前缀 `.log`、`.err.log`。

运行状态 `run.json`，接受步记录 `steps.jsonl`，原生输出 `raw/shell0.obj` 至 `raw/shell400.obj`，残差 `raw/residual.txt`，Newton/接触数 `raw/counter.txt`，主拉伸 `raw/stretch.txt`。只读取 completed_steps 以内帧；原生 OBJ 使用 `%le`，约7位有效数字，重算应变存在坐标舍入误差，不能要求与原生双精度内部值逐位相同。

报告命令（Windows项目Python）：

```
.venv\Scripts\python.exe scripts/cipc_report_bunny.py output/cipc_bunny_native_90k
```

未完成时必须显式加 `--accepted-prefix`，报告自动进入 report_prefix 并在动画标注未完成。完整报告进入 report，包含 trajectory.gif、final.png、metrics.png、audit.json。所有面使用真实轨迹渲染；检查动画后才能报告是否披覆/滑落。审计检查全接受帧有限值、固定兔子、应变及原生收敛日志，未声称独立连续CCD审计。

前置原生运行检查 `output/cipc_bunny_native_probe`：真实90K网格2步，均4次Newton，末三次残差约1e-10，累计求解37.775 s；第二步z=.597057，与自由落体隐式Euler解析值完全一致（输出精度内）；兔子坐标误差约5e-8 m（OBJ舍入）。没有接触，不能当作正式实验完成，也不能推断接触阶段耗时。

旧 `bunny_paper_scale` 自写求解器套件未被修改或终止，其结果单独保留；当前优先交付本原生最细层实验，不将两个实验混合，不声称加速。

## 正式 400 步结果（2026-11-13）

`output/cipc_bunny_native_90k/run.json` 已标记 completed=true，400/400 步、物理时间4 s，结束时间为 2026-11-13T05:37:43.793410+00:00。已确认对应完整命令行的原生进程退出，日志最后为 CIPC_ACCEPTED 400/400。

- 400步全部严格收敛，0步预览早停。原始 residual.txt 共17171次Newton、400组；逐组末三残差与 steps.jsonl 一致，counter.txt 400行迭代数也逐步一致。末三残差的全程最大值9.98749e-5，小于1e-4；单步最多281次Newton。原始日志交叉核对保存在 `native_log_audit.json`。
- 原生最大主拉伸1.064613（约6.4613%伸长），低于1.068。401个接受状态（含初始帧）全部有限，OBJ坐标重算最大1.064614365882564；差异与OBJ七位有效数字输出有关。固定兔子最大坐标误差4.99894e-8 m。未执行独立连续CCD认证。
- 累计求解86465.026567 s（24小时1分5秒）；程序单调时钟墙钟86688.949171 s（24小时4分49秒），其中初始化0.801669 s。UTC开始/结束时间差88761.535649 s（24小时39分22秒），比单调计时多2072.586478 s；日志未解释该差异，不能自行归因为求解耗时。上述时间不包含本次离线审计与渲染。
- 报告：`report/trajectory.gif` 是4秒完整轨迹；`report/contact.gif` 是0至0.8秒接触滑落阶段近景，慢放且标注真实物理时刻；`report/metrics.png` 为拉伸、Newton迭代数及单步求解耗时；`report/audit.json` 为全部接受帧审计。
- 没有地板：在1秒接受帧中布料所有顶点均低于兔子底部（布料最高z=-1.417693 m，兔子底部z=0），4秒时布料z范围[-64.92206,-63.98939] m。后段持续下落不能解释为在地面稳定铺开。
- 已检查近景动画抽帧：0.2秒布料开始披覆耳部，0.4秒包覆兔子并起皱，0.6秒滑向下方，0.72秒可见布料与兔子分离，0.8秒继续下落。完整GIF61帧和近景GIF21帧均可解码，最终图与指标图已检查。完整轨迹为容纳约65米下落范围，物体显示很小，行为判断以标注时刻的实际近景为主。

该结果证明这组重建输入可由原生C-IPC严格完成400步，不构成PSQ/PD++多层算法已通过验收，也不能据此单独定位旧实现的全部失败原因。未做隔离重复计时，不作加速结论。
