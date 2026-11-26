# L1独立分辨率探索与L0对照

2026-11-15。L1输出：`output/shirt55k_L1_preview20/`；同步对照：`output/shirt55k_L0_L1_comparison/`。

**L1完成时间推进，但没有成功折衣。网格加密没有解决当前强制接受设置下的夹爪执行问题。**

| 指标 | L0 | L1 |
|---|---:|---:|
| 顶点 / 面 | 533 / 999 | 1807 / 3499 |
| 完成物理步 | 1540 | 1540 |
| 物理时段 | 12.8333 s | 12.8333 s |
| 强制接受步数 | 1540 | 1540 |
| 严格收敛步数 | 0 | 0 |
| 最大主拉伸 | 1.099996092 | 1.099998517 |
| 最大夹爪刚体原点跟踪误差 | 0.356872 m | 0.427157 m |
| 推进/取回/状态传递累计耗时 | 774.297 s | 554.388 s |

L1衣服仍大体展开，没有被搬移成预期折叠。以上耗时属于单次未收敛探索，不构成算法加速或模拟质量改善的证据。

## 对照含义

两次运行均从各层静止初态开始，用原55,068面网格作PSQ弹性代理，独立BDF1推进当前层。**本次L1没有读取L0轨迹，没有VelPro速度注入，不是L0→L1渐进积分实验。** 这样用于检查独立加密分辨率能否改变失败现象，避免直接继承已停滞的L0运动。

配置、材料、控制轨迹哈希、步数、时间步以及七项原生DLL哈希已和L0逐项核对一致，见L1目录的`comparison_setup.json`。同样采用每步20次Newton、32次回溯、原max-axis阈值，启用用户授权的强制接受模式；CCD、应变步长限制和接触摩擦仍保留。

L1使用`output/shirt55k_psq_ready/mesh_1.npz`；直接到L2的内在映射有132868个非零元素。导出的原生包位于`output/shirt55k_L1_operator/L1_to_55k.bin`。L2只作弹性能缓冲，不新增质量、接触或动力学自由度。

## 验证边界

全部保存状态有限；逐帧重算最大主拉伸未超过1.10。原生日志与保存状态逐步对应，全部强制接受，没有将预算耗尽计为收敛。仍未额外执行独立逐帧碰撞审计。回放使用真实求解的夹爪姿态，不以目标动作替换。不能仅凭该结果断言网格分辨率不影响最终可收敛解，也不能据此确认停滞的唯一根因。

## 产物与入口

- L1的`report/motion.gif`、`report/final.png`、`report/metrics.png`、`report/audit.json`。
- 对照目录的`comparison.gif`、`final.png`、`comparison.json`：统一物理时间、相同视角、逐采样帧标注收敛/强制接受。
- 原始布料位置、实际夹爪姿态、逐步计时和原生日志均保留。

```powershell
.venv/Scripts/python.exe scripts/run_shirt_psq.py --level 1 --psq-bundle output/shirt55k_L1_operator/L1_to_55k.bin --force-updates --max-iterations 20 --output output/NEW_L1 > output/NEW_L1.log 2>&1
.venv/Scripts/python.exe scripts/report_shirt_psq.py output/NEW_L1
```

本轮没有调整材料、关闭约束或改动原生求解器以追求成功，也没有自动启动L2或其他参数试验。
