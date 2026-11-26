# 推荐入口

| 工作 | 入口 |
|---|---|
| 非线性 PSQ 演示 | `python examples/psq_demo.py` |
| 真正的 PSQ CPU 动力学 | `python -m progressive.run` |
| 网格层级生成 | `native/intrinsic_builder` + `python -m progressive.prepare_intrinsic` |
| 完整 CUDA 后端构建 | `python scripts/build_release_backend.py` |
| Bunny 严格收敛链 | `scripts/run_bunny_velpro_chain.py --event-fallback --native-package ... --assets ...` |
| Shirt 主运行/监控 | `scripts/run_shirt_psq.py`、`scripts/supervise_shirt_psq.py` |
| Shirt 新策略审计 | `scripts/audit_shirt_event.py` |

详细命令见根 README 和 `docs/BUILD_RELEASE.md`。`build_flag_*_fix.py`、`build_bunny_event_fallback.py`、`probe_*`、`diagnose_*` 等旧入口保留历史实验逻辑，其中一些依赖未附带的旧输出。新的完整构建不需要逐个运行历史修复脚本。
