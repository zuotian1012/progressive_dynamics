# 输入与依赖

算法及其必要源码依赖已经整理进仓库：`third_party/libuipc` 为修改后的原生后端；`third_party/surface_multigrid`、`eigen`、`libigl` 和 `probabilistic_quadrics` 用于网格层级预处理。

`tests/fixtures/intrinsic_expansion` 和 `intrinsic_grip` 保留自生成的小型网格、映射及展开历史。根 README 提供从其中 OBJ 重新生成层级并仿真的完整命令，运行不需要原工作区。

大型场景输入仍独立提供：

| 场景 | 需要的内容 | 已有准备入口 |
|---|---|---|
| Bunny | 碰撞体、布料各层网格、映射及锚定三角形 | `progressive/prepare_bunny.py`；需要提供原始 Bunny OBJ |
| Shirt | 衣服各层网格、PSQ bundle、机器人碰撞模型、夹爪轨迹 | `scripts/fetch_shirt_assets.py` 及原层级准备工具；来源为 Scene527 bundle |

原 Scene527 来源 NOTICE 尚未选择开源许可证；本源码包不附带该场景资产。已有运行入口接受显式输入目录。不要用指向原工作区的软链接代替独立输入包。
