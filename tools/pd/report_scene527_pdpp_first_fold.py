"""Summarize verified Scene527 first-fold runs, including partial sweeps."""

import argparse
import json
from pathlib import Path
import re

import numpy as np

from tools.pd.run_integrator_benchmark import read_mesh, read_p
from tools.pd.scene527_grasp_audit import audit_case


def load(path):
    return json.loads(path.read_text())


def recent_step(case):
    path = case/'step_mapping.jsonl'
    if not path.is_file():
        return None
    with path.open() as stream:
        rows = stream.readlines()
    for row in reversed(rows):
        try:
            return json.loads(row)
        except json.JSONDecodeError:
            continue
    return None


def initialization_failure(case):
    path = case/'physics.log'
    if not path.exists():
        return None
    log = path.read_text()
    if 'World is not valid, skipping init.' not in log:
        return None
    gap = re.search(r'too close \(distance=([\deE+.-]+), thickness=([\deE+.-]+)\)', log)
    problems = []
    if 'Intersection detected between' in log:
        problems.append('静止网格自交')
    if gap:
        distance, thickness = (float(value)*1000 for value in gap.groups())
        problems.append(f'同衣片检测距离 {distance:.3f} mm，小于合计厚度 {thickness:.3f} mm')
    return '初始场景校验失败' + ('，' + '，'.join(problems) if problems else '')


def generate(output):
    mesh_dir = output / 'hierarchy'
    hierarchy = load(mesh_dir / 'manifest.json')
    motion = load(output / 'motion/trajectory_check.json')
    rest = [read_mesh(mesh_dir, i)[0] for i in range(7)]
    lines = ['# Scene527 首折 · PD++ 分辨率扫描', '',
        '使用 Scene527 双 X5 与纯接触摩擦。布–桌 μ=1，布–布 μ=2，布–夹爪 μ=√2。'
        '保留 `d_hat=1.5 mm`、`dt=1/120 s` 和 Newton 上限 100。', '',
        '材料保持 `stretch_E=9800`、`shear_E=4`、`ν=0.49`、`ρ=800 kg/m³`、'
        '单侧厚度 `0.1 mm`、`κ=1e-7 N·m` 与 `strain_rate=100`。实际膜参数为 `λ≈2.57929` 和 `μ_shear≈1.34228`。', '',
        '每组从其最粗层独立运行 BDF1，后续层使用 VelPro。除原生最细层外，弹性目标统一为 55,068 面。'
        '惯性质量、夹爪接触、桌面接触和自接触仍使用当前层。VelPro 保留本层位置，预测时读取父层同一子步起点的速度。', '',
        f"原动作保留到源帧 {motion['original_last_source_frame']}，之后维持张爪退离姿势 {motion['hold_frames']} 帧。"
        f"右手闭爪段 TCP 目标最多下调 {motion['depth_m']*1000:.2f} mm，共有 {motion['actions']} 个动作帧；"
        '与改动前的历史原生结果比较时，需注意这项轨迹差异。', '',
        '## 网格与延拓验收', '',
        '| 层 | 顶点 | 三角面 | 最小角（°） | 最大长高比 | 一角 < 5° 的面数 | 资产 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for i, mesh in enumerate(hierarchy['levels']):
        lines.append(f"| L{i} | {mesh['vertices']:,} | {mesh['triangles']:,} | {mesh['minimum_angle_deg']:.3f} | "
                     f"{mesh['aspect_ratio_max']:.2f} | {mesh['sliver_faces_below_5deg']} | {mesh['source_obj'] or '本轮生成'} |")
    checked = []
    for item in hierarchy['operators']:
        matrix = read_p(mesh_dir,item['kind'],item['source'],item['target'])
        checked.append(float(np.max(np.abs(matrix.sum(axis=1)-1))))
    bridge = hierarchy['mappings'][3]
    rest_error = max(operator['rest_error_m'] for operator in hierarchy['operators'])
    lines += ['', f'12 个延拓矩阵最大行和误差为 {max(checked):.3e}，加壳体偏移后的静止位置最大分量误差为 {rest_error:.3e} m。',
              f"4,090→7,021 是保持朝向的近邻表面绑定，线性锚点距离 P95 为 {bridge['projection_distance_p95_m']*1000:.3f} mm，"
              f"最大为 {bridge['projection_distance_max_m']*1000:.3f} mm。该段保留了两份原始资产，绑定存在近似对应。"]
    importer = mesh_dir/'genesis_import_check.json'
    if importer.is_file():
        checked_import = load(importer)
        assert len(checked_import) == len(rest) and all(row['exact_face_order'] for row in checked_import)
        maximum = max(row['world_rest_max_error_m'] for row in checked_import)
        lines.append(f'七层通过 Genesis 导入核对，三角面索引顺序完全一致，世界坐标的最大分量差为 {maximum:.3e} m。')
    lines += ['',
              '## 求解器与抓持结果', '',
              '抓持主判据为动作第 8.0 秒（源帧 480）的夹爪 TCP 到对应布料材料点的距离，每手 ≤100 mm 记为 `PASS`。'
              '闭爪时把 TCP 投影到最近的布料三角面，固定面编号和重心权重。8.0 秒时用同一个材料点和回放中的 IPC 夹爪位姿测量。'
              '该面是几何近似对应。逐步 `physics.log` 保留旧版补丁跟踪诊断。', '',
              '| 运行 | 顶点 | 动作帧 | Newton P50 / P95 / max¹ | 触顶子步 | 触顶残差 / 容差 P50 / max | 动作耗时（s） | 动作 FPS | 左手 8 s | 右手 8 s | 双手 |',
              '| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |']
    verified = sorted(output.glob('*/*/verified.json'))
    for path in verified:
        info = load(path)
        solver = info['solver']
        grasp = audit_case(output,path.parent)
        def hand(name):
            if grasp is None:
                return '无回放'
            score = grasp['hands'][name]
            distance = f"，{score['distance_m']*1000:.1f} mm" if 'distance_m' in score else ''
            return score['verdict'] + distance
        iterations = ' / '.join(f'{x:g}' for x in solver['checker_calls_p50_p95_max'])
        capped = load(path.parent/'solver_summary.json')['newton_cap_locations']
        ratios = [step['last_residual']/step['last_abs_tol'] for step in capped if step['last_residual'] is not None]
        residuals = f'{np.median(ratios):.2f} / {max(ratios):.2f}' if ratios else '—'
        key = path.parent.relative_to(output)
        lines.append(f"| {key} | {info['vertices']:,} | {info['frames']} | {iterations} | "
                     f"{solver['newton_cap_steps']} | {residuals} | {info['action_wall_s']:.1f} | {info['frames']/info['action_wall_s']:.2f} | "
                     f"{hand('left')} | {hand('right')} | {grasp['verdict'] if grasp else '待检'} |")
    lines += ['', '¹ 统计每个物理子步实际执行的 Newton checker 次数，含初始沉降。各层 `grasp_8s.json` 保存面编号、重心权重和判定。', '',
              '## 完整链与轨迹误差', '',
              '| 最粗层 | 计划层数 | 已完成 | 已完成层总耗时（s） | 相邻层动作轨迹 RMS（mm，按层顺序） | L6 对同轨迹原生 IPC RMS（mm） | 状态 |',
              '| --- | ---: | ---: | ---: | --- | --- | --- |']
    direct = output/'direct/L6'
    active = []
    for start in range(5):
        group = output/f'L{start}_start'
        results = [level for level in range(start,7) if (group/f'L{level}/verified.json').is_file()]
        elapsed = sum(load(group/f'L{level}/verified.json')['wall_s'] for level in results)
        error, comparison = [], '待运行'
        previous = None
        for level in results:
            with np.load(group/f'L{level}/state.npz',allow_pickle=False) as state:
                current = state['x'][61:]
            if previous is not None:
                matrix = read_p(mesh_dir,'adjacent',level-1,level)
                error.append(f'L{level-1}→L{level} {rms(previous,current,matrix,rest[level-1],rest[level]):.2f}')
            previous = current
        if results and results[-1] == 6 and (direct/'verified.json').is_file():
            with np.load(direct/'state.npz',allow_pickle=False) as state:
                comparison = f'{rms(state["x"][61:],previous):.2f}'
        status = '完成'
        if len(results) < 7-start:
            pending = group/f'L{start+len(results)}'
            step = recent_step(pending)
            status = '待完成'
            if step:
                stage = '源帧 ' + str(step['source_frame']) if step['source_frame'] is not None else '沉降'
                active.append(f"{pending.relative_to(output)} 已记录至{stage}、IPC 子步 {step['native_frame']}。")
            elif reason := initialization_failure(pending):
                status = f'{pending.name} 初始化失败'
                active.append(f"{pending.relative_to(output)} {reason}。 [日志]({(pending/'physics.log').as_posix()})")
        lines.append(f"| L{start} | {7-start} | {len(results)} | {elapsed:.1f} | {', '.join(error) or '—'} | {comparison} | {status} |")
    if active:
        lines += ['', *active]
    lines += ['', '相邻层比较用空间坐标位移 `x_f-rest_f - U(x_c-rest_c)`，按顶点和动作子步求均方根。'
               '同拓扑 Direct Fine 比较直接使用对应顶点。两种误差均以米制世界坐标计算，表中转换为毫米。', '',
              '每层的 `state.npz` 保存全部物理子步的布料位置和速度，`replay.npz` 保存动作帧的真实布料和夹片姿态，'
              '`solver_steps.csv` 保存逐步耗时与 Newton 退出残差。`queue.log` 记录队列阶段，当前物理进度见运行目录的 `step_mapping.jsonl`。', '']
    temporary = output/'REPORT.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(output/'REPORT.md')


def rms(coarse, fine, matrix=None, coarse_rest=None, fine_rest=None):
    total = 0.
    for offset in range(0,len(fine),30):
        lower = coarse[offset:offset+30]
        upper = fine[offset:offset+30]
        if matrix is not None:
            lower = (matrix @ (lower - coarse_rest).reshape(len(lower),-1).T).T.reshape(upper.shape)
            upper = upper - fine_rest
        delta = upper - lower
        total += np.sum(delta**2)
    return 1000 * np.sqrt(total / (len(fine)*fine.shape[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    generate(parser.parse_args().output.resolve())
