"""Produce the final Chinese result record only after all ten levels pass."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    chain=json.loads((a.root/'chain.json').read_text());assert chain['status']=='completed'
    data=[]
    for k in range(10):
        run=a.root/f'L{k}';m=json.loads((run/'run.json').read_text());audit=json.loads((run/'audit.json').read_text());extra=json.loads((run/'continuation_audit.json').read_text());steps=json.loads((run/'steps.json').read_text())
        assert m['completed'] and m['completed_steps']==500 and m['dt']==.04
        assert m['per_iteration_acceptance_required'] and extra['passed'] and extra['no_unhandled_line_search_exhaustion']
        assert audit['accepted_prefix_checks_passed'] and not audit['zero_motion_steps']
        assert not any(x['intersections'] for x in audit['intersection_samples'])
        assert not extra['protected_files_changed']
        data.append(dict(level=k,vertices=m['vertices'],faces=m['faces'],wall_seconds=m['wall_seconds'],newton_steps=m['native_steps'],coarse_stops=m['preview_steps'],rescues=extra['rescue_events'],strict_energy_decreases=extra['strict_energy_decreases'],max_newton=max(r['native']['newton_iterations'] for r in steps),maximum_stretch=audit['maximum_stretch'],backend=m['library_sha256']['uipc_backend_cuda.dll'],preconditioner=m['config']['linear_system']['fem_preconditioner']))
    assert data[-1]['newton_steps']==500 and data[-1]['coarse_stops']==0
    assert len(set(d['backend'] for d in data))==1 and all(d['preconditioner']=='diag' for d in data)
    lines=['旗帜 L0–L9 完整结果：逐层、逐次验收通过','',
        '每层 500 步 × 0.04 秒 = 20 秒。所有层使用同一独立运行库；L1–L9 使用相邻层 VelPro 速度输入。',
        '线性求解相对容差 1e-3；Newton 速度增量容差 0.05；线搜索上限 8；Newton 预算 300；diag 预条件器。',
        '半隐式、强制更新、小 alpha 停止均关闭。粗层仅允许经过三尺度证据验证的非下降停止，L9 必须 Newton 收敛。','',
        '层级 | 顶点/三角形 | 墙钟秒 | Newton 收敛步 | 粗层停止步 | 完整梯度重试次数 | 最大 Newton 次数 | 最大拉伸比']
    for d in data:
        lines.append(f'L{d["level"]} | {d["vertices"]}/{d["faces"]} | {d["wall_seconds"]:.3f} | {d["newton_steps"]} | {d["coarse_stops"]} | {d["rescues"]} | {d["max_newton"]} | {d["maximum_stretch"]:.9f}')
    lines.extend(['','修复经过：',
        '1. 原始 L3 在第 442 步被拒绝。8 次线搜索均未下降，微小尺度方向诊断受浮点噪声影响，不能判为可靠的非下降。',
        '2. 检查还发现旧验收只看帧最终状态，可能掩盖此前未处理的线搜索失败。旧 L0/L1/L2 分别有 25/143/46 条此类记录。因此旧结果保留作对照，正式结果从 L0 全部重新计算。',
        '3. 新实现恢复当前迭代状态，在无法可靠判定的粗层线搜索失败处使用完整 PSQ 梯度重试；沿用原可行步、CCD 和 8 次线搜索，必须获得非增能量，失败则恢复并报错。通过后恢复普通近似梯度迭代。',
        f'4. 本次完整链共触发 {sum(d["rescues"] for d in data)} 次重试，其中 {sum(d["strict_energy_decreases"] for d in data)} 次严格降低全能量。没有未处理的内部线搜索耗尽。',
        '5. 这是显式修复扩展，不应称为未修改的论文算法。','',
        '验收范围：',
        '全部 5000 个时间步完成；保存状态均有限、固定边界不动、拉伸比不超过 1.1、无冻结步。',
        'VelPro 写入速度的回读误差及写入前后位置误差均为 0；每层 9 个抽样状态未检测出自交。抽样检查不等于独立的连续碰撞证明。',
        '90 个受保护文件哈希保持一致。没有修改叠衣 demo 的共享源码、默认运行库或已有通过结果。','',
        '展示素材：report/hierarchy.mp4（L0/L3/L6/L9，同相机、同时间、原速）；report/hierarchy.png（十层四时刻）；report/metrics.png（时间与运动统计）；failure_and_repair.png（失败与修复证据）。',
        '报告中的耗时为共享 GPU 环境下的实测墙钟时间，包含本层初始化与仿真循环，不含资产导出、后验验收与渲染；不可直接作为论文性能对比。',
        '层数、顶点范围、厚度、应变限制和目标时长沿用此前论文规模设计；几何、材料、外力等仍是本地重建输入。没有 Direct 对照或严格论文性能复现结论。','',
        '重现入口：.venv/Scripts/python.exe scripts/continue_flag_strict.py --output <新的输出目录>',
        '构建入口：scripts/build_flag_l3_fix.py --variant rescue；独立源码：native/flag_l3_repair；独立运行库：output/flag_L3_fix_20261118/rescue_runtime/src。'])
    (a.root/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=['run_flag_fine.py','continue_flag_strict.py','audit_flag_fine.py','audit_flag_fine_search.py','audit_paper_run.py','audit_paper_non_descent.py','audit_flag_refined_probes.py','report_flag_fine.py','report_flag_failure.py','summarize_flag_strict.py']
    manifest={str((ROOT/'scripts'/f).resolve()):hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in files}
    (a.root/'final_validation.json').write_text(json.dumps(dict(passed=True,levels=data,validation_scripts_sha256=manifest),indent=2))
    print(json.dumps(data,indent=2))

if __name__=='__main__':main()
