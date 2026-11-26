"""Write a compact, evidence-linked catalogue of the two paper-scale experiments."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();rows=[]
    for stage in ['flag_L0_boundary','flag_tiny_probe/L0','bunny_tiny_probe/L0','gamma_short/flag/L0','gamma_short/flag/L1','gamma_short/bunny/L0','gamma_short/bunny/L1']:
        path=root/stage
        if (path/'run.json').exists():rows.append((stage,path))
    for scene in ['flag','bunny']:
        for path in sorted((root/'gamma_full'/scene).glob('L[0-9]'),key=lambda p:int(p.name[1:])):
            if (path/'run.json').exists():rows.append((str(path.relative_to(root)),path))
        for path in sorted((root/f'{scene}_budget1000').glob('L[0-9]'),key=lambda p:int(p.name[1:])):
            if (path/'run.json').exists():rows.append((str(path.relative_to(root)),path))
    for path in sorted((root/'bunny_event_window').glob('L[0-9]'),key=lambda p:int(p.name[1:])):
        if (path/'run.json').exists():rows.append((str(path.relative_to(root)),path))
    for name in ['flag_L7_fixed','bunny_L3_fixed_contact']:
        path=root/'mas_probe'/name
        if (path/'run.json').exists():rows.append((str(path.relative_to(root)),path))
    records=[]
    for name,path in rows:
        try:m=json.loads((path/'run.json').read_text())
        except json.JSONDecodeError:continue
        entry=dict(path=name,scene=m['scene'],level=m['level'],vertices=m['vertices'],proxy_vertices=m['proxy_vertices'],
            status=m['status'],accepted_steps=m['completed_steps'],requested_steps=m['steps'],dt=m['dt'],
            native_steps=m['native_steps'],preview_steps=m['preview_steps'],small_step_preview_steps=m.get('small_step_preview_steps',0),
            video=str(path/'report/accepted.mp4') if (path/'report/accepted.mp4').exists() else None)
        if (path/'audit.json').exists():entry['audit']=json.loads((path/'audit.json').read_text())
        records.append(entry)
    result=dict(paper='https://pcs-sim.github.io/pd%2B%2B/PD%2B%2B_main.pdf',psq='https://pcs-sim.github.io/psq/psq-combined.pdf',
        complete_paper_reproduction_claim=False,
        confirmed_settings=dict(flag=dict(levels=10,approx_vertices=[550,35000],thickness_m=.0004,strain_limit=1.1,duration_s=20),
            bunny=dict(levels=8,approx_vertices=[1500,90000],thickness_m=.00007,strain_limit=1.068,friction=.3,dt=.01,expected_event='contact followed by sliding off')),
        local_choices=['Reconstructed meshes, dimensions and placement','E=30000 Pa, nu=.3, rho=200 kg/m3',
            'Flag dt=.04 inferred from the PD2024 description; body acceleration y=8+5*sin(2*pi*1.3*t)',
            'Bunny 4 s coarse observation and 1 s fine contact window are local choices; paper duration not recovered','PSQ gamma=1e-8 for small-step previews, author benchmark gamma not recovered'],
        backend_differences=['Native max-axis velocity decrement .001 rather than paper RMS','Native adaptive contact stiffness','Fixed strain barrier stiffness'],
        execution_conditions='Some budget diagnostics shared the GPU. Wall times are not matched performance benchmarks.',
        runs=records,
        latest_chains={branch:json.loads((root/branch/'chain.json').read_text()) for branch in ['flag_budget1000','bunny_event_window'] if (root/branch/'chain.json').exists()})
    probe=root/'mas_probe/flag_L7_diagnostic/run.json'
    if probe.exists():
        pm=json.loads(probe.read_text())
        result['rejected_acceleration_probe']=dict(path=str(probe.relative_to(root)),
            preconditioner='mas',accepted_steps=pm['completed_steps'],error=pm.get('error'),
            pin_error_m=pm.get('rejected_step',{}).get('pin_error_m'),adopted=False)
    fixed_probe=root/'mas_probe/fixed_projection_comparison.json'
    if fixed_probe.exists():
        result['isolated_mas_fix_diagnostic']=json.loads(fixed_probe.read_text())
    (root/'catalog.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['PD++ 旗帜与兔子：论文规模重建实验','',
        '当前不声明完整复现论文，也不声明 VelPro 性能或时间连续性已被证明。',
        '所有新运行关闭原生 semi-implicit 与强制更新。Newton 收敛、已验证非下降停止、PSQ 小步长停止分别记录。',
        'PSQ 小步长停止来自 Algorithm 1 的 alpha<gamma 规则；gamma=1e-8 是本地选择。最细层禁止该停止。','',
        '已对齐：旗帜 10 层（551→35,000 顶点）、0.4 mm 厚、10% 应变限制、20 s 目标；兔子 8 层（1,443→90,000 顶点）、0.07 mm 厚、6.8% 应变限制、摩擦 0.3、dt=0.01 s。',
        '尚未恢复作者原始网格、完整材料与外力配置。旗帜 dt=.04 为推定，兔子 4 s 粗层与 1 s 细层观察窗口均为本地选择。',
        '论文兔子场景预期先碰撞再滑落，滑落本身不是失败。','',
        '兔子细层首次对比采用前 1 秒接触—滑落窗口（论文未给出该例总时长）。粗层 4 秒长测保留；L4 的 4 秒请求主动停止后，仅提取已经验收的前 100 步作为显式标注的 1 秒子窗口，并继续 L5–L7。原 4 秒运行没有改称完成。','',
        '可展示素材：report/*_hierarchy.png 是网格规模图；各运行 report/accepted.mp4 和 .gif 是真实已接受状态；audit.json 是独立输出检查；comparison/level_metrics.png 是几何诊断。',
        '几何 RMS 差异与速度曲线不是论文 Eq.12 时间连续性指标。早期视频可能只展示前 1 或 5 秒；最终旗帜对比请求完整 20 秒，并受所选层共同已接受时间限制。片尾时间和字幕明确标注；完整已接受轨迹在 positions.npy 中。','',
        '已发现：旗帜长测 L0 的后约 16 s 完全静止，不能把完成 500 步等同于成功持续摆动。严格非下降分支旗帜第 16 步、兔子第 44 步被拒绝，失败之前的轨迹单独保留。',
        'flag_L0 为最初只选到单个固定点的无效设置试验，已标为 EXCLUDED，不用作算法失败证据。','',
        '预算对照保持材料、时间步、容差不变，只把 max_iter 从 300 提升至 1000。但旗帜重跑前段已出现偏差，不能仅据越过旧失败时刻建立预算变化的因果结论。部分对照并用 GPU，耗时不能作为公平性能比较。','',
        '运行状态（生成本文件时）：']
    for r in records:
        lines.append(f'{r["path"]}: {r["status"]}; {r["accepted_steps"]}/{r["requested_steps"]} 步；Newton={r["native_steps"]}, preview={r["preview_steps"]}')
    if 'rejected_acceleration_probe' in result:
        lines+=['',f"可选 MAS 加速探针未采用：第一步虽报 Newton 收敛，但固定点偏移 {result['rejected_acceleration_probe']['pin_error_m']:.3g} m，超过当前 1e-10 m 固定点检查；0 步验收。未据此判断根因，主线保持 diag。"]
    if 'isolated_mas_fix_diagnostic' in result:
        lines+=['','独立 MAS 修正库：将预条件器输出投影回自由 DOF 后，单步固定点偏移为 0，验收通过；线性迭代总数从 27580 降至 3565。但与 diag 的最终位置最大差约 7.17 mm，尚未建立轨迹等价性或公平耗时加速结论，未用于这两组主线。详情见 mas_probe/fixed_projection_comparison.json；默认运行库未替换。']
        lines+=['另有兔子 L3 的修正 MAS 接触探针完成 50 步（0.5 s），16 步 Newton 收敛、34 步粗层小步长停止，最大拉伸 1.05708；9 个抽样时刻无自交。该探针单列，未替代主线轨迹。']
    lines+=['','下一步证据重点：粗层小步长停止为何频繁出现；冻结时的搜索方向与真实 PSQ 势能是否一致；严格最细层结果；同物理配置的 Direct / Semi / Full / VelPro 对照。',
        '',result['paper'],result['psq']]
    (root/'说明.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(root/'说明.txt')

if __name__=='__main__':main()
