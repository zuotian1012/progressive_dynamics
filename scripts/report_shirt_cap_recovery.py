"""Compare a recorded L1 cap stall with a bounded recovery validation run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from trimesh.triangles import closest_point
from shirt_demo.report import load_run,draw,plt


def summarize(root):
    m=json.loads((root/'run.json').read_text());assert m['status']!='running'
    rows=json.loads((root/'steps.json').read_text());assert len(rows)==m['completed_steps']
    streak=longest=0
    for r in rows:
        streak=streak+1 if r['native']['hit_newton_limit'] else 0
        longest=max(longest,streak)
    return m,rows,dict(status=m['status'],requested_steps=m['steps'],saved_steps=m['completed_steps'],
        interrupted_uncommitted_step=m.get('interrupted_uncommitted_step'),
        saved_time_s=m['completed_steps']*m['dt'],native_converged_steps=m['native_converged_steps'],
        preview_steps=m['preview_stopped_steps'],newton_limit_frames=[r['step'] for r in rows if r['native']['hit_newton_limit']],
        longest_newton_limit_streak=longest,physics_seconds=m['physics_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root;bm,br,bs=summarize(root/'baseline');nm,nr,ns=summarize(root/'recovery')
    for k in ['config','material_parameters','mesh_sha256','robot_q_sha256','lower_sha256','lower_run_sha256','map_sha256','psq_bundle_sha256']:
        assert bm[k]==nm[k],k
    data=[load_run(root/s) for s in ['baseline','recovery']]
    n=min(bm['completed_steps'],nm['completed_steps'])
    caps=[json.loads(line) for line in (root/'recovery_cap.jsonl').read_text().splitlines()]
    # Unlike an offline replay of selected faces, these minima were reduced
    # over every face for the same actual Newton direction in the live run.
    recovery=[r for r in caps if r['recovered_boundary_faces']>0]
    global_improved=[r for r in caps if r['directional_min']>r['previous_directional_min']]
    audits=[json.loads((root/s/'report/audit.json').read_text()) for s in ['baseline','recovery']]
    lower=load_run(Path(nm['lower_run']))
    closure=[]
    if n>=500:
        for label,d in [('L0',lower),('L1 before',data[0]),('L1 recovery',data[1])]:
            for hand,link in [('left',1),('right',5)]:
                pose=d['robot'][500,link]
                tcp=pose[:3,:3]@np.array([.155,.001786,.014])+pose[:3,3]
                cp=closest_point(d['x'][500][d['f']],np.broadcast_to(tcp,(len(d['f']),3)))
                closure.append(dict(run=label,hand=hand,step=500,tcp_to_surface_mm=float(np.linalg.norm(cp-tcp,axis=1).min()*1000)))
    remaining_path=json.loads((root/'remaining_boundary_path.json').read_text()) if (root/'remaining_boundary_path.json').exists() else None
    result=dict(same_physical_inputs_and_solver_configuration=True,baseline=bs,recovery=ns,
        candidate_resolves_observed_stall=ns['longest_newton_limit_streak']==0 and ns['saved_steps']>=935,
        remaining_boundary_path=remaining_path,
        baseline_actual_direction_replay=json.loads((root/'actual_direction_checks.json').read_text()),
        live_recovery_audit=dict(sampled_records=len(caps),records_with_recovered_faces=len(recovery),
            records_with_larger_global_cap=len(global_improved),examples=global_improved[:8]),
        geometry=[dict(run=s,maximum_principal_stretch=v['maximum_principal_stretch'],minimum_sigma=v['minimum_singular_value'],
            maximum_body_surface_tracking_error_m=v['maximum_body_surface_tracking_error_m'],
            velocity_transfer_verified=v['velocity_transfer_audit']['all_injected_velocities_verified'],
            preview_exits_verified=v['preview_audit']['all_preview_evidence_verified']) for s,v in zip(['baseline','recovery'],audits)],
        common_saved_steps=n,closure_tcp_to_surface=closure,
        maximum_common_prefix_position_difference_m=float(np.max(np.abs(data[0]['x'][:n+1]-data[1]['x'][:n+1]))),
        limitations=['Runs start from the same initial state but are not bitwise deterministic; one run each is not a statistical speed comparison.',
            'The diagnostic prefix does not complete the first fold, release or final settling.',
            'Preview termination is distinct from Newton convergence; right-grasp geometry is not corrected by a step-cap change.',
            'Sampled live cap records may miss recovery events between samples; absence of recorded recovery is not proof it was unused.'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for label,m,rows,d in zip(['Before','Recovery'],[bm,nm],[br,nr],data):
        steps=np.arange(1,len(rows)+1)
        ax[0,0].plot(steps,[r['native']['newton_iterations'] for r in rows],label=label)
        alpha=[r['native']['last_line_search_alpha'] if not r['native']['preview_non_descent'] else np.nan for r in rows]
        ax[0,1].semilogy(steps,alpha,label=label)
        motion=np.linalg.norm(np.diff(d['x'][:len(rows)+1],axis=0),axis=2).max(1)*1000
        ax[1,0].plot(steps,motion,label=label)
    if caps:
        t=np.array([r['frame'] for r in caps])+np.array([r['iteration'] for r in caps])/1024
        ax[1,1].semilogy(t,[r['previous_directional_min'] for r in caps],'.',ms=2,label='Previous formula on same direction')
        ax[1,1].semilogy(t,[r['directional_min'] for r in caps],'.',ms=2,label='Recovery formula')
    for axis,title,y in zip(ax.flat,['Newton iterations','Last alpha (preview exits excluded)','Maximum saved cloth step motion','Live global caps: identical directions'],['Iterations','Alpha','mm','Alpha']):
        axis.set(title=title,ylabel=y,xlabel='Physical step',xlim=(max(0,n-100),max(bm['completed_steps'],nm['completed_steps'])))
        axis.legend(fontsize=8)
    fig.savefig(root/'cap_comparison.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(12,10))
    for j,(label,d) in enumerate([('Approved L0',lower),('L1 before',data[0]),('L1 recovery',data[1])]):
        for k,top in enumerate([False,True]):
            axis=fig.add_subplot(2,3,k*3+j+1,projection='3d')
            draw(axis,d,n,f'{label} | step {n} | {n*nm["dt"]:.3f}s',True)
            if top:
                axis.view_init(elev=90,azim=-90);axis.set_zticks([]);axis.set_zlabel('')
    fig.suptitle('Actual saved states at a common time; first fold incomplete')
    fig.tight_layout(rect=(0,0,1,.96),h_pad=3);fig.savefig(root/'common_state_comparison.png',dpi=140);plt.close(fig)
    right=[r for r in closure if r['hand']=='right']
    text=f'''L1 近边界步长回退：真实方向诊断与针对性验证

本轮结论：{'该验证前缀未再观察到触顶，尚不能推广为全面解决。' if result['candidate_resolves_observed_stall'] else '候选修正未消除真实停滞，不能作为已经解决问题的配置。'}

基线：保存 {bs['saved_steps']} 步，Newton 触顶步为 {bs['newton_limit_frames']}。
修正试跑：状态 {ns['status']}，保存 {ns['saved_steps']}/{ns['requested_steps']} 步（{ns['saved_time_s']:.6f} 秒）。
修正试跑的 Newton 触顶步：{ns['newton_limit_frames']}；最长连续触顶 {ns['longest_newton_limit_streak']} 步。
修正试跑原生收敛 {ns['native_converged_steps']} 步，独立标记的预览退出 {ns['preview_steps']} 步。
本轮仅验证诊断前缀，没有完成 1540 步的一折、释放与最终沉降。

真实方向证据
基线的第 3393 面是极小全局步长的实际限制面，与上次保存状态推测的第 844 面不同。
基线第 933 步第 0 次 Newton 方向，该面原 cap=4.375145241665932e-6；候选 helper 对同一原生 F/dF、SVD、范数输入允许 cap=1，201 点路径检查通过。
这证明局部的保守回退问题真实存在，但所保存的限制面不能独自预测其他面修改后的全局最小值。
修正试跑采样到 {len(recovery)} 条记录有近边界面恢复步长，其中 {len(global_improved)} 条记录的全局最小 cap 大于旧公式对同一真实方向的全局最小 cap。
采样并未覆盖每一次 Newton 迭代；未采到事件不能证明分支从未使用。

改动
只在原 directional cap 的近边界回退分支，沿同一方向搜索满足原有 64-epsilon 安全余量的终点。
端点满足上拉伸约束时，用谱范数凸性约束整段；秩保护仍使用原来的 Weyl 界。
找不到可行内点时保留原来的 legacy 界。没有修改 1.1 上限、材料、接触、L0 输入、机器人轨迹、Newton 容差或线搜索预算。
另加可选的连续触顶诊断停止条件；它只在已提交的物理步之间停止，不改变单步求解或强制接受。
改动在隔离的 PSQ 后端内，原版已安装的 libuipc 不受影响。

验证
生产 CUDA helper：10000 个非边界输入与旧版逐项完全一致；10005 个一般方向通过 201 点路径检查。
5 类边界方向通过 1001 点路径检查，包括收缩、向外拉伸、中途秩退化风险、零方向、先内后外；不可行初态仍拒绝。
真实方向已用保留原生 SVD 输入的 CUDA 接口复算；详见 actual_direction_checks.json。
两次试跑的速度注入、位置未被注入覆盖、预览退出证据均经审计；保存状态的拉伸和秩统计见 comparison.json。

抓取问题仍独立存在
闭合第 500 步，右 TCP 到完整布料三角形表面的最近距离（mm）：
{json.dumps(right,ensure_ascii=False,indent=2)}
这是几何检查，不是接触力证明。本轮没有修改抓取策略。

局限
两次运行同初态、同物理输入和求解设置，但不逐位确定；一次运行各一条，不能据此宣称统计性的加速或全面解决停滞。
预览退出不等于 Newton 收敛。未做独立自交/穿透验收。
旧 DLL 和源文件保存在 before/；本轮构建及测试证据保存在 build_provenance.json 和各检查 JSON。
cap_comparison.png：触顶、步长、移动量及同方向全局 cap 对照。
common_state_comparison.png：同一已保存时刻的 L0、修正前 L1、修正后 L1。
'''
    (root/'RESULTS.txt').write_text(text,encoding='utf-8')
    if remaining_path:
        with (root/'RESULTS.txt').open('a',encoding='utf-8') as out:
            out.write(f'''
剩余瓶颈的高精度证据
第 {remaining_path['frame']} 步 Newton {remaining_path['iteration']}，实际全局限制面 {remaining_path['face']}：
原生 cap={remaining_path['native_cap']:.17g}；仅针对这个面的真实 F/dF，以 80 位十进制复算 alpha={remaining_path['checked_alpha']}，整个线段仍严格低于 1.1 的双精度表示，并满足秩条件。
该方向略向外拉伸，不能回到原 64-epsilon 安全余量内，因此本轮“寻找向内终点”的分支无能为力；方向无关的旧界又过分保守。
这只是局部面证明，不是全局可行步长、真实坐标更新、接触 CCD 或能量下降的证明；没有把该大步长强行应用到仿真。
下一项针对性工作：为近边界分支计算真实方向与严格拉伸边界的交点，处理浮点误差，并重新验证全局步长、接触及能量回溯；不应只增加 Newton 次数或放宽 1.1 上限。
证据文件：remaining_boundary_path.json；重算脚本：scripts/check_psq_recorded_boundary_path.py。
''')
    print(json.dumps(dict(baseline=bs,recovery=ns,live_recovered_records=len(recovery),live_global_improvements=len(global_improved)),indent=2))


if __name__=='__main__':main()
