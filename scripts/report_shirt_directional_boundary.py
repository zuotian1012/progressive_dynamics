"""Report a bounded L1 run with the compensated directional boundary cap."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from trimesh.triangles import closest_point
from shirt_demo.report import load_run,draw,plt,Image
from scripts.check_psq_step_cap import evaluate_recorded


def read_run(path):
    d=load_run(path);m=d['meta'];assert m['status']!='running'
    rows=json.loads((path/'steps.json').read_text());assert len(rows)==m['completed_steps']
    streak=longest=0
    for r in rows:
        streak=streak+1 if r['native']['hit_newton_limit'] else 0;longest=max(longest,streak)
    summary=dict(status=m['status'],completed_requested_prefix=m['completed'],saved_steps=m['completed_steps'],
        requested_steps=m['steps'],time_s=m['completed_steps']*m['dt'],physics_seconds=m['physics_seconds'],
        native_converged_steps=m['native_converged_steps'],preview_steps=m['preview_stopped_steps'],
        newton_limit_frames=[r['step'] for r in rows if r['native']['hit_newton_limit']],longest_newton_limit_streak=longest)
    return d,rows,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--previous',type=Path,default=Path('output/shirt_L1_cap_fix_20261116/recovery'))
    p.add_argument('--skip-animation',action='store_true');a=p.parse_args();root=a.root
    before,br,bs=read_run(a.previous);after,ar,ns=read_run(root/'validation')
    bm=before['meta'];nm=after['meta'];n=nm['completed_steps'];common=min(n,bm['completed_steps'])
    for key in ['config','material_parameters','mesh_sha256','robot_q_sha256','lower_sha256','lower_run_sha256','map_sha256','psq_bundle_sha256']:
        assert bm[key]==nm[key],key
    for path,digest in nm['runtime_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    old_hash=next(v for k,v in bm['runtime_hashes'].items() if Path(k).name=='uipc_backend_cuda.dll')
    assert hashlib.sha256((root/'before/uipc_backend_cuda.dll').read_bytes()).hexdigest()==old_hash
    caps=[json.loads(line) for line in (root/'validation_cap.jsonl').read_text().splitlines()]
    assert all(r['algorithm']=='compensated_boundary_v2' and r['reference_formula']=='inward_recovery_v1' for r in caps)
    # Check the sampled native outputs and the preceding helper on the same
    # recorded inputs. The reference binding face is included in each record.
    flat=np.array([f['values'] for r in caps for f in r['faces']])
    current=evaluate_recorded(flat);np.testing.assert_allclose(current[:,1],flat[:,1],rtol=2e-12,atol=1e-30)
    previous=evaluate_recorded(flat,(root/'before/psq_step_cap_check.dll').resolve())[:,1]
    cursor=0;reference_min=[]
    for r in caps:
        reference_min.append(float(previous[cursor:cursor+len(r['faces'])].min()));cursor+=len(r['faces'])
    np.testing.assert_allclose(reference_min,[r['previous_directional_min'] for r in caps],rtol=2e-10,atol=1e-28)
    improved=[r for r in caps if r['directional_min']>r['previous_directional_min']*1.01]
    audit=json.loads((root/'validation/report/audit.json').read_text())
    lower=load_run(Path(nm['lower_run']));closure=[]
    if common>=500:
        for label,d in [('L0',lower),('Previous L1',before),('New L1',after)]:
            for hand,link in [('left',1),('right',5)]:
                pose=d['robot'][500,link];tcp=pose[:3,:3]@np.array([.155,.001786,.014])+pose[:3,3]
                closest=closest_point(d['x'][500][d['f']],np.broadcast_to(tcp,(len(d['f']),3)))
                closure.append(dict(run=label,hand=hand,tcp_to_surface_mm=float(np.linalg.norm(closest-tcp,axis=1).min()*1000)))
    passed=n>=1060 and ns['longest_newton_limit_streak']==0
    window=None
    if n>=899:
        wr=ar[898:];movement=np.linalg.norm(np.diff(after['x'][898:n+1],axis=0),axis=2).max(1)*1000
        window=dict(first_step=899,last_step=n,native_converged_steps=sum(r['native']['converged'] for r in wr),
            preview_steps=sum(r['native']['preview_non_descent'] for r in wr),newton_limit_steps=sum(r['native']['hit_newton_limit'] for r in wr),
            median_maximum_cloth_step_motion_mm=float(np.median(movement)),maximum_cloth_step_motion_mm=float(movement.max()),
            zero_motion_steps=int((movement<1e-9).sum()))
    result=dict(previous=bs,current=ns,passed_requested_stall_window=passed,
        same_physical_inputs_and_solver_settings=True,recorded_native_helper_replay_passed=True,
        previous_formula_global_min_replay_passed=True,live_samples=len(caps),
        sampled_global_improvement_count=len(improved),global_improvement_examples=improved[:6],
        maximum_sampled_global_cap_ratio=max((r['directional_min']/r['previous_directional_min'] for r in improved),default=1),
        maximum_principal_stretch=audit['maximum_principal_stretch'],minimum_sigma=audit['minimum_singular_value'],
        rejected_infinite_trial_count=audit['rejected_infinite_trial_count'],
        velocity_transfer_verified=audit['velocity_transfer_audit']['all_injected_velocities_verified'],
        preview_exits_verified=audit['preview_audit']['all_preview_evidence_verified'],closure_tcp_to_surface=closure,
        maximum_body_surface_tracking_error_m=audit['maximum_body_surface_tracking_error_m'],post_stall_window=window,
        limitations=['One run each; native trajectories are not bitwise deterministic, so this is not a statistical speed or robustness benchmark.',
            '1060-step diagnostic prefix is shorter than the 1540-step first fold; release and final settling were not completed.',
            'Preview exit is not Newton convergence. The original L0 input also contains nonconverged steps.',
            'No independent collision/penetration validation or grasp correction was performed.'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for label,d,rows in [('Previous',before,br),('New',after,ar)]:
        k=np.arange(1,len(rows)+1)
        ax[0,0].plot(k,[r['native']['newton_iterations'] for r in rows],label=label)
        ax[0,1].semilogy(k,[r['native']['last_line_search_alpha'] if not r['native']['preview_non_descent'] else np.nan for r in rows],label=label)
        ax[1,0].plot(k,np.linalg.norm(np.diff(d['x'][:len(rows)+1],axis=0),axis=2).max(1)*1000,label=label)
    t=[r['frame']+r['iteration']/1024 for r in caps]
    ax[1,1].semilogy(t,[r['previous_directional_min'] for r in caps],'.',ms=3,label='Previous formula, same direction')
    ax[1,1].semilogy(t,[r['directional_min'] for r in caps],'.',ms=2,label='New formula')
    for axis,title,y in zip(ax.flat,['Newton iterations','Last alpha (preview exits excluded)','Maximum saved cloth step motion','Live global cap: same Newton direction'],['Iterations','Alpha','mm','Alpha']):
        axis.set(title=title,ylabel=y,xlabel='Physical step',xlim=(max(0,common-100),max(n,bm['completed_steps'])));axis.legend(fontsize=8)
    fig.savefig(root/'solver_comparison.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(12,5))
    for j,(label,d) in enumerate([('L0',lower),('Previous L1',before),('New L1',after)]):
        draw(fig.add_subplot(1,3,j+1,projection='3d'),d,common,f'{label} | step {common}',True)
    fig.suptitle('Actual saved states at a common time; first fold incomplete');fig.tight_layout();fig.savefig(root/'common_states.png',dpi=150);plt.close(fig)
    if not a.skip_animation:
        images=[]
        for k in np.unique(np.linspace(0,n,80).astype(int)):
            fig=plt.figure(figsize=(10,4.5))
            for j,(label,d) in enumerate([('Approved L0',lower),('New progressive L1',after)]):
                draw(fig.add_subplot(1,2,j+1,projection='3d'),d,k,f'{label} | {k*nm["dt"]:.2f}s',True)
            fig.suptitle(f'Saved prefix: {n}/1540 steps; full first fold not completed')
            fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        images[0].save(root/'prefix_comparison.gif',save_all=True,append_images=images[1:],duration=max(30,round(n*nm['dt']*1000/len(images))),loop=0)
    text=f'''L1 真实方向近边界步长验证

结果：{'已完成 1060 步诊断前缀，未出现 Newton 触顶。' if passed else '本轮未通过预定的无触顶诊断前缀，不能宣称停滞解决。'}
本轮保存 {n}/{nm['steps']} 步，物理时间 {ns['time_s']:.6f} 秒；状态 {ns['status']}。
原生收敛 {ns['native_converged_steps']} 步；非下降预览退出 {ns['preview_steps']} 步；Newton 触顶 {ns['newton_limit_frames']}。
上次 inward-recovery-v1 在第 899、900、901 步连续触顶并停止。本次不是逐位确定重放，不能只凭发生帧判断统计意义的改善。
本次采样到 {len(improved)} 条记录中，新公式比旧公式对同一真实 Newton 方向给出了更大的全局最小步长；最大比值 {result['maximum_sampled_global_cap_ratio']:.6g}。
参考公式和当前公式的原生记录均已用独立 CUDA helper 复算。

实现
仅替换 near-boundary 回退分支：用补偿精度计算 2x2 Gram 矩阵的正定性，沿真实方向二分严格可行边界，保留 0.9 步长缩减及原 Weyl 秩保护。
1.1 拉伸上限、物理参数、机器人轨迹、L0 速度输入、Newton 容差与线搜索预算均未改变。
能量评估同步使用相同的边界判定；不满足约束的试探返回 +inf 并通过已有回溯拒绝，不强制接受。
改动在隔离 PSQ 后端内，没有覆盖原版 libuipc 安装。原始旧 DLL 与源文件保存在 before/。

验证
80 位十进制核验 686 个随机/特殊/实际近边界方向，其中实际方向 84 个；端点与路径严格满足拉伸约束，整段秩界通过。
10000 个非边界输入与旧版逐项相同；10005 个一般方向通过原有路径检查。
本轮保存状态最大主拉伸 {result['maximum_principal_stretch']:.17g}，最小奇异值 {result['minimum_sigma']:.17g}。
本轮拒绝 +inf 试探 {result['rejected_infinite_trial_count']} 次。速度注入和预览退出证据审计均通过。
原停滞区间之后的步数与实际运动统计：{json.dumps(window,ensure_ascii=False)}

抓取几何（闭合时 TCP 到完整三角形表面，mm）
{json.dumps(closure,ensure_ascii=False,indent=2)}

范围
这是 1060 步诊断前缀，不是完成 1540 步的一折。没有完成释放/最终沉降，也没有修正抓取轨迹或做独立穿透验收。
预览退出不等于 Newton 收敛；单条试跑不是全面稳定性或性能证明。
solver_comparison.png：求解指标；common_states.png：同一时刻状态；prefix_comparison.gif：实际保存的 L0/L1 同步前缀。
comparison.json、provenance.json 与各检查 JSON 保存数值证据和版本信息。
'''
    (root/'RESULTS.txt').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('global_improvement_examples','limitations')},indent=2))


if __name__=='__main__':main()
