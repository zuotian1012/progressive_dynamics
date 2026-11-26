"""Render actual saved L0/L1/L2 states for a progressive finest-level run."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz
from trimesh.triangles import closest_point
from progressive.prolongation import ShellProlongation
from shirt_demo.report import load_run,draw,plt,Image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--l0',type=Path,default=Path('output/shirt_pd_first_fold_20261116/L0'))
    p.add_argument('--l1',type=Path,default=Path('output/shirt_L1_directional_boundary_20261116/validation'))
    p.add_argument('--l2',type=Path,help='L2 run to display (default: root/L2)')
    p.add_argument('--skip-animation',action='store_true');a=p.parse_args();root=a.root
    paths=[a.l0,a.l1,a.l2 or root/'L2'];data=[load_run(path) for path in paths];m=[d['meta'] for d in data]
    assert m[2]['status']!='running' and m[2]['finest_requires_convergence']
    n=m[2]['completed_steps'];assert n>0 and all(v['completed_steps']>=n for v in m[:2])
    for v in m[:2]:
        expected_config=copy.deepcopy(v['config'])
        budget=m[2]['parameters'].get('line_search_max_iterations')
        if budget is not None:expected_config['line_search']['max_iter']=budget
        assert v['dt']==m[2]['dt'] and expected_config==m[2]['config'] and v['robot_q_sha256']==m[2]['robot_q_sha256'] and v['material_parameters']==m[2]['material_parameters']
    rows=[json.loads((path/'steps.json').read_text()) for path in paths]
    assert len(rows[2])==n
    assert all(r['native']['converged'] and not r['native']['preview_non_descent'] and not r['native']['hit_newton_limit'] and not r['native']['hit_line_search_limit'] for r in rows[2])
    audit=json.loads((paths[2]/'report/audit.json').read_text());assert audit['velocity_transfer_audit']['all_injected_velocities_verified']
    assets=Path(m[2]['parameters']['assets']);transfers=[]
    for level in [1,2]:
        transfers.append(ShellProlongation(data[level-1]['x'][0],data[level]['x'][0],data[level-1]['f'],
            load_npz(assets/f'map_{level-1}_{level}.npz'),np.load(assets/f'anchors_{level-1}_{level}.npy')))
    samples=[]
    for k in sorted({0,min(60,n),min(500,n),min(788,n),n}):
        row=dict(step=k,time_s=k*m[2]['dt'],cross_level_rms_mm=[],right_tcp_surface_mm=[])
        for level,P in enumerate(transfers,1):
            delta=data[level]['x'][k]-P.position(data[level-1]['x'][k])
            row['cross_level_rms_mm'].append(float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))*1000))
        for d in data:
            pose=d['robot'][k,5];tcp=pose[:3,:3]@np.array([.155,.001786,.014])+pose[:3,3]
            cp=closest_point(d['x'][k][d['f']],np.broadcast_to(tcp,(len(d['f']),3)))
            row['right_tcp_surface_mm'].append(float(np.linalg.norm(cp-tcp,axis=1).min()*1000))
        samples.append(row)
    summary=[]
    for d,rr in zip(data,rows):
        mm=d['meta'];summary.append(dict(level=mm['level'],vertices=mm['vertices'],faces=mm['faces'],
            saved_steps=mm['completed_steps'],matched_prefix_steps=n,
            native_converged_in_prefix=sum(r['native']['converged'] for r in rr[:n]),
            preview_steps_in_prefix=sum(r['native']['preview_non_descent'] for r in rr[:n]),
            newton_limits_in_prefix=sum(r['native']['hit_newton_limit'] for r in rr[:n])))
    result=dict(completed_requested_prefix=m[2]['completed'],status=m[2]['status'],L2_saved_steps=n,requested_steps=m[2]['steps'],
        physical_seconds=n*m[2]['dt'],full_first_fold_steps=1540,full_first_fold_completed=n>=1540,
        levels=summary,geometry_samples=samples,L2_physics_seconds=m[2]['physics_seconds'],
        L2_maximum_principal_stretch=audit['maximum_principal_stretch'],L2_minimum_sigma=audit['minimum_singular_value'],
        velocity_transfer_audit=audit['velocity_transfer_audit'],all_saved_L2_steps_native_converged=True,
        L2_rejected_infinite_trials=audit['rejected_infinite_trial_count'],
        run_paths=[str(path.resolve()) for path in paths],L2_solver_overrides=m[2]['solver_overrides'],
        failed_step=json.loads((paths[2]/'failed_step.json').read_text()) if (paths[2]/'failed_step.json').exists() else None,
        limitations=['Only actual saved states through the common prefix are shown; no upsampled cloth is presented as simulation.',
            'L0/L1 inputs include preview exits; L0 also has Newton-limit steps, recorded in the prefix summary.',
            'L2 uses the retained native default tolerance with semi-implicit disabled; this is not a claim of the paper RMS tolerance.',
            'Grasp misalignment is intentionally retained. No position-consistency penalty, attachments, or robot-target adjustment was added.',
            'No independent contact penetration/self-intersection certification or performance benchmark.'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
    for level,rr in enumerate(rows):
        ax[0].plot(np.arange(1,n+1),[r['native']['newton_iterations'] for r in rr[:n]],label=f'L{level}')
    ax[0].set(ylabel='Newton iterations',title='Actual native solver counts');ax[0].legend()
    for level,P in enumerate(transfers,1):
        ks=np.unique(np.linspace(0,n,101).astype(int));error=[]
        for k in ks:
            delta=data[level]['x'][k]-P.position(data[level-1]['x'][k]);error.append(float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))*1000))
        ax[1].plot(ks,error,label=f'L{level} vs P(L{level-1})')
    ax[1].set(xlabel='Physical step',ylabel='Vertex RMS difference (mm)',title='Cross-level shape difference');ax[1].legend()
    fig.savefig(root/'metrics.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(15,5))
    for level,d in enumerate(data):
        draw(fig.add_subplot(1,3,level+1,projection='3d'),d,n,f'L{level} | {m[level]["faces"]:,} faces | {n*m[2]["dt"]:.2f}s',level<2)
    fig.suptitle(f'Actual states at step {n}; full first fold not completed')
    fig.tight_layout();fig.savefig(root/'last_saved_comparison.png',dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    if not a.skip_animation:
        images=[]
        for k in np.unique(np.linspace(0,n,70).astype(int)):
            fig=plt.figure(figsize=(15,4.8))
            for level,d in enumerate(data):
                draw(fig.add_subplot(1,3,level+1,projection='3d'),d,k,f'L{level} | {m[level]["faces"]:,} faces | {k*m[2]["dt"]:.2f}s',level<2)
            fig.suptitle(f'Accepted prefix only: {n}/{m[2]["steps"]} requested steps | L2 status: {m[2]["status"]} | physical time shown in panels')
            fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        images[0].save(root/'three_levels.gif',save_all=True,append_images=images[1:],duration=max(30,round(n*m[2]['dt']*1000/len(images))),loop=0)
    text=f'''L0 -> L1 -> L2 原生 PSQ/VelPro 仿真

L2 状态：{m[2]['status']}；保存 {n}/{m[2]['steps']} 步，物理时间 {n*m[2]['dt']:.6f} 秒。
L2 网格：27811 顶点 / 55068 三角形。L2 全部 {n} 个保存步达到原生收敛，没有预览退出、Newton 触顶或线搜索预算耗尽接受。
首次失效步如存在，保存在 failed_step.json，未作为接受状态写入动画。
这不是完整 1540 步的一折；上游 L1 只有 1060 步，因此本轮最多推进同一前缀。

实现与检查
直接复用已完成的 L1 轨迹，以 map_1_2 的 U 传递速度；L2 位置始终由自己的动力学求解得到，没有用 P(L1) 覆盖位置。
L2 的最细弹性代理为恒等映射，与 L0/L1 共享相同 55068 面静止几何和材料参数。
身份算子的位置/梯度测试、62 步接入短测、逐步速度读回和位置不变检查通过。
没有加入位置一致性项，没有修改机器人轨迹、抓取点或绑定衣服；保留用户接受的抓取偏差。
本轮没有修改原生后端二进制；使用前一轮通过验证的补偿方向步长实现。
L2 最大主拉伸：{result['L2_maximum_principal_stretch']:.17g}；最小奇异值：{result['L2_minimum_sigma']:.17g}；拒绝 +inf 试探 {result['L2_rejected_infinite_trials']} 次。
L2 求解计时：{m[2]['physics_seconds']:.3f} 秒（含速度传递/读回；不是性能基准）。
求解配置覆盖项：{json.dumps(m[2]['solver_overrides'])}；线搜索预算调整只增加回溯次数，不放宽收敛或能量接受条件。

几何采样
cross_level_rms_mm 顺序为 L1 对 P(L0)、L2 对 P(L1)；right_tcp_surface_mm 顺序为 L0、L1、L2。
{json.dumps(samples,ensure_ascii=False,indent=2)}

限制
上游轨迹包含非收敛预览，不能把完整三级链条称为全部 Newton 收敛。
沿用原生默认容差，不宣称复现论文的 RMS 收敛阈值。未做独立自交/穿透认证或抓取成功验收。
动画与快照只使用实际保存的同一时刻状态，最细层展示原始完整三角网格。
'''
    (root/'RESULTS.txt').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('velocity_transfer_audit','limitations')},indent=2))


if __name__=='__main__':main()
