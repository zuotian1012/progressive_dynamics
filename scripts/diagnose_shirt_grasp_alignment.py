"""Locate pre-grasp cross-level drift without changing robot or cloth states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz
from trimesh.triangles import closest_point,points_to_barycentric
from progressive.prolongation import ShellProlongation
from shirt_demo.assets import Robot,LINKS
from shirt_demo.report import load_run,draw,plt


def nearest(vertices,faces,point):
    points=closest_point(vertices[faces],np.broadcast_to(point,(len(faces),3)))
    distances=np.linalg.norm(points-point,axis=1);i=int(distances.argmin())
    return float(distances[i]),points[i],i


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lower',type=Path,required=True)
    p.add_argument('--fine',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    data=[load_run(a.lower),load_run(a.fine)];m=[d['meta'] for d in data]
    assert min(q['completed_steps'] for q in m)>=500
    x=[d['x'] for d in data];f=[d['f'] for d in data];poses=[d['robot'] for d in data]
    rows=[json.loads((path/'steps.json').read_text()) for path in [a.lower,a.fine]]
    assets=Path(m[1]['parameters']['assets']);U=load_npz(assets/'map_0_1.npz')
    P=ShellProlongation(x[0][0],x[1][0],f[0],U,np.load(assets/'anchors_0_1.npy'))
    np.testing.assert_allclose(P.position(x[0][0]),x[1][0],atol=1e-12,rtol=0)
    robot=Robot();tcp_local=np.array([.155,.001786,.014]);samples=[]
    indices=sorted(set(range(0,501,2))|{1,3,5,10,15,20,30,40,60,120,240,360,420,450,480,498,500})
    for k in indices:
        predicted=P.position(x[0][k]);diff=x[1][k]-predicted
        linear_diff=(x[1][k]-x[1][0])-U@(x[0][k]-x[0][0]);offset_change=P.offset(x[0][k])-P.offset(x[0][0])
        np.testing.assert_allclose(diff,linear_diff-offset_change,atol=1e-12,rtol=0)
        if k<=60:q=robot.q[0]
        else:
            frame,sub=divmod(k-1-60,2)
            q=robot.q[max(0,frame-1)]+(sub+1)/2*(robot.q[frame]-robot.q[max(0,frame-1)])
        target=robot.poses(q)
        row=dict(step=k,time_s=k*m[1]['dt'],rms_psq_difference_mm=float(np.sqrt(np.mean(np.sum(diff*diff,axis=1)))*1000),
            max_psq_difference_mm=float(np.linalg.norm(diff,axis=1).max()*1000),
            rms_linear_displacement_difference_mm=float(np.sqrt(np.mean(np.sum(linear_diff**2,axis=1)))*1000),
            rms_offset_change_mm=float(np.sqrt(np.mean(np.sum(offset_change**2,axis=1)))*1000),hands=[])
        for hand,link in [('left',1),('right',5)]:
            target_tcp=target[LINKS[link]][:3,:3]@tcp_local+target[LINKS[link]][:3,3]
            for level in [0,1]:
                pose=poses[level][k,link];tcp=pose[:3,:3]@tcp_local+pose[:3,3]
                dist,point,face=nearest(x[level][k],f[level],tcp)
                row['hands'].append(dict(level=level,hand=hand,tcp=tcp.tolist(),closest_point=point.tolist(),face=face,
                    tcp_surface_distance_mm=dist*1000,cloth_minus_tcp_mm=((point-tcp)*1000).tolist(),
                    tcp_target_error_mm=float(np.linalg.norm(tcp-target_tcp)*1000)))
            fine_tcp=np.array(row['hands'][-1]['tcp'])
            dist,point,face=nearest(predicted,f[1],fine_tcp)
            bary=points_to_barycentric(predicted[f[1][face]][None],point[None])[0]
            material_point=bary@x[1][k][f[1][face]]
            row['hands'].append(dict(level='P(L0)',hand=hand,tcp_surface_distance_mm=dist*1000,
                closest_point=point.tolist(),face=face,barycentric=bary.tolist(),
                corresponding_actual_material_point=material_point.tolist(),
                corresponding_material_error_mm=((material_point-point)*1000).tolist(),
                corresponding_material_tcp_distance_mm=float(np.linalg.norm(material_point-fine_tcp)*1000)))
        samples.append(row)
    stages=[]
    for level,r in enumerate(rows):
        for start,end in [(1,10),(1,60),(61,240),(241,360),(361,500)]:
            rr=r[start-1:end]
            stages.append(dict(level=level,first_step=start,last_step=end,
                native_converged=sum(q['native']['converged'] for q in rr),preview=sum(q['native']['preview_non_descent'] for q in rr),
                newton_limits=sum(q['native']['hit_newton_limit'] for q in rr)))
    closure=samples[-1];right=[r for r in closure['hands'] if r['hand']=='right']
    result=dict(scope='Read-only diagnosis of saved states; no target, material, integrator or state was changed.',
        lower=str(a.lower.resolve()),fine=str(a.fine.resolve()),initial_psq_rest_matches=True,
        stage_solve_status=stages,samples=samples,
        closure_right=right,
        first_preview_steps=[next(q['step'] for q in r if q['native']['preview_non_descent']) for r in rows],
        facts=['Cross-level drift is present before closure and before either level first exits by preview.',
            'The same fine topology placed at P(L0) is close to the L1 TCP at closure, whereas the actual fine cloth is far away.',
            'The current native adapter transfers lower velocities and preserves fine positions; it has no position-consistency penalty.',
            'The independent Python solver has an optional consistency term; its default weight is zero.'],
        limits=['Geometric distance does not prove a contact-force or friction failure.',
            'This does not isolate how much drift comes from VelPro integration, contact discretization, shared-proxy approximation or preview termination.',
            'The nearest actual surface point can change material identity as the shirt moves; a corresponding-material point is also reported.',
            'The right-gripper collider pose itself is not modified; a per-level target adjustment would be a different boundary-condition experiment.'])
    (a.output/'diagnosis.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained');ks=[r['step'] for r in samples]
    for level,label in [(0,'L0 actual'),(1,'L1 actual'),('P(L0)','Mapped L0 surface')]:
        ax[0].plot(ks,[next(h['tcp_surface_distance_mm'] for h in r['hands'] if h['hand']=='right' and h['level']==level) for r in samples],label=label)
    ax[0].set(title='Right TCP to cloth surface',ylabel='Distance (mm)',ylim=(0,260));ax[0].legend(fontsize=8)
    ax[1].plot(ks,[r['rms_psq_difference_mm'] for r in samples],label='Actual L1 minus P(L0)')
    ax[1].plot(ks,[r['rms_offset_change_mm'] for r in samples],label='PSQ offset change only')
    ax[1].set(title='Cross-level shape difference',ylabel='Vertex RMS (mm)');ax[1].legend(fontsize=8)
    for level in [0,1]:
        ax[2].plot(ks,[next(h['tcp_target_error_mm'] for h in r['hands'] if h['hand']=='right' and h['level']==level) for r in samples],label=f'L{level}')
    ax[2].set(title='Right TCP target tracking error',ylabel='Distance (mm)');ax[2].legend(fontsize=8)
    for axis in ax:
        axis.set(xlabel='Physical step');axis.axvline(60,color='gray',ls=':',lw=.8);axis.axvline(500,color='gray',ls=':',lw=.8)
    fig.savefig(a.output/'alignment_metrics.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(14,10))
    for j,k in enumerate([60,500]):
        mapped=dict(data[1]);mapped['x']=np.array([P.position(x[0][k])]);mapped['robot']=np.array([poses[1][k]])
        for i,(label,d,index) in enumerate([('L0',data[0],k),('Actual L1',data[1],k),('Mapped P(L0), diagnostic only',mapped,0)]):
            axis=fig.add_subplot(2,3,j*3+i+1,projection='3d');draw(axis,d,index,f'{label} | step {k}',True)
            if k==500:
                pose=poses[1][k,5];tcp=pose[:3,:3]@tcp_local+pose[:3,3];axis.scatter(*tcp,color='red',s=30,label='L1 right TCP')
    fig.suptitle('Mapped surface is a geometric diagnostic, not an L1 simulation result')
    fig.tight_layout(rect=(.02,0,1,.96),h_pad=4);fig.savefig(a.output/'alignment_states.png',dpi=140,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    selected={r['step']:r for r in samples}
    lines=['L1 抓取偏差诊断（只读）','',
        f"初态 PSQ 映射与 L1 静止网格一致；第 10 步两层均为原生收敛，布形 RMS 差已达 {selected[10]['rms_psq_difference_mm']:.3f} mm。",
        f"第 60 步沉降结束 RMS 差 {selected[60]['rms_psq_difference_mm']:.3f} mm；第 500 步闭合时 {closure['rms_psq_difference_mm']:.3f} mm。",
        '右侧闭合几何：'+json.dumps(right,ensure_ascii=False,indent=2),'',
        '结论：实际 L1 布形在闭合前已偏离 L0，原抓手轨迹因此没有夹到预期布面。映射后的 L0 布面与抓手仍接近，初态映射并未错位。',
        '当前接口仅传递 U*v_L0，保留 L1 独立位置；它没有自动消除已积累位置差的一致性项。误差已在首次预览退出前出现，不能全部归咎于预览退出。',
        '这些结果没有分离积分方案、接触离散、共享代理近似和预览退出各自的贡献，不宣称唯一动力学根因。',
        '建议先固定机器人轨迹，做沉降—闭合短对照，检验原生后端接入已有质量加权一致性项后，抓取区域的位置误差能否收敛到可夹持范围；必须同时审计接触、应变与跨层误差。',
        '单独移动 L1 抓手属于改变层间边界条件，应明确另列为抓取自适应实验。没有把任何映射布形写入仿真状态，也没有将衣服绑定到抓手。',
        '', 'alignment_metrics.png：距离/误差曲线；alignment_states.png：沉降与闭合状态；diagnosis.json：全部记录。']
    (a.output/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (a.output/'provenance.json').write_text(json.dumps(dict(inputs={str(p):digest(p) for base in [a.lower,a.fine] for p in [base/'run.json',base/'positions.npy',base/'robot.npy']},script=digest(__file__)),indent=2))
    print(json.dumps(dict(closure_right=right,first_preview_steps=result['first_preview_steps'],shape_rms_mm={k:selected[k]['rms_psq_difference_mm'] for k in [10,60,240,500]}),indent=2))


if __name__=='__main__':main()
