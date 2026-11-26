"""Render a completed fresh shirt fold at any level; distinguish coarse exits from convergence."""
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import imageio.v2 as imageio
from progressive.prolongation import frames
from progressive.diagnostics import nearest_material_point
from shirt_demo.report import load_run,draw,plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--movie-frames',type=int,default=180)
    p.add_argument('--solver-label',default='Repaired MAS');args=p.parse_args()
    root=args.run;data=load_run(root);m=data['meta'];n=m['completed_steps'];level=m['level']
    assert m['completed'] and n==m['steps']==3476, 'Do not render an incomplete run as a full fold'
    visible_steps=np.unique(np.r_[np.linspace(0,n,args.movie_frames,dtype=int),[60,596,1076,2216,2936,n]])
    lowest=min(float(data['x'][k,:,2].min()) for k in visible_steps)
    data['plot_bounds']=((.15,1.30),(-.55,.55),(min(.60,lowest-.02),1.28))
    records=json.loads((root/'steps.json').read_text());assert len(records)==n
    native=[r['native'] for r in records]
    preview=[i+1 for i,r in enumerate(native) if r.get('preview_non_descent')]
    other_failures=[i+1 for i,r in enumerate(native) if (not r['converged'] or r['hit_line_search_limit'] or r['hit_newton_limit']) and not r.get('preview_non_descent')]
    log=root.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace')
    pcg=[int(s) for s in re.findall(r'Iterative linear solver iteration count: (\d+)',log)]
    dofs={int(s) for s in re.findall(r'GlobalLinearSystem has (\d+) DoFs',log)}
    assert len(dofs)==1, 'Expected fixed global system dimension'
    dof_count=dofs.pop()
    if m.get('finest_requires_convergence'):
        assert not preview and not other_failures, 'Finest result must satisfy its strict acceptance policy'
    out=root/'report';out.mkdir(exist_ok=True)
    x=data['x'];f=data['f'];inv=frames(x[0],f)[2];stretch=[]
    for k in range(n+1):
        tri=x[k][f];F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)@inv
        a=np.sum(F[:,:,0]**2,axis=1);b=np.sum(F[:,:,1]**2,axis=1);c=np.sum(F[:,:,0]*F[:,:,1],axis=1)
        stretch.append(float(np.sqrt(.5*(a+b+np.sqrt((a-b)**2+4*c*c))).max()))
    np.save(out/'maximum_stretch.npy',stretch)
    pickup={}
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax,(side,j) in zip(axes,[('left',1),('right',5)]):
        wrist=data['robot'][596:1077,j]
        tip=np.einsum('nij,j->ni',wrist[:,:3,:3],[.155,.001786,.014])+wrist[:,:3,3]
        face,weights,distance=nearest_material_point(x[596],f,tip[0])
        point=np.einsum('j,njk->nk',weights,x[596:1077][:,f[face]])
        local=np.einsum('nij,nj->ni',np.linalg.inv(wrist[:,:3,:3]),point-wrist[:,:3,3])
        lift=(point[:,2]-point[0,2])*1000
        jaw_lift=(tip[:,2]-tip[0,2])*1000
        pickup[side]=dict(material_face=int(face),material_weights=np.asarray(weights).tolist(),
            initial_material_distance_mm=float(distance*1000),final_material_lift_mm=float(lift[-1]),
            actual_tip_lift_mm=float(jaw_lift[-1]),maximum_relative_drift_mm=float(np.linalg.norm(local-local[0],axis=1).max()*1000),
            material_height_change_last_second_mm=float(lift[-1]-lift[-121]))
        ax.plot(np.arange(596,1077)/120,lift,label='Tracked cloth material point')
        ax.plot(np.arange(596,1077)/120,jaw_lift,'--',label='Actual gripper tip')
        ax.set(title=side,xlabel='Simulation time (s)',ylabel='Lift from closure (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'pickup_tracking.png',dpi=150);plt.close(fig)
    (out/'pickup_measurements.json').write_text(json.dumps(dict(hands=pickup,
        limitation='Fixed material-point tracking is geometric evidence, not a contact-force or no-slip proof.'),indent=2))
    summary=dict(from_step_zero=True,completed=True,level=level,vertices=m['vertices'],faces=m['faces'],global_dofs=dof_count,steps=n,physical_seconds=n*m['dt'],physics_seconds=m['physics_seconds'],
        native_converged_steps=sum(r['converged'] for r in native),coarse_non_descent_exit_steps=preview,
        other_nonconverged_or_budget_exhausted_steps=other_failures,newton_limit_steps=sum(r['hit_newton_limit'] for r in native),
        maximum_single_pcg_iterations=max(pcg),pcg_cap_count=sum(v>=2*dof_count for v in pcg),
        maximum_newton_iterations=max(r['newton_iterations'] for r in native),maximum_principal_stretch=max(stretch),
        maximum_origin_error_mm=max(max(r['body_origin_tracking_errors_m']) for r in records)*1000,
        all_velocity_injections_verified=all(r['velocity_readback_error']<=1e-12 and r['position_injection_error']<=1e-12 for r in records),
        scope=m['convergence_policy']+' See release_measurements.json for physical release outcome.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    fig=plt.figure(figsize=(15,9))
    for i,(k,label) in enumerate([(60,'Settled'),(596,'Closed'),(1076,'Lift / hold'),(2216,'Fold arc'),(2936,'Open'),(3476,'Final')]):
        ax=fig.add_subplot(2,3,i+1,projection='3d');draw(ax,data,k,f'{label} | step {k}')
    fig.tight_layout(pad=2,h_pad=4);fig.savefig(out/'stages.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection='3d');draw(ax,data,n,f'Fresh L{level} fold | {args.solver_label} | final state');ax.view_init(elev=25,azim=-70)
    ax.plot([1.15,1.15],[-.5,.5],[.8,.8],color='#b35e20',linewidth=2,label='Far table edge');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'final.png',dpi=150);plt.close(fig)
    keys=np.unique(np.linspace(0,n,args.movie_frames,dtype=int));fps=12;speed=(n*m['dt'])/(len(keys)/fps)
    with imageio.get_writer(out/'full_fold.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=2) as writer:
        for k in keys:
            fig=plt.figure(figsize=(8,6),dpi=100);ax=fig.add_subplot(projection='3d')
            draw(ax,data,int(k),f'L{level} full fold | physical time {k*m["dt"]:.2f} s | step {k}')
            count=sum(v<=k for v in preview)
            acceptance='Strict finest convergence' if m.get('finest_requires_convergence') else f'Coarse non-descent exits so far: {count}'
            fig.text(.5,.02,f'{args.solver_label} | {speed:.2f}x playback | {acceptance}',ha='center',fontsize=9)
            fig.tight_layout(rect=[0,.045,1,1]);fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig)
    print('Rendered',out/'full_fold.mp4',flush=True)


if __name__=='__main__':main()
