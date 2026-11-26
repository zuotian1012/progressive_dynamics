"""Audit and render a modified complete first fold from actual saved states."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import ipctk
from scipy.spatial import ConvexHull
from trimesh.triangles import closest_point,points_to_barycentric
from shirt_demo.report import load_run,draw,plt,Image
from shirt_demo.hierarchy import validate_level


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);p.add_argument('--skip-animation',action='store_true');a=p.parse_args()
    root=a.experiment/'run';out=root/'report';plan=json.loads((a.experiment/'plan.json').read_text());d=load_run(root);m=d['meta'];s=plan['stages_physical_steps']
    assert m['completed'] and m['completed_steps']==s['observe_end']
    # Keep a fixed camera over the entire run, including cloth falling off the table.
    xmin=d['x'].min(axis=(0,1));xmax=d['x'].max(axis=(0,1))
    d['plot_bounds']=[(min(.15,float(xmin[0])-.03),max(1.18,float(xmax[0])+.03)),
        (min(-.55,float(xmin[1])-.03),max(.55,float(xmax[1])+.03)),
        (min(.78,float(xmin[2])-.03),max(1.26,float(xmax[2])+.03))]
    audit=json.loads((out/'audit.json').read_text());rows=json.loads((root/'steps.json').read_text())
    finest_level=validate_level(m['parameters']['assets'],m['level'])
    prefix=Path(plan['prefix_trajectory']);old=json.loads((prefix.parent/'run/run.json').read_text())
    for key in ['material_parameters','robot_drive_strength']:assert m[key]==old[key],key
    # Identical DLLs may be loaded from an isolated, archived package directory.
    runtime_by_name=lambda value: {Path(path).name: digest for path,digest in value.items()}
    assert runtime_by_name(m['runtime_hashes'])==runtime_by_name(old['runtime_hashes'])
    expected_config=copy.deepcopy(old['config'])
    budget=m['parameters'].get('line_search_max_iterations')
    if budget is not None:
        assert 8<=budget<=128 and m['solver_overrides']['line_search.max_iter']==budget
        expected_config['line_search']['max_iter']=budget
    assert m['config']==expected_config
    if m['level']==0:
        if audit.get('revised_finest_verified'):
            assert audit['rebuilt_hierarchy_verified']
        else:
            for key in ['mesh_sha256','psq_bundle_sha256']:assert m[key]==old[key],key
    else:
        assert 0<m['level']<=finest_level and audit['rebuilt_hierarchy_verified']
        assert audit['velocity_transfer_audit']['all_injected_velocities_verified']
        lower=json.loads((Path(m['lower_run'])/'run.json').read_text())
        assert lower['level']==m['level']-1 and lower['completed']
        assert m['robot_q_sha256']==lower['robot_q_sha256'] and m['trajectory_sha256']==lower['trajectory_sha256']
    assert hashlib.sha256(prefix.read_bytes()).hexdigest()==plan['prefix_sha256']
    with np.load(prefix) as before,np.load(plan['trajectory']) as after:
        for key in ['joint_q','openness']:np.testing.assert_array_equal(before[key],after[key][:plan['exact_prefix_frames']])
        assert np.all(after['openness'][plan['stages_control_frames']['closed']:plan['stages_control_frames']['pre_release_hold_end']+1]<1e-10)
        assert np.all(after['openness'][plan['stages_control_frames']['opened']:]>.999)
    x=d['x'];f=d['f'];poses=d['robot'];end=m['completed_steps'];close=s['closed'];release=s['pre_release_hold_end'];hands={}
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for side,j in [('left',1),('right',5)]:
        ks=np.arange(close,end+1);w=poses[ks,j];tip=w[0,:3,:3]@np.array([.150,.001786,0])+w[0,:3,3]
        cp=closest_point(x[close][f],np.broadcast_to(tip,(len(f),3)));dist=np.linalg.norm(cp-tip,axis=1);face=int(dist.argmin())
        bary=points_to_barycentric(x[close][f[face]][None],cp[face][None])[0];point=np.einsum('j,njk->nk',bary,x[:,f[face]][ks])
        local=np.einsum('nij,nj->ni',np.linalg.inv(w[:,:3,:3]),point-w[:,:3,3]);drift=np.linalg.norm(local-local[0],axis=1)
        phases=[]
        for name,begin,finish in [('pickup',close,s['transfer_hold_end']),('fold_arc',s['transfer_hold_end'],s['fold_arc_end']),('placement',s['fold_arc_end'],release)]:
            i=begin-close;k=finish-close;phases.append(dict(phase=name,first_step=begin,last_step=finish,
                maximum_drift_from_phase_start_mm=float(np.linalg.norm(local[i:k+1]-local[i],axis=1).max()*1000),
                material_start_m=point[i].tolist(),material_end_m=point[k].tolist()))
        hands[side]=dict(face=face,vertex_ids=f[face].tolist(),barycentric=bary.tolist(),
            maximum_closed_interval_drift_mm=float(drift[:release-close+1].max()*1000),
            material_at_release_start_m=point[release-close].tolist(),material_final_m=point[-1].tolist(),phases=phases)
        axes[0].plot(ks*m['dt'],(point[:,2]-.8)*1000,label=side)
        axes[1].plot(ks[:release-close+1]*m['dt'],drift[:release-close+1]*1000,label=side)
    for ax in axes:
        for name in ['transfer_hold_end','fold_arc_end','placement_end','pre_release_hold_end','opened','retreat_end']:ax.axvline(s[name]*m['dt'],color='gray',alpha=.3)
        ax.legend();ax.grid(alpha=.2)
    axes[0].set(ylabel='Tracked material height above table (mm)');axes[1].set(ylabel='Closed-grip wrist-relative drift (mm)',xlabel='Physical time (s)')
    fig.savefig(out/'fold_grasp_tracking.png',dpi=140);plt.close(fig)
    stages=[('Settled',60),('Before fold',s['transfer_hold_end']),('Arc midpoint',(s['transfer_hold_end']+s['fold_arc_end'])//2),('Fold arc end',s['fold_arc_end']),('Placed',s['placement_end']),('Opened',s['opened']),('Retreated',s['retreat_end']),('Final',end)]
    base_area=ConvexHull(x[60,:,:2]).volume;stage_metrics=[];phases=[];start=1
    for name,k in stages:
        stage_metrics.append(dict(stage=name,step=k,footprint_convex_hull_ratio=float(ConvexHull(x[k,:,:2]).volume/base_area),
            height_percentiles_mm=(np.percentile(x[k,:,2]-.8,[0,50,90,100])*1000).tolist()))
        rr=rows[start-1:k];phases.append(dict(stage=name,first_step=start,last_step=k,native=sum(r['native']['converged'] for r in rr),preview=sum(r['native']['preview_non_descent'] for r in rr),newton_limits=sum(r['native']['hit_newton_limit'] for r in rr)));start=k+1
    velocity=np.diff(x[-61:],axis=0)/m['dt']
    edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    collision_mesh=ipctk.CollisionMesh(np.asarray(x[0]),edges,f.astype(np.int32))
    intersection_samples=[dict(stage=name,step=k,self_intersection=bool(ipctk.has_intersections(collision_mesh,np.asarray(x[k])))) for name,k in stages]
    result=dict(completed_steps=end,physical_seconds=end*m['dt'],physics_seconds=m['physics_seconds'],
        level=m['level'],unchanged_mesh_material_contact_solver_runtime=m['level']==0 and not audit.get('revised_finest_verified',False),
        revised_finest_verified=audit.get('revised_finest_verified',False),
        unchanged_material_contact_solver_runtime=budget is None,line_search_budget_override=budget,progressive_velocity_transfer_verified=m['level']>0,exact_prefix_verified=True,
        native_converged_steps=m['native_converged_steps'],preview_steps=m['preview_stopped_steps'],newton_limit_steps=m['newton_limit_steps'],
        all_preview_exits_verified=audit['preview_audit']['all_preview_evidence_verified'] if audit['preview_audit'] else m['preview_stopped_steps']==0,maximum_principal_stretch=audit['maximum_principal_stretch'],
        maximum_robot_surface_target_error_mm=audit['maximum_body_surface_tracking_error_m']*1000,hands=hands,stages=stage_metrics,phases=phases,
        final_half_second_rms_speed_mm_s=float(np.sqrt(np.mean(np.sum(velocity**2,axis=-1)))*1000),
        final_half_second_rms_displacement_mm=float(np.sqrt(np.mean(np.sum((x[-1]-x[-61])**2,axis=-1)))*1000),
        sampled_self_intersections=intersection_samples,fold_quality_passed=None,limitations=['Motion completion and footprint reduction do not establish a correctly aligned layered fold.',
            'Fixed material-point tracking is geometric evidence, not a contact-force proof. No holding requirement after opening starts.',
            'Identical prefix targets do not guarantee identical repeat-run cloth trajectories.',
            'No independent continuous intersection audit. Preview exits are not Newton convergence.'])
    table_audit=out/'table_extent_audit.json'
    if table_audit.exists():
        result['table_extent_audit']=json.loads(table_audit.read_text())
        if result['table_extent_audit']['final_below_top_vertex_count']:
            result['fold_quality_passed']=False
            result['limitations'].append('Cloth hangs below the finite tabletop; lower height is not evidence of a flatter fold.')
    (out/'custom_fold_audit.json').write_text(json.dumps(result,indent=2))
    fig=plt.figure(figsize=(16,8))
    for i,(name,k) in enumerate(stages):draw(fig.add_subplot(2,4,i+1,projection='3d'),d,k,f'{name} | {k*m["dt"]:.2f} s',True)
    fig.tight_layout();fig.savefig(out/'fold_stages.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(10,8))
    for i,(label,elev,azim) in enumerate([('Overview',48,-125),('Overhead',90,-90),('Side',10,-90),('Opposite',25,55)]):
        ax=fig.add_subplot(2,2,i+1,projection='3d');draw(ax,d,end,label,True);ax.view_init(elev=elev,azim=azim)
        if elev==90:ax.set_proj_type('ortho');ax.set_zticks([]);ax.set_zlabel('')
    fig.tight_layout();fig.savefig(out/'fold_final_views.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(11,5));material=x[0,f,0].mean(1);norm=plt.Normalize(float(material.min()),float(material.max()))
    for i,(label,k) in enumerate([('Settled',60),('Final',end)]):
        ax=fig.add_subplot(1,2,i+1,projection='3d');draw(ax,d,k,label,False);ax.collections[0].set_facecolor(plt.cm.viridis(norm(material)));ax.view_init(elev=90,azim=-90);ax.set_proj_type('ortho');ax.set_zticks([]);ax.set_zlabel('')
    fig.suptitle('Fixed original material colors: settled and after release');fig.tight_layout();fig.savefig(out/'fold_material_tracking.png',dpi=140);plt.close(fig)
    images=[]
    for k in ([] if a.skip_animation else np.unique(np.linspace(0,end,160).astype(int))):
        fig=plt.figure(figsize=(10,4.5),dpi=85)
        for i in range(2):
            ax=fig.add_subplot(1,2,i+1,projection='3d');draw(ax,d,k,f'L{m["level"]} modified first fold | {k*m["dt"]:.2f} s',True)
            if i:ax.view_init(elev=90,azim=-90);ax.set_proj_type('ortho');ax.set_zticks([]);ax.set_zlabel('')
        fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    if images:images[0].save(out/'full_fold.gif',save_all=True,append_images=images[1:],duration=round(end*m['dt']*100/len(images))*10,loop=0)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
