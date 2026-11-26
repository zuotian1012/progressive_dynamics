"""Measure an explicit grasp/lift/hold pilot using fixed material points."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from trimesh.triangles import closest_point, points_to_barycentric
from shirt_demo.report import load_run, draw, plt, Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot',type=Path)
    p.add_argument('--baseline',type=Path,default=Path('output/shirt_L0_root_cause_20261116/full_fold_L0'))
    p.add_argument('--skip-animation',action='store_true')
    a=p.parse_args();root=a.pilot/'run';out=root/'report';out.mkdir(exist_ok=True)
    plan=json.loads((a.pilot/'plan.json').read_text());d=load_run(root);m=d['meta']
    assert m['completed']
    audit=json.loads((out/'audit.json').read_text())
    old=json.loads((a.baseline/'run.json').read_text())
    for key in ['mesh_sha256','psq_bundle_sha256','material_parameters','config','runtime_hashes','robot_drive_strength']:
        assert m[key]==old[key],key
    x=d['x'];poses=d['robot'];f=d['f'];s=plan['stages_physical_steps'];close=s['closed'];lift=s['lift_end'];end=s['hold_end']
    assert end==m['completed_steps']
    settling_error=float(np.max(np.abs(x[:61]-np.load(a.baseline/'positions.npy',mmap_mode='r')[:61])))
    baseline_positions=np.load(a.baseline/'positions.npy',mmap_mode='r')
    np.testing.assert_array_equal(x[0],baseline_positions[0])
    np.testing.assert_array_equal(poses[0],np.load(a.baseline/'robot.npy',mmap_mode='r')[0])
    transfer='transfer_end' in s
    if transfer:
        with np.load(plan['prefix_trajectory']) as prefix,np.load(plan['trajectory']) as current:
            np.testing.assert_array_equal(prefix['joint_q'],current['joint_q'][:plan['exact_prefix_frames']])
            np.testing.assert_array_equal(prefix['openness'],current['openness'][:plan['exact_prefix_frames']])
        assert hashlib.sha256(Path(plan['prefix_trajectory']).read_bytes()).hexdigest()==plan['prefix_sha256']
    ks=np.arange(close,end+1);hands={};fig,ax=plt.subplots(3 if transfer else 2,1,figsize=(10,9 if transfer else 7),layout='constrained')
    for side,j in [('left',1),('right',5)]:
        w=poses[ks,j];target=w[0,:3,:3]@np.array([.150,.001786,0])+w[0,:3,3]
        candidates=closest_point(x[close][f],np.broadcast_to(target,(len(f),3)))
        dist=np.linalg.norm(candidates-target,axis=1);face=int(dist.argmin())
        bary=points_to_barycentric(x[close][f[face]][None],candidates[face][None])[0]
        point=np.einsum('j,njk->nk',bary,x[ks][:,f[face]])
        local=np.einsum('nij,nj->ni',np.linalg.inv(w[:,:3,:3]),point-w[:,:3,3])
        drift=np.linalg.norm(local-local[0],axis=1)
        dz=point[:,2]-point[0,2]
        jaw=np.einsum('nij,j->ni',w[:,:3,:3],np.array([.150,.001786,0]))+w[:,:3,3]
        jaw_lift=jaw[:,2]-jaw[0,2]
        hold=s.get('transfer_end',lift)-close
        info=dict(face=face,vertex_ids=f[face].tolist(),barycentric=bary.tolist(),
            initial_pinch_center_to_surface_mm=float(dist[face]*1000),
            material_lift_at_lift_end_mm=float(dz[lift-close]*1000),material_lift_at_hold_end_mm=float(dz[-1]*1000),
            gripper_lift_at_hold_end_mm=float(jaw_lift[-1]*1000),
            maximum_wrist_relative_drift_mm=float(drift.max()*1000),
            hold_height_drop_mm=float((dz[hold]-dz[-1])*1000),
            maximum_hold_drift_mm=float(np.linalg.norm(local[hold:]-local[hold],axis=1).max()*1000))
        info['geometric_hold_passed']=bool(dz[-1]>=.045 and drift.max()<=.01 and abs(dz[hold]-dz[-1])<=.003)
        if transfer:
            start=s['lift_hold_end']-close
            point_motion=point[start:]-point[start];jaw_motion=jaw[start:]-jaw[start]
            transfer_drift=np.linalg.norm(local[start:]-local[start],axis=1)
            hold_delta=point[hold:]-point[hold]
            info['transfer']=dict(first_step=s['lift_hold_end'],motion_end_step=s['transfer_end'],hold_end_step=end,
                material_displacement_at_motion_end_mm=((point[hold]-point[start])*1000).tolist(),
                material_displacement_at_hold_end_mm=(point_motion[-1]*1000).tolist(),
                gripper_displacement_at_hold_end_mm=(jaw_motion[-1]*1000).tolist(),
                maximum_wrist_relative_drift_mm=float(transfer_drift.max()*1000),
                maximum_motion_follow_error_mm=float(np.linalg.norm(point_motion-jaw_motion,axis=1).max()*1000),
                maximum_final_hold_displacement_mm=float(np.linalg.norm(hold_delta,axis=1).max()*1000))
            info['transfer']['geometric_transfer_passed']=bool(transfer_drift.max()<=.01 and
                np.linalg.norm(point_motion[-1]-jaw_motion[-1])<=.01 and np.linalg.norm(hold_delta,axis=1).max()<=.003)
            ax[2].plot(ks[start:]*m['dt'],point_motion[:,0]*1000,label=f'{side} material')
            ax[2].plot(ks[start:]*m['dt'],jaw_motion[:,0]*1000,'--',label=f'{side} jaw')
        hands[side]=info
        ax[0].plot(ks*m['dt'],dz*1000,label=f'{side} material');ax[0].plot(ks*m['dt'],jaw_lift*1000,'--',label=f'{side} jaw')
        ax[1].plot(ks*m['dt'],drift*1000,label=side)
    for axis in ax:
        axis.axvline(lift*m['dt'],color='gray',ls=':');axis.legend();axis.grid(alpha=.25)
    ax[0].set(ylabel='Lift from closure (mm)');ax[1].set(ylabel='Wrist-relative drift (mm)',xlabel='Physical time (s)')
    if transfer:
        ax[2].set(ylabel='Transfer X displacement (mm)',xlabel='Physical time (s)')
        for axis in ax:
            for key in ['lift_hold_end','transfer_end']:axis.axvline(s[key]*m['dt'],color='gray',ls=':')
    fig.savefig(out/'grasp_tracking.png',dpi=140);plt.close(fig)
    result=dict(completed_steps=m['completed_steps'],physics_seconds=m['physics_seconds'],
        unchanged_mesh_material_contact_solver_runtime=True,settling_max_difference_m=settling_error,
        settling_end_max_difference_m=float(np.max(np.abs(x[60]-baseline_positions[60]))),
        native_converged_steps=m['native_converged_steps'],preview_steps=m['preview_stopped_steps'],newton_limit_steps=m['newton_limit_steps'],
        all_preview_evidence_verified=audit['preview_audit']['all_preview_evidence_verified'],
        maximum_principal_stretch=audit['maximum_principal_stretch'],peak_robot_surface_error_mm=audit['maximum_body_surface_tracking_error_m']*1000,
        stages=s,hands=hands,both_geometric_holds_passed=all(h['geometric_hold_passed'] for h in hands.values()),
        limitations=['Fixed barycentric material-point tracking is a geometric grasp test, not a contact-force proof.',
            'Identical settling inputs do not imply bitwise repeatable trajectories; measured repeat-run differences are reported.',
            'Criteria: final material lift >=45 mm, maximum wrist-relative drift <=10 mm, hold height change <=3 mm.',
            'This pilot ends with jaws closed; release and a complete fold are not validated.'])
    if transfer:
        result.update(exact_original_trajectory_prefix_verified=True,
            both_geometric_transfers_passed=all(h['transfer']['geometric_transfer_passed'] for h in hands.values()))
        result['limitations'].append('Transfer criteria: <=10 mm wrist-relative drift/final following error; <=3 mm final hold movement.')
    (out/'grasp_pilot_audit.json').write_text(json.dumps(result,indent=2))
    stages=[('Settled',60),('At grasp',s['grasp_position']),('Closed',close),('Close hold',s['close_hold_end']),('Lifted',lift),('Hold end',end)]
    if transfer:stages=[('Closed',close),('Lifted',lift),('Before transfer',s['lift_hold_end']),('Transfer midpoint',(s['lift_hold_end']+s['transfer_end'])//2),('Transfer end',s['transfer_end']),('Hold end',end)]
    fig=plt.figure(figsize=(13,8))
    for i,(label,k) in enumerate(stages):draw(fig.add_subplot(2,3,i+1,projection='3d'),d,k,f'{label} | step {k}',True)
    fig.tight_layout();fig.savefig(out/'grasp_stages.png',dpi=140);plt.close(fig)
    images=[]
    for k in ([] if a.skip_animation else np.unique(np.linspace(0,end,100).astype(int))):
        fig=plt.figure(figsize=(10,4.5),dpi=85)
        for i in range(2):
            axis=fig.add_subplot(1,2,i+1,projection='3d');draw(axis,d,k,f'L0 {"transfer" if transfer else "grasp"} pilot | {k*m["dt"]:.2f} s',True)
            if i:axis.view_init(elev=90,azim=-90);axis.set_proj_type('ortho');axis.set_zticks([]);axis.set_zlabel('')
        fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    if images:images[0].save(out/'grasp_pilot.gif',save_all=True,append_images=images[1:],duration=round(end*m['dt']*100/len(images))*10,loop=0)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
