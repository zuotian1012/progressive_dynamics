"""Compare completed shirt pilots differing only in semi-implicit termination."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import plt

def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [d for k in sorted(set(a)|set(b)) for d in differences(a.get(k),b.get(k),f'{path}.{k}' if path else k)]
    return [] if a==b else [dict(path=path,on=a,off=b)]

def load(path):
    meta=json.loads((path/'run.json').read_text());assert meta['completed'] and meta['completed_steps']==786
    rows=json.loads((path/'steps.json').read_text())
    audit=json.loads((path/'report/audit.json').read_text())
    grip=json.loads((path/'report/grasp_diagnostics.json').read_text())
    times=np.load(path/'timing.npz')['step_seconds']
    phases={}
    for name,start,end in [('settle',1,60),('approach_and_close',61,498),('lift',499,786)]:
        phase=rows[start-1:end]
        phases[name]=dict(start=start,end=end,physics_seconds=float(times[start-1:end].sum()),
            newton_iterations=sum(r['native']['newton_iterations'] for r in phase),
            newton_limit_steps=sum(r['native']['hit_newton_limit'] for r in phase),
            line_search_limit_steps=sum(r['native']['hit_line_search_limit'] for r in phase))
    summary=dict(physics_seconds=meta['physics_seconds'],newton_iterations=sum(r['native']['newton_iterations'] for r in rows),
        newton_limit_steps=meta['newton_limit_steps'],line_search_limit_steps=meta['line_search_limit_steps'],
        native_converged_steps=meta['native_converged_steps'],
        inferred_semi_implicit_stops=len(audit['semi_implicit_stop_frames_inferred']),
        default_tolerance_stops=len(audit['default_tolerance_stop_frames']),other_stops=len(audit['other_stop_frames']),
        maximum_principal_stretch=audit['maximum_principal_stretch'],
        maximum_body_surface_tracking_error_m=audit['maximum_body_surface_tracking_error_m'],
        grip_region_tracking=audit['grip_region_tracking'],grip_diagnostics=grip,
        screening=audit['screening'],screening_passed=audit['screening_passed'],phases=phases)
    return meta,rows,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--on',type=Path,required=True);p.add_argument('--off',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    on,ron,son=load(a.on);off,roff,soff=load(a.off)
    diff=differences(on['config'],off['config']);assert diff==[dict(path='newton.semi_implicit.enable',on=1,off=0)]
    for key in ['runtime_hashes','mesh_sha256','robot_q_sha256','psq_bundle_sha256','material_parameters','strain_step_cap','steps','dt']:
        assert on[key]==off[key],key
    assert soff['inferred_semi_implicit_stops']==0
    result=dict(config_differences=diff,on=son,off=soff,physics_time_ratio_off_over_on=soff['physics_seconds']/son['physics_seconds'],
        caveats=['One trajectory per setting; full histories diverge, so this is not a paired identical-contact-state experiment.',
                 'Geometric grasp metrics are not contact-force evidence; no independent collision penetration audit.',
                 'Disabling semi-implicit termination retains the default displacement tolerance and non-strict failure policy.'])
    matched={}
    for hand,j in [('left',1),('right',5)]:
        groups={name:summary['grip_region_tracking'][hand]['tracked_vertex_ids'] for name,summary in [('on_closure_nearest8',son),('off_closure_nearest8',soff)]}
        groups['common_vertices']=sorted(set(groups['on_closure_nearest8'])&set(groups['off_closure_nearest8']))
        matched[hand]={}
        for name,ids in groups.items():
            row=dict(vertex_ids=ids)
            for label,path in [('on',a.on),('off',a.off)]:
                x=np.load(path/'positions.npy');w=np.load(path/'robot.npy')[498:,j];point=x[498:,ids].mean(1)
                local=np.einsum('nij,nj->ni',np.linalg.inv(w[:,:3,:3]),point-w[:,:3,3])
                row[label]=dict(max_drift_m=float(np.linalg.norm(local-local[0],axis=1).max()),final_lift_m=float(point[-1,2]-point[0,2]))
            matched[hand][name]=row
    result['matched_material_tracking']=matched
    a.output.mkdir(exist_ok=True,parents=True);(a.output/'comparison.json').write_text(json.dumps(result,indent=2))
    (a.output/'matched_material_tracking.json').write_text(json.dumps(matched,indent=2))
    fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained')
    for path,label,rows,summary in [(a.on,'Semi-implicit ON',ron,son),(a.off,'Semi-implicit OFF',roff,soff)]:
        t=np.arange(1,787)*on['dt']
        axes[0].plot(t,[r['native']['newton_iterations'] for r in rows],label=label,alpha=.8)
        axes[1].plot(t,np.load(path/'report/body_surface_tracking_error.npy').max(1)*1000,label=label)
        x=np.load(path/'positions.npy');poses=np.load(path/'robot.npy');ks=np.arange(498,787)
        for hand,j in [('left',1),('right',5)]:
            ids=summary['grip_region_tracking'][hand]['tracked_vertex_ids'];w=poses[ks,j]
            local=np.einsum('nij,nj->ni',np.linalg.inv(w[:,:3,:3]),x[ks][:,ids].mean(1)-w[:,:3,3])
            drift=np.linalg.norm(local-local[0],axis=1)
            axes[2].plot(ks*on['dt'],drift*1000,label=f'{label}, {hand}',ls='-' if hand=='left' else '--')
    axes[0].set(ylabel='Newton iterations',yscale='log');axes[1].set(ylabel='Max collider target error (mm)')
    axes[2].set(xlabel='Physical time (s)',ylabel='Wrist drift: closure nearest 8 per run (mm)')
    axes[1].axhline(10,color='gray',ls=':');axes[2].axhline(20,color='gray',ls=':')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(a.output/'comparison.png',dpi=140);plt.close(fig)
    print(json.dumps({key:{k:v for k,v in row.items() if k not in ['phases','grip_diagnostics','grip_region_tracking']} for key,row in [('on',son),('off',soff)]},indent=2))

if __name__=='__main__':main()
