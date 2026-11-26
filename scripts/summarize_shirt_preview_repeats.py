"""Summarize independent L0 preview repeats without treating preview as convergence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import plt

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
paths=[root/f'repeat_{i}' for i in (1,2,3)];rows=[];metas=[]
for path in paths:
    m=json.loads((path/'run.json').read_text());metas.append(m);assert m['completed'] and m['completed_steps']==786
    audit=json.loads((path/'report/audit.json').read_text());preview=audit['preview_audit'];assert preview['all_preview_evidence_verified']
    records=json.loads((path/'steps.json').read_text());phases={}
    for name,start,end in [('settle',1,60),('approach_close',61,498),('lift',499,786)]:
        part=records[start-1:end]
        phases[name]=dict(preview=sum(r['native']['preview_non_descent'] for r in part),converged=sum(r['native']['converged'] for r in part),
            failed=sum(not(r['native']['converged'] or r['native']['preview_non_descent']) for r in part))
    rows.append(dict(run=path.name,physics_seconds=m['physics_seconds'],native_converged_steps=m['native_converged_steps'],
        preview_stopped_steps=m['preview_stopped_steps'],preview_at_first_iteration=len(preview['preview_at_first_newton_frames']),
        newton_limit_steps=m['newton_limit_steps'],non_preview_line_search_limit_steps=len(preview['non_preview_line_search_limit_frames']),
        maximum_stretch=audit['maximum_principal_stretch'],minimum_sigma=audit['minimum_singular_value'],
        peak_surface_error_mm=1000*audit['maximum_body_surface_tracking_error_m'],
        left=audit['grip_region_tracking']['left'],right=audit['grip_region_tracking']['right'],
        screening=audit['screening'],screening_passed=audit['screening_passed'],phases=phases))
for m in metas[1:]:
    assert m['parameters'].get('robot_drive_scale',1.)==metas[0]['parameters'].get('robot_drive_scale',1.)
    for k in ['config','runtime_hashes','material_parameters','mesh_sha256','robot_q_sha256','psq_bundle_sha256','strain_step_cap']:
        assert m[k]==metas[0][k],k
old=json.loads(Path('output/shirt_semi_implicit_control_20261116/semi_off_grasp/run.json').read_text())
for k in ['config','material_parameters','mesh_sha256','robot_q_sha256','psq_bundle_sha256','strain_step_cap']:
    assert old[k]==metas[0][k],k
reference=np.load(paths[0]/'positions.npy',mmap_mode='r');differences=[]
for path in paths[1:]:
    other=np.load(path/'positions.npy',mmap_mode='r');diff=np.linalg.norm(other-reference,axis=2)
    differences.append(dict(run=path.name,reference=paths[0].name,maximum_vertex_difference_m=float(diff.max()),final_rms_m=float(np.sqrt((diff[-1]**2).mean()))))
drive_scale=metas[0]['parameters'].get('robot_drive_scale',1.)
result=dict(runs=rows,same_build_and_inputs=True,robot_drive_scale=drive_scale,solver_and_physical_inputs_match_previous_semi_off=drive_scale==1.,solver_and_cloth_inputs_match_previous_semi_off=True,
    all_repeats_passed=all(r['screening_passed'] for r in rows),passed_repeats=sum(r['screening_passed'] for r in rows),
    repeats=3,trajectory_differences=differences,
    caveats=['Preview exits retain feasible iterates but are not Newton convergence.',
             'Three repetitions assess observed consistency, not statistical reliability.',
             'Geometric screening is local; no independent collision penetration audit or proof of correct contact forces.'])
(root/'repeat_summary.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained');labels=[r['run'] for r in rows];ids=np.arange(3)
ax[0,0].bar(ids,[r['native_converged_steps'] for r in rows],label='Tolerance convergence');ax[0,0].bar(ids,[r['preview_stopped_steps'] for r in rows],bottom=[r['native_converged_steps'] for r in rows],label='Non-descent preview');ax[0,0].legend(fontsize=8);ax[0,0].set(ylabel='Physical steps')
failed=[786-r['native_converged_steps']-r['preview_stopped_steps'] for r in rows]
if any(failed):
    ax[0,0].bar(ids,failed,bottom=[r['native_converged_steps']+r['preview_stopped_steps'] for r in rows],color='red',label='Unclassified failure');ax[0,0].legend(fontsize=8)
ax[0,1].bar(ids,[r['peak_surface_error_mm'] for r in rows]);ax[0,1].axhline(10,color='r',ls='--');ax[0,1].set(ylabel='Peak collider target error (mm)')
for hand,shift in [('left',-.18),('right',.18)]:
    ax[1,0].bar(ids+shift,[1000*r[hand]['max_wrist_relative_drift_m'] for r in rows],.36,label=hand)
    ax[1,1].bar(ids+shift,[1000*r[hand]['peak_mean_region_lift_m'] for r in rows],.36,label=hand)
ax[1,0].axhline(20,color='r',ls='--');ax[1,0].set(ylabel='Region maximum wrist drift (mm)');ax[1,0].legend()
ax[1,1].axhline(20,color='r',ls='--');ax[1,1].set(ylabel='Region peak lift (mm)')
for axis in ax.flat:axis.set_xticks(ids,labels)
fig.savefig(root/'repeat_summary.png',dpi=140);plt.close(fig)
print(json.dumps(result,indent=2))
