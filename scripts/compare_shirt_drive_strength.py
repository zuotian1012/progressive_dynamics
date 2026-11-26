"""Compare three preview repetitions for baseline and doubled soft targets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import plt

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
groups=[json.loads((path/'repeat_summary.json').read_text()) for path in [root,root/'drive2']]
assert [g['robot_drive_scale'] for g in groups]==[1,2]
baseline=json.loads((root/'repeat_1/run.json').read_text())
for folder in [root,root/'drive2']:
    for i in (1,2,3):
        m=json.loads((folder/f'repeat_{i}/run.json').read_text())
        for key in ['config','runtime_hashes','material_parameters','mesh_sha256','robot_q_sha256','psq_bundle_sha256','strain_step_cap','steps','dt','coarse_preview']:
            assert m[key]==baseline[key],key
def interval(values):return dict(min=min(values),max=max(values),mean=float(np.mean(values)))
rows=[]
for g in groups:
    runs=g['runs']
    rows.append(dict(drive_scale=g['robot_drive_scale'],passed_repeats=g['passed_repeats'],repeats=3,
        peak_surface_error_mm=interval([r['peak_surface_error_mm'] for r in runs]),
        left_drift_mm=interval([r['left']['max_wrist_relative_drift_m']*1000 for r in runs]),
        right_drift_mm=interval([r['right']['max_wrist_relative_drift_m']*1000 for r in runs]),
        physics_seconds=interval([r['physics_seconds'] for r in runs]),
        newton_limit_steps_total=sum(r['newton_limit_steps'] for r in runs),
        preview_stopped_steps=[r['preview_stopped_steps'] for r in runs],
        non_preview_line_search_limit_steps=[r['non_preview_line_search_limit_steps'] for r in runs]))
result=dict(only_physical_change='SoftTransformConstraint strengths [300,1000] -> [600,2000]',
    same_native_build_solver_cloth_contact_and_trajectory=True,groups=rows,
    limitations=['Three independent runs per setting, not statistical reliability or a deterministic paired-contact replay.',
                 'Passing is the pre-existing local geometric and declared-stop screening; preview remains distinct from Newton convergence.'])
(root/'drive_comparison.json').write_text(json.dumps(result,indent=2))
fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
specs=[('peak_surface_error_mm','Peak collider target error (mm)',10),('left_drift_mm','Left region maximum drift (mm)',20),('right_drift_mm','Right region maximum drift (mm)',20),('physics_seconds','Physics time (s)',None)]
for ax,(key,label,limit) in zip(axes.flat,specs):
    for j,g in enumerate(groups):
        if key=='left_drift_mm':v=[r['left']['max_wrist_relative_drift_m']*1000 for r in g['runs']]
        elif key=='right_drift_mm':v=[r['right']['max_wrist_relative_drift_m']*1000 for r in g['runs']]
        else:v=[r[key] for r in g['runs']]
        ax.scatter(j+np.array([-.06,0,.06]),v,s=40);ax.plot([j-.13,j+.13],[np.mean(v)]*2,color='black',lw=2)
    if limit:ax.axhline(limit,color='red',ls='--',lw=1)
    ax.set(xticks=[0,1],xticklabels=['Strength 1x','Strength 2x'],ylabel=label,xlim=(-.4,1.4),ylim=(0,None))
fig.savefig(root/'drive_comparison.png',dpi=150);plt.close(fig);print(json.dumps(result,indent=2))
