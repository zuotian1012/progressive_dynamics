"""Read-only settling comparison; geometric differences are not an artifact oracle."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz
from progressive.prolongation import ShellProlongation,frames
from shirt_demo.report import load_run,draw,plt

root=Path('output/shirt_settling_artifacts_20261116');root.mkdir(exist_ok=True)
paths=[Path('output/shirt_pd_first_fold_20261116/L0'),
       Path('output/shirt_L1_directional_boundary_20261116/validation'),
       Path('output/shirt_pd_L1_to_L2_20261116/L2_ls32')]
data=[load_run(p) for p in paths]
rows=[json.loads((p/'steps.json').read_text()) for p in paths]
assets=Path(data[2]['meta']['parameters']['assets']);transfers=[];maps=[]
for level in (1,2):
    U=load_npz(assets/f'map_{level-1}_{level}.npz')
    P=ShellProlongation(data[level-1]['x'][0],data[level]['x'][0],data[level-1]['f'],U,
                        np.load(assets/f'anchors_{level-1}_{level}.npy'))
    transfers.append(P)
    maps.append(dict(level=level,weight_min=float(U.data.min()),weight_max=float(U.data.max()),
        max_translation_preservation_error=float(np.abs(np.asarray(U.sum(axis=1)).ravel()-1).max()),
        rest_psq_error_m=float(np.abs(P.position(data[level-1]['x'][0])-data[level]['x'][0]).max())))

samples=[]
for k in (0,1,5,10,15,22,30,60):
    entry=dict(step=k,time_s=k/120,levels=[],cross_level=[])
    for level,d in enumerate(data):
        inv=frames(d['x'][0],d['f'])[2];tri=d['x'][k][d['f']]
        F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)@inv
        sigma=np.linalg.svd(F,compute_uv=False)
        entry['levels'].append(dict(level=level,
            z_percentiles_m=np.percentile(d['x'][k,:,2],[0,5,50,95,100]).tolist(),
            maximum_stretch=float(sigma[:,0].max()),
            triangles_stretch_above_1_099=int(np.count_nonzero(sigma[:,0]>1.099)),
            fraction_stretch_above_1_099=float(np.mean(sigma[:,0]>1.099)),
            first_preview_step=next((r['step'] for r in rows[level] if r['native']['preview_non_descent']),None),
            native_converged_steps_to_here=sum(r['native']['converged'] for r in rows[level][:k])))
    for level,P in enumerate(transfers,1):
        actual=data[level]['x'][k];mapped=P.position(data[level-1]['x'][k]);delta=actual-mapped
        ac=actual-actual.mean(axis=0);mc=mapped-mapped.mean(axis=0)
        u,s,vt=np.linalg.svd(mc.T@ac);fix=np.eye(3);fix[2,2]=np.linalg.det(u@vt)
        rotation=u@fix@vt;shape=ac-mc@rotation
        entry['cross_level'].append(dict(level=level,
            rms_difference_mm=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))*1000),
            mean_difference_mm=(delta.mean(axis=0)*1000).tolist(),
            rms_after_rigid_alignment_mm=float(np.sqrt(np.mean(np.sum(shape**2,axis=1)))*1000)))
    samples.append(entry)
result=dict(run_paths=[str(p.resolve()) for p in paths],maps=maps,samples=samples,
    scope='Saved physical states only. No new solve, target, material, solver or integrator changes.',
    limitations=['Cross-level differences are not by themselves evidence of a bug or an artifact.',
      'Rigid-aligned RMS removes bulk translation/rotation, but still includes legitimate resolution differences.',
      'Wire edges are disabled on every surface; face-normal shading still depends on mesh resolution.',
      'No causal isolation of velocity transfer, shared energy, contact or preview termination; no map-boundary correlation claim.'])
(root/'diagnosis.json').write_text(json.dumps(result,indent=2))
fig=plt.figure(figsize=(15,12))
for row,k in enumerate((0,10,60)):
    for level,d in enumerate(data):
        ax=fig.add_subplot(3,3,row*3+level+1,projection='3d')
        draw(ax,d,k,f'L{level} | step {k} | {k/120:.3f} s',False)
        ax.set(xlim=(.28,1.12),ylim=(-.43,.43),zlim=(.78,1.18))
        ax.set_box_aspect((.84,.86,.4))
fig.suptitle('Actual settling states | identical camera and shading, no mesh-edge overlay',fontsize=14)
fig.tight_layout(rect=(0,0,1,.965))
fig.savefig(root/'settling_comparison.png',dpi=140,bbox_inches='tight');plt.close(fig)
fig=plt.figure(figsize=(15,5))
for level,d in enumerate(data):
    draw(fig.add_subplot(1,3,level+1,projection='3d'),d,60,f'L{level} | settling step 60 | 0.500 s',False)
fig.suptitle('Actual settling states, before grasp | same camera, shading and axis limits')
fig.tight_layout();fig.savefig(root/'settling_end.png',dpi=140,bbox_inches='tight');plt.close(fig)
print(json.dumps(result,indent=2))
