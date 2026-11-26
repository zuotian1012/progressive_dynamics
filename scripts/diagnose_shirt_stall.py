"""Read-only diagnosis of saved relaxed shirt states and complete Newton logs.

The compression experiment checks only strain/rank feasibility, not energy or
contact, and is never written back into a simulated trajectory.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root=Path('output/shirt55k_L0_relaxed_grasp_20261116')
out=root/'diagnosis';out.mkdir(exist_ok=True)
meta=json.loads((root/'run.json').read_text());records=json.loads((root/'steps.json').read_text())
x=np.load(root/'positions.npy',mmap_mode='r')[:meta['completed_steps']+1];faces=np.load(root/'faces.npy')
tri=x[0][faces];edges=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)
inverse=np.linalg.inv(np.linalg.cholesky(edges.transpose(0,2,1)@edges).transpose(0,2,1))
areas=np.linalg.norm(np.cross(edges[:,:,0],edges[:,:,1]),axis=1)/2
lengths=np.stack([np.linalg.norm(tri[:,1]-tri[:,0],axis=1),np.linalg.norm(tri[:,2]-tri[:,1],axis=1),np.linalg.norm(tri[:,0]-tri[:,2],axis=1)],axis=1)
quality=4*np.sqrt(3)*areas/np.sum(lengths**2,axis=1)
heights=2*areas/lengths.max(axis=1)
lim=meta['material_parameters']['strain_limit']
log=root.with_suffix('.log').read_text(encoding='utf-8-sig')
parts=re.split(r'>>> Begin Frame: (\d+)',log)
frame_logs={int(parts[i]):parts[i+1] for i in range(1,len(parts),2)}
trajectory=[];example=[]
for frame,z in enumerate(x):
    t=z[faces];F=np.stack([t[:,1]-t[:,0],t[:,2]-t[:,0]],axis=-1)@inverse
    s=np.linalg.svd(F,compute_uv=False);face=int(s[:,0].argmax())
    trajectory.append(dict(frame=frame,max_stretch=float(s[face,0]),max_stretch_face=face,
                           min_singular_value=float(s[:,1].min()),strain_gap=float(lim-s[face,0])))
    if frame in [60,302,303,310,317,318,321]:
        # Uniform 1% contraction of this state: F(a)=(1-.01*a)F.
        # Every a in [0,1] decreases sigma_max and preserves positive rank.
        gap=np.minimum(lim-s[:,0],s[:,1]-1e-12)
        caps=np.minimum(1,.9*gap/np.linalg.norm(-.01*F,axis=(1,2)))
        binding=int(caps.argmin())
        example.append(dict(frame=frame,hypothetical_direction='uniform 1% contraction; not actual Newton direction',
                            weyl_cap=float(caps[binding]),binding_face=binding,
                            full_step_max_stretch=float(.99*s[:,0].max()),full_step_min_singular_value=float(.99*s[:,1].min()),
                            full_segment_strain_and_rank_feasible=bool(s[:,0].max()<lim and .99*s[:,1].min()>1e-12),
                            contact_or_energy_feasibility_tested=False))

rows=[]
for r in records:
    frame=r['step'];native=r['native'];text=frame_logs[frame]
    res=[tuple(map(float,v)) for v in re.findall(r'MaxTranslationChecker> Residual/AbsTol: ([^/]+)/([^\s]+)',text)]
    row=dict(frame=frame,converged=native['converged'],newton_iterations=native['newton_iterations'],
             line_search_trials=native['line_search_trials'],extra_backtracking_trials=native['line_search_trials']-native['newton_iterations'],
             last_alpha=native['last_line_search_alpha'],last_ccd_toi=native['last_ccd_toi'],last_cfl_alpha=native['last_cfl_alpha'],
             contact_step_caps=text.count('Feasible Step Filter:'),ccd_caps=text.count('CCD Filter:'),cfl_caps=text.count('CFL Filter:'),
             line_search_exhaustions=text.count('Line Search Exits with Max Iteration'),
             first_residual=res[0][0] if res else None,last_residual=res[-1][0] if res else None,
             absolute_tolerance=res[-1][1] if res else None,
             max_stretch=trajectory[frame]['max_stretch'],strain_gap=trajectory[frame]['strain_gap'])
    if 303<=frame<=317:
        assert row['newton_iterations']==row['line_search_trials']==1024
        assert not any(row[k] for k in ['contact_step_caps','ccd_caps','cfl_caps','line_search_exhaustions'])
        assert not row['converged']
    rows.append(row)
focus=136
diagnosis=dict(scope='Offline saved-state/log/source diagnosis; no replay, no solver or asset modification',
    target_frames=[303,317],target_frame_count=15,newton_iterations_in_target=15*1024,
    target_backtracking_trials=0,target_line_search_failures=0,
    limiting_mechanism='PSQ strain/rank feasible-step limiter inferred by eliminating all other shrink paths in source and logs',
    limiter_formula='min_faces min(1, 0.9*min(limit-sigma_max, sigma_min-1e-12)/||delta_F||_F)',
    actual_newton_directions_saved=False,actual_binding_triangle_proven=False,
    suspect_triangle=dict(face=focus,vertex_ids=faces[focus].tolist(),rest_edge_lengths_m=lengths[focus].tolist(),
                          rest_area_m2=float(areas[focus]),minimum_altitude_m=float(heights[focus]),
                          longest_edge_over_altitude=float(lengths[focus].max()/heights[focus]),
                          quality=float(quality[focus]),quality_definition='4*sqrt(3)*area/sum(edge_length_squared); equilateral=1',
                          quality_rank_from_worst=int(np.sum(quality<quality[focus]))+1,total_faces=len(faces),
                          initial_centroid_m=x[0][faces[focus]].mean(0).tolist()),
    constant_alpha_illustration=dict(alpha=.0001,updates=1018,beta=float((1-.0001)**1018),threshold=.001,
                                     explanation='Illustration only; beta not recorded in run. K_min=6 excludes first six iterations.'),
    compression_counterexamples=example,
    limitations=['Actual Newton directions/intermediate iterates were not saved, so exact binding triangle and tight directional step cannot be reconstructed.',
                 'No claim that a larger step decreases total energy or passes contact checks.',
                 'PSQ gradient/energy consistency and linear-system conditioning are not ruled out as upstream contributors.',
                 'Rank floor is 1e-12; saved states have sigma_min far above it, but intermediate-iterate rank margins were not saved.'],
    input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'run.json',root/'steps.json',root/'positions.npy',root.with_suffix('.log'),Path('native/libuipc/psq_proxy.cu'),Path('external/libuipc-psq-src/src/backends/cuda/engine/advance_ipc.cu')]})
(out/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2));(out/'frame_metrics.json').write_text(json.dumps(rows,indent=2));(out/'strain_metrics.json').write_text(json.dumps(trajectory,indent=2))
fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
ids=np.array([r['frame'] for r in rows]);mask=(ids>=290)&(ids<=321);selected=[r for r in rows if 290<=r['frame']<=321]
ax[0].semilogy(ids[mask],[r['last_alpha'] for r in selected],'.-');ax[0].set_ylabel('Last step fraction')
ax[1].semilogy(ids[mask],[r['last_residual']/r['absolute_tolerance'] for r in selected],'.-');ax[1].axhline(1,color='r',ls='--');ax[1].set_ylabel('Last residual / tolerance')
ax[2].plot(ids[mask],[r['newton_iterations'] for r in selected],'.-');ax[2].set(ylabel='Newton iterations',xlabel='Physical step')
for a in ax:a.axvspan(303,317,color='orange',alpha=.15);a.grid(alpha=.25)
fig.suptitle('303-317: no backtracking failures; tiny PSQ feasible steps');fig.savefig(out/'stall_metrics.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for i,(frame,title) in enumerate([(0,'Rest mesh, XY projection'),(302,'Frame 302, XY close-up')]):
    ax=axes[i];z=x[frame];t=z[faces[focus]];closed=np.vstack([t,t[0]])
    ax.triplot(z[:,0],z[:,1],faces,color='#7195a5',linewidth=.4)
    ax.plot(closed[:,0],closed[:,1],color='red',linewidth=2,zorder=3);ax.scatter(t[:,0],t[:,1],c='red',s=22,zorder=4)
    if i:
        c=t.mean(0);ax.set(xlim=(c[0]-.045,c[0]+.045),ylim=(c[1]-.045,c[1]+.045))
        for vertex,point in zip(faces[focus],t):ax.annotate(str(vertex),point[:2],xytext=(5,5),textcoords='offset points')
    ax.set_aspect('equal');ax.set(xlabel='World x (m)',ylabel='World y (m)',title=title+'\nRed: triangle 136')
fig.tight_layout();fig.savefig(out/'suspect_triangle.png',dpi=150);plt.close(fig)
lines=['Shirt L0 stall diagnosis (read only)',
       'Frames 303-317 each used 1024 Newton iterations and exactly 1024 energy trials: no backtracking and no line-search exhaustion.',
       'No contact-feasible, CCD or CFL shrink events in these frames. By source control flow, the tiny accepted steps came from the custom PSQ strain/rank cap.',
       'Final alpha ranges from '+str(min(r['last_alpha'] for r in rows if 303<=r['frame']<=317))+' to '+str(max(r['last_alpha'] for r in rows if 303<=r['frame']<=317))+'.',
       'Triangle 136 is the worst rest triangle by quality, with 66.94mm longest edge and 0.620mm minimum altitude.',
       'Its saved maximum stretch stays near 1.099996 (limit 1.1), leaving about 4e-6 strain margin.',
       'The limiter divides this unsigned margin by ||delta_F||, so it restricts compression as well as stretch.',
       'At saved frame 302 a hypothetical uniform 1% contraction is strain/rank-safe for the entire full segment, but the current limiter allows only alpha='+str(example[1]['weyl_cap'])+'. This is not the actual Newton direction and has not been checked for energy/contact.',
       'Tiny alpha also prevents semi-implicit beta from decaying quickly; strict increment tolerance remains unmet.',
       'Actual binding face and tight safe alpha for the failed Newton directions require a targeted instrumented replay; those directions were not saved.',
       'Recommended next experiment: instrument limiter face, raw direction and separate strain/contact caps; compare a direction-aware stretch cap while retaining rank/CCD protections and the 1.1 strain limit. Do not simply increase the Newton cap or relax the physical strain bound.']
(out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(diagnosis,indent=2))
