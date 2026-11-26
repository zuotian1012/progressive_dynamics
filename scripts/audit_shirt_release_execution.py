"""Inspect saved release states; no simulation or contact-force claims."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import ipctk
from shirt_demo.assets import Robot, LINKS
from shirt_demo.report import load_run, draw, plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BASE = Path('output/shirt_L2_feasibility_fix_20261118')
OUT = BASE / 'release_inspection'
OUT.mkdir(exist_ok=True)
ROOTS = [Path('output/shirt_four_level_full_fold_20261118') / f'L{i}/run' for i in (0, 1)] + [BASE / 'L2/run']
STAGES = {2576:'Placement end',2696:'Opening begins',2756:'Opening 25%',2816:'Opening 50%',2876:'Opening 75%',2936:'Opening ends',3026:'Retreat 50%',3116:'Retreat ends',3296:'Observe 50%',3476:'Final'}

def pair_mesh(a, fa, b, fb):
    n = len(a)
    v = np.concatenate([a, b])
    f = np.concatenate([fa, fb+n]).astype(np.int32)
    e = np.unique(np.sort(f[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    m = ipctk.CollisionMesh(v,e,f)
    m.can_collide = ipctk.make_vertex_patches_filter(np.r_[np.zeros(n,dtype=np.int32),np.ones(len(b),dtype=np.int32)])
    return m,v,e,f

def proximity(mesh,v,threshold=.0015):
    c=ipctk.NormalCollisions()
    c.collision_set_type=ipctk.NormalCollisions.IPC
    c.use_area_weighting=False
    c.build(mesh,v,threshold,0.)
    return c

def target_at(robot,meta,k):
    step=k-1
    settle=meta['parameters']['settle_steps'];sub=meta['parameters']['substeps']
    if step<settle:q=robot.q[0]
    else:
        frame,s=divmod(step-settle,sub)
        previous=robot.q[max(0,frame-1)]
        q=previous+(s+1)/sub*(robot.q[frame]-previous)
    ts=robot.poses(q)
    return np.stack([ts[name] for name in LINKS])

def world(v,t):return v@t[:3,:3].T+t[:3,3]

def tip_patch(body):
    # Distal planar tips only, defined in the original local collision coordinates.
    v,f=body['vertices'],body['faces']
    f=f[np.all(v[f,0]>.055,axis=1)]
    ids,inv=np.unique(f,return_inverse=True)
    return v[ids],inv.reshape(-1,3)

def tip_distance(body1,body2,t1,t2):
    v1,f1=tip_patch(body1);v2,f2=tip_patch(body2)
    mesh,v,_,_=pair_mesh(world(v1,t1),f1,world(v2,t2),f2)
    c=proximity(mesh,v,.5)
    assert not c.empty()
    return float(np.sqrt(c.compute_minimum_distance(mesh,v))*1000)

def relative_angle(t1,t2):
    rotations=[]
    for t in [t1,t2]:
        u,_,vh=np.linalg.svd(t[:3,:3]);rotations.append(u@vh)
    return float(np.degrees(np.arccos(np.clip((np.trace(rotations[0].T@rotations[1])-1)/2,-1,1))))

results=[]
for level,root in enumerate(ROOTS):
    meta=json.loads((root/'run.json').read_text())
    x=np.load(root/'positions.npy',mmap_mode='r');poses=np.load(root/'robot.npy',mmap_mode='r');faces=np.load(root/'faces.npy')
    robot=Robot(meta['trajectory_path'])
    bodies=[dict(np.load(root/f'collider_{j}.npz')) for j in range(8)]
    meshes=[pair_mesh(x[0],faces,world(b['vertices'],poses[0,j]),b['faces']) for j,b in enumerate(bodies)]
    ks=list(STAGES) if level==2 else [2696,2936,3116,3476]
    rows=[]
    for k in ks:
        target=target_at(robot,meta,k)
        row=dict(step=k,stage=STAGES[k],hands=[],contacts=[],cloth_max_height_mm=float((x[k,:,2].max()-.8)*1000))
        for side,j in [('left',2),('right',6)]:
            # Origins are fixed reference markers, NOT the clear space between pads.
            a,b=poses[k,j:j+2];ta,tb=target[j:j+2]
            row['hands'].append(dict(side=side,
                actual_distal_patch_min_distance_mm=tip_distance(bodies[j],bodies[j+1],a,b),
                target_distal_patch_min_distance_mm=tip_distance(bodies[j],bodies[j+1],ta,tb),
                actual_relative_finger_rotation_deg=relative_angle(a,b),
                target_relative_finger_rotation_deg=relative_angle(ta,tb),
                finger_affine_singular_values=[np.linalg.svd(t[:3,:3],compute_uv=False).tolist() for t in [a,b]],
                actual_finger_origin_separation_mm=float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),
                target_finger_origin_separation_mm=float(np.linalg.norm(ta[:3,3]-tb[:3,3])*1000),
                finger_origin_target_errors_mm=(np.linalg.norm(poses[k,j:j+2,:3,3]-target[j:j+2,:3,3],axis=1)*1000).tolist(),
                actual_finger_origin_heights_mm=((poses[k,j:j+2,2,3]-.8)*1000).tolist(),
                target_finger_origin_heights_mm=((target[j:j+2,2,3]-.8)*1000).tolist()))
        for j,b in enumerate(bodies):
            mesh,_,edges,ff=meshes[j]
            v=np.concatenate([x[k],world(b['vertices'],poses[k,j])])
            c=proximity(mesh,v)
            ids=set()
            for ii in range(len(c)):
                ids.update(int(z) for z in c[ii].vertex_ids(edges,ff) if 0<=z<len(x[k]))
            row['contacts'].append(dict(body=j,name=LINKS[j],proximity_count=len(c),
                minimum_distance_mm=None if c.empty() else float(np.sqrt(c.compute_minimum_distance(mesh,v))*1000),
                cloth_stencil_vertex_ids=sorted(ids),
                cloth_stencil_height_range_mm=None if not ids else ((np.array([x[k,sorted(ids),2].min(),x[k,sorted(ids),2].max()])-.8)*1000).tolist()))
        rows.append(row)
        print('L',level,k,'separation',[(round(h['actual_finger_origin_separation_mm'],2),round(h['target_finger_origin_separation_mm'],2)) for h in row['hands']], 'contacts',[r['proximity_count'] for r in row['contacts']],flush=True)
    # Follow the same initially grasped material points throughout release.
    tracked=[]
    pickup=json.loads((BASE/'pickup_verification.json').read_text()) if level==2 else None
    if pickup:
        for side,h in pickup['hands'].items():
            tri=faces[h['face']];bary=np.array(h['barycentric'])
            pts=np.einsum('j,njk->nk',bary,x[2696:3477,tri])
            tracked.append(dict(side=side,release_start_point=pts[0].tolist(),final_point=pts[-1].tolist(),
                release_to_final_displacement_mm=float(np.linalg.norm(pts[-1]-pts[0])*1000),
                heights_mm={str(k):float((pts[k-2696,2]-.8)*1000) for k in ks if k>=2696}))
    results.append(dict(level=level,root=str(root.resolve()),samples=rows,initial_grasp_material=tracked))
    (OUT/'measurements.json').write_text(json.dumps(dict(levels=results,limitations=[
        'Finger origin separation is NOT pad clearance; rotations and body distortions are separate.',
        'Distal patch distance is minimum triangle separation for tip faces whose three local x coordinates exceed 55 mm; it does not establish a parallel jaw aperture.',
        'Independent IPC Toolkit proximity within 1.5 mm, dmin=0, using original unsimplified collision geometry.',
        'Proximity counts are stencils, not independent contact points; no native normal forces or friction forces were saved.',
        'Only discrete states are sampled. Geometry alone cannot prove frictional force balance.'
    ]),indent=2))

fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for i,side in enumerate(['left','right']):
    ax=axes[0,i]
    for res in results:
        rows=res['samples']
        ax.plot([r['step'] for r in rows],[r['hands'][i]['actual_distal_patch_min_distance_mm'] for r in rows],'.-',label=f"L{res['level']} actual")
        axes[1,i].plot([r['step'] for r in rows],[r['hands'][i]['actual_relative_finger_rotation_deg'] for r in rows],'.-',label=f"L{res['level']} actual")
    rows=results[-1]['samples']
    ax.plot([r['step'] for r in rows],[r['hands'][i]['target_distal_patch_min_distance_mm'] for r in rows],'k--',label='Target (all levels)')
    ax.set(title=side.capitalize(),xlabel='Physical step',ylabel='Distal tip minimum surface distance (mm)')
    axes[1,i].set(xlabel='Physical step',ylabel='Relative finger rotation (degrees)')
    axes[1,i].axhline(0,color='black',ls='--',label='Target: parallel fingers')
    for ax in axes[:,i]:
        for k in [2696,2936,3116]:ax.axvline(k,color='gray',alpha=.25)
        ax.legend();ax.grid(alpha=.2)
fig.savefig(OUT/'opening_comparison.png',dpi=150);plt.close(fig)

data=load_run(ROOTS[-1]);meta=data['meta'];robot=Robot(meta['trajectory_path'])
fig=plt.figure(figsize=(14,9))
for idx,k in enumerate([2696,2816,2936,3476]):
    ax=fig.add_subplot(2,2,idx+1,projection='3d');draw(ax,data,k,f'L2 {STAGES[k]} | step {k}')
    for j in [2,3,6,7]:
        v,f=data['colliders'][j]
        actual=world(v,data['robot'][k,j]);target=world(v,target_at(robot,meta,k)[j])
        ax.add_collection3d(Poly3DCollection(actual[f],facecolors='#ee8b32',alpha=.8))
        ax.add_collection3d(Poly3DCollection(target[f],facecolors='#47c77b',alpha=.15,edgecolors='#258548',linewidth=.1))
    row=next(r for r in results[-1]['samples'] if r['step']==k)
    ids=sorted(set(z for c in row['contacts'] for z in c['cloth_stencil_vertex_ids']))
    if ids:
        p=data['x'][k,ids];ax.scatter(*p.T,color='red',s=5,depthshade=False)
fig.suptitle('Blue: cloth | Orange: actual fingers | Green transparent: target fingers | Red: cloth proximity stencils',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'release_stages.png',dpi=150);plt.close(fig)
print('Reports:',OUT.resolve(),flush=True)
