"""Measure physical opening and cloth separation in a saved release continuation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import ipctk
from shirt_demo.assets import Robot, LINKS
from shirt_demo.report import load_run, draw, plt


def transformed(v, t):
    return v@t[:3,:3].T+t[:3,3]


def pair_proximity(a, fa, b, fb, threshold):
    n=len(a);v=np.concatenate((a,b));f=np.concatenate((fa,fb+n)).astype(np.int32)
    e=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    mesh=ipctk.CollisionMesh(v,e,f)
    mesh.can_collide=ipctk.make_vertex_patches_filter(np.r_[np.zeros(n,dtype=np.int32),np.ones(len(b),dtype=np.int32)])
    contacts=ipctk.NormalCollisions();contacts.collision_set_type=ipctk.NormalCollisions.IPC
    contacts.use_area_weighting=False;contacts.build(mesh,v,threshold,0.)
    return dict(proximity_stencils=len(contacts),minimum_distance_mm=None if contacts.empty() else float(np.sqrt(contacts.compute_minimum_distance(mesh,v))*1000))


def angle(a,b):
    rotations=[]
    for t in [a,b]:
        u,_,vh=np.linalg.svd(t[:3,:3]);rotations.append(u@vh)
    return float(np.degrees(np.arccos(np.clip((np.trace(rotations[0].T@rotations[1])-1)/2,-1,1))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('probe',type=Path)
    p.add_argument('--reference',type=Path,default=Path('output/shirt_L2_feasibility_fix_20261118/L2/run'))
    p.add_argument('--reference-label',default='Original')
    p.add_argument('--run-label',default='Repaired MAS')
    args=p.parse_args()
    full_run=(args.probe/'run.json').exists()
    if full_run:
        data=load_run(args.probe);info=data['meta'];meta=info;source=args.probe;start=0
        end=info['completed_steps'];data['x']=data['x'][:end+1];data['robot']=data['robot'][:end+1]
        before=load_run(args.reference)
    else:
        info=json.loads((args.probe/'probe.json').read_text());source=Path(info['source']);start=info['snapshot_step']
        data=load_run(source);meta=data['meta'];before=data.copy()
        data['x']=np.load(args.probe/'probe_positions.npy');data['robot']=np.load(args.probe/'probe_robot.npy')
    robot=Robot(meta['trajectory_path'])
    # Include the far table edge and any cloth hanging below it; do not crop at z=.78.
    before['plot_bounds']=data['plot_bounds']=((.50,1.30),(-.50,.40),(.62,1.18))
    bodies=[dict(np.load(source/f'collider_{j}.npz')) for j in range(8)]
    end=start+len(data['x'])-1
    stages=[k for k in [1616,2216,2576,2696,2816,2936,3116,3476] if start<=k<=end]
    output=[]
    def targets(k):
        frame,s=divmod(k-1-meta['parameters']['settle_steps'],meta['parameters']['substeps'])
        q0=robot.q[max(0,frame-1)];q=q0+(s+1)/meta['parameters']['substeps']*(robot.q[frame]-q0)
        poses=robot.poses(q);return np.stack([poses[n] for n in LINKS])
    for k in stages:
        target=targets(k)
        for label,d,i in [('original',before,k),('repaired',data,k-start)]:
            poses=d['robot'][i];x=d['x'][i]
            surface_error=max(np.linalg.norm(transformed(b['vertices'],poses[j])-transformed(b['vertices'],target[j]),axis=1).max() for j,b in enumerate(bodies))*1000
            row=dict(step=k,variant=label,max_origin_error_mm=float(np.linalg.norm(poses[:,:3,3]-target[:,:3,3],axis=1).max()*1000),
                max_surface_target_error_mm=float(surface_error),cloth_min_height_above_table_mm=float((x[:,2].min()-.8)*1000),
                cloth_max_height_above_table_mm=float((x[:,2].max()-.8)*1000),hands=[],robot_proximity=[])
            inside=(x[:,0]>.1501)&(x[:,0]<1.1499)&(abs(x[:,1])<.9999)
            below=x[:,2]<.799
            row['vertices_below_top_inside_table_footprint']=int(np.sum(inside&below))
            row['vertices_hanging_outside_table']=int(np.sum(~inside&below))
            row['minimum_height_inside_table_footprint_mm']=float((x[inside,2].min()-.8)*1000)
            for side,j in [('left',2),('right',6)]:
                tips=[]
                for index in [j,j+1]:
                    b=bodies[index];f=b['faces'][np.all(b['vertices'][b['faces'],0]>.055,axis=1)]
                    ids,inv=np.unique(f,return_inverse=True);tips.append((b['vertices'][ids],inv.reshape(-1,3)))
                va,fa=tips[0];vb,fb=tips[1]
                actual=pair_proximity(transformed(va,poses[j]),fa,transformed(vb,poses[j+1]),fb,.5)
                desired=pair_proximity(transformed(va,target[j]),fa,transformed(vb,target[j+1]),fb,.5)
                row['hands'].append(dict(side=side,actual_tip_gap_mm=actual['minimum_distance_mm'],target_tip_gap_mm=desired['minimum_distance_mm'],
                    relative_finger_angle_deg=angle(poses[j],poses[j+1])))
            for j in range(8):
                b=bodies[j]
                row['robot_proximity'].append(dict(body=LINKS[j],**pair_proximity(x,d['f'],transformed(b['vertices'],poses[j]),b['faces'],.0015)))
            output.append(row)
            print(json.dumps(row),flush=True)
    (args.probe/'release_measurements.json').write_text(json.dumps(dict(samples=output,from_step_zero=full_run,
        reference_run=str(args.reference.resolve()) if full_run else str(source.resolve()),measured_run=str(args.probe.resolve()),
        reference_label=args.reference_label,run_label=args.run_label,
        completed=info.get('completed',False) if full_run else info.get('all_probe_steps_converged',False),
        limitation='Geometry proximity is not a contact-force measurement.'+('' if full_run else ' Snapshot caches were reconstructed.')),indent=2))
    shown=[k for k in [2576,2936,3116,3476] if k in stages]
    if shown:
        lowest=min(float(d['x'][k-offset,:,2].min()) for d,offset in [(before,0),(data,start)] for k in shown)
        before['plot_bounds']=data['plot_bounds']=((.50,1.30),(-.50,.40),(min(.62,lowest-.02),1.18))
        fig=plt.figure(figsize=(5*len(shown),10))
        for col,k in enumerate(shown):
            draw(fig.add_subplot(2,len(shown),col+1,projection='3d'),before,k,f'{args.reference_label} | step {k}')
            draw(fig.add_subplot(2,len(shown),len(shown)+col+1,projection='3d'),data,k-start,f'{args.run_label} | step {k}')
        fig.tight_layout(pad=2,h_pad=4);fig.savefig(args.probe/'release_comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
