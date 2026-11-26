"""Independent cloth/gripper proximity samples; not native contact-force measurements."""
import argparse,json
from pathlib import Path
import numpy as np
import ipctk


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);a=p.parse_args()
    root=a.experiment/'run';m=json.loads((root/'run.json').read_text());assert m['completed']
    plan=json.loads((a.experiment/'plan.json').read_text());s=plan['stages_physical_steps']
    x=np.load(root/'positions.npy',mmap_mode='r');poses=np.load(root/'robot.npy',mmap_mode='r');f=np.load(root/'faces.npy');n=x.shape[1]
    colliders=[dict(np.load(root/f'collider_{i}.npz')) for i in range(8)]
    samples=[]
    for stage in ['placement_end','pre_release_hold_end','opened','retreat_end','observe_end']:
        k=s[stage];hands=[]
        for j,body in enumerate(colliders):
            w=body['vertices']@poses[k,j,:3,:3].T+poses[k,j,:3,3]
            v=np.concatenate([x[k],w]);faces=np.concatenate([f,body['faces']+n]).astype(np.int32)
            edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
            mesh=ipctk.CollisionMesh(v,edges,faces)
            mesh.can_collide=ipctk.make_vertex_patches_filter(np.r_[np.zeros(n,dtype=np.int32),np.ones(len(w),dtype=np.int32)])
            contacts=ipctk.NormalCollisions();contacts.collision_set_type=ipctk.NormalCollisions.IPC;contacts.use_area_weighting=False
            contacts.build(mesh,v,.0015,0.)
            hands.append(dict(collider=j,active_proximity_count=len(contacts),
                minimum_active_distance_mm=None if contacts.empty() else float(np.sqrt(contacts.compute_minimum_distance(mesh,v)))*1000))
        samples.append(dict(stage=stage,step=k,colliders=hands));print(stage,sum(v['active_proximity_count'] for v in hands),flush=True)
    result=dict(samples=samples,scope='Independent IPC Toolkit cloth versus actual unsimplified gripper geometry; proximity threshold 1.5 mm, dmin=0.',
        limitations=['Candidate multiplicities/thresholds need not match native contact constraints. No contact-force or friction measurement.',
            'Zero counts only certify no proximity within the threshold at these discrete sampled states.'])
    (root/'report/release_proximity.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
