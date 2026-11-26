"""Measure the real shirt/robot sequence, including an optional force audit.

Tracked barycentric material points are measurements only; no simulation constraints
are introduced by this offline script. Incomplete phases remain pending.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from progressive.run import prepare_scene
from progressive.affine import AffineObstacles
from progressive.contact import IPCContact
from progressive.runtime import numerical_threads
from progressive.diagnostics import body_contact_forces,nearest_material_point


def measure(root,forces=False):
    cfg=json.loads((root/'experiment.json').read_text())
    args=argparse.Namespace(**cfg['parameters']);args.assets=Path(args.assets)
    if args.scene!='shirt':raise ValueError('This measurement is for Scene527')
    if getattr(args,'hold_after_frame',None)!=300 or args.frames<331:
        raise ValueError('This protocol requires hold-after-frame 300 and at least 331 source frames')
    close=args.settle_steps+219*args.substeps
    lift=args.settle_steps+301*args.substeps
    end=args.settle_steps+331*args.substeps
    h,bodies,fixed,_,_,_,dt,_=prepare_scene(args)
    result={'source':str(root.resolve()),'complete_experiment':cfg['completed'],
            'closure_state':close,'lift_state':lift,'hold_end_state':end,'runs':[],
            'scope':'Original robot trajectory through frame 300, then hold; material tracking plus optional reconstructed IPC contact forces, not a no-slip proof'}
    for folder in sorted(root.glob('*_L*')):
        if not (folder/'run.json').exists():continue
        meta=json.loads((folder/'run.json').read_text());accepted=meta['completed_steps']
        row={'folder':folder.name,'accepted_steps':accepted,'phase':'pending_closure','hands':{}}
        result['runs'].append(row)
        if accepted<close:continue
        x=np.load(folder/'positions.npy',mmap_mode='r');poses=np.load(folder/'robot.npy',mmap_mode='r')
        level=meta['level'];embedding=AffineObstacles(h.rests[level],h.faces[level],bodies,fixed)
        finish=min(end,accepted);indices=np.arange(close,finish+1)
        row['phase']='hold_complete' if finish>=end else ('holding' if finish>=lift else 'lifting')
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for axis,side in zip(axes,('left','right')):
            wrist_name=f'{side}_link'+('16' if side=='left' else '26')
            wrist_id=embedding.names.index(wrist_name)
            wrist=poses[indices,wrist_id]
            tip=np.einsum('nij,j->ni',wrist[:,:3,:3],np.array([.155,.001786,.014]))+wrist[:,:3,3]
            face,weights,distance=nearest_material_point(x[close],h.faces[level],tip[0])
            ids=h.faces[level][face]
            points=np.einsum('j,njk->nk',weights,np.asarray(x[indices][:,ids]))
            local=np.einsum('nij,nj->ni',np.linalg.inv(wrist[:,:3,:3]),points-wrist[:,:3,3])
            shift=points[:,2]-points[0,2];jaw=tip[:,2]-tip[0,2]
            drift=np.linalg.norm(local-local[0],axis=1)
            row['hands'][side]={'material_face':face,'material_vertex_ids':ids.tolist(),'barycentric_weights':weights.tolist(),
                                'nearest_tip_at_closure_m':distance,
                                'peak_material_point_lift_m':float(shift.max()),'final_material_point_lift_m':float(shift[-1]),
                                'final_tip_lift_m':float(jaw[-1]),'max_wrist_relative_drift_m':float(drift.max())}
            if finish>=lift:
                hold_values=shift[max(0,lift-close):]
                row['hands'][side]['hold_material_point_drop_m']=float(hold_values[0]-hold_values[-1])
            axis.plot(indices*dt,shift*1000,label='Fixed cloth material point')
            axis.plot(indices*dt,jaw*1000,'--',label='Actual gripper tip')
            axis.set(title=side,xlabel='Physical time (s)',ylabel='Lift from closure (mm)');axis.grid(alpha=.25);axis.legend()
        fig.savefig(folder/'grasp_tracking.png',dpi=150);plt.close(fig)
        if forces:
            contact=IPCContact(embedding.positions(embedding.q0),embedding.faces,embedding.n,
                               args.dhat,args.dmin,args.contact_stiffness,args.friction)
            def state(k):
                q=embedding.q0.copy();q[:embedding.n]=x[k]
                for i,t in enumerate(poses[k]):q[embedding.n+4*i:embedding.n+4*i+4]=np.vstack((t[:3,3],t[:3,:3].T))
                return embedding.positions(q)
            samples=[]
            for k in np.unique(np.linspace(close,finish,min(15,finish-close+1)).astype(int)):
                before,now=state(k-1),state(k)
                record={'step':int(k),'bodies':body_contact_forces(contact,embedding,before,now,dt)}
                samples.append(record)
            row['force_samples']=samples
    (root/'grasp_measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
    print([(r['folder'],r['phase']) for r in result['runs']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--forces',action='store_true');a=p.parse_args()
    with numerical_threads(1):measure(a.root,a.forces)
