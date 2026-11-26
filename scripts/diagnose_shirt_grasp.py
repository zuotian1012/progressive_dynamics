"""Locate material drift and collider tracking errors after a completed grip pilot."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.assets import Robot,LINKS
from shirt_demo.report import plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
m=json.loads((root/'run.json').read_text());assert m['completed']
audit=json.loads((root/'report/audit.json').read_text());x=np.load(root/'positions.npy');poses=np.load(root/'robot.npy')
k0=498;ks=np.arange(k0,787);result={};fig,ax=plt.subplots(2,1,figsize=(9,7),layout='constrained')
for hand,j in [('left',1),('right',5)]:
    ids=audit['grip_region_tracking'][hand]['tracked_vertex_ids'];w=poses[ks,j]
    local=np.einsum('nij,nkj->nki',np.linalg.inv(w[:,:3,:3]),x[ks][:,ids]-w[:,:3,3][:,None,:])
    tcp=np.einsum('nij,j->ni',w[:,:3,:3],np.array([.155,.001786,.014]))+w[:,:3,3]
    subsets={}
    for count in [1,2,4,8]:
        point=x[ks][:,ids[:count]].mean(1);rel=local[:,:count].mean(1);drift=np.linalg.norm(rel-rel[0],axis=1)
        crossing=np.flatnonzero(drift>.02)
        subsets[str(count)]=dict(vertex_ids=ids[:count],max_relative_drift_m=float(drift.max()),
             max_drift_step=int(ks[drift.argmax()]),first_drift_above_20mm_step=int(ks[crossing[0]]) if len(crossing) else None,
             peak_mean_lift_m=float((point[:,2]-point[0,2]).max()),final_mean_lift_m=float(point[-1,2]-point[0,2]))
        if count in [1,8]:ax[0].plot(ks*m['dt'],drift*1000,label=f'{hand}, nearest {count}',ls='-' if count==1 else '--')
    result[hand]=dict(subsets=subsets,nearest_material_vertex_final_tcp_distance_m=float(np.linalg.norm(x[786,ids[0]]-tcp[-1])))
surface=np.load(root/'report/body_surface_tracking_error.npy');index,link=map(int,np.unravel_index(surface.argmax(),surface.shape));step=index+1
robot=Robot();frame,sub=divmod(index-m['parameters']['settle_steps'],m['parameters']['substeps'])
q=robot.q[max(0,frame-1)]+(sub+1)/m['parameters']['substeps']*(robot.q[frame]-robot.q[max(0,frame-1)])
target=robot.poses(q)[LINKS[link]];actual=poses[step,link];u,s,v=np.linalg.svd(actual[:3,:3]);rot=u@v
angle=np.arccos(np.clip((np.trace(rot.T@target[:3,:3])-1)/2,-1,1))
records=json.loads((root/'steps.json').read_text())
result['peak_surface_error']=dict(step=step,control_frame=frame,link=LINKS[link],error_m=float(surface.max()),
        origin_error_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])),rotation_error_degrees=float(np.degrees(angle)),
        affine_singular_values=s.tolist(),native=records[index]['native'],
        inferred_semi_implicit_stop=step in audit['semi_implicit_stop_frames_inferred'])
result['interpretation']='Material-to-wrist drift is geometric evidence of lost tracking; contact-force history and the cause of left grasp loss are not measured.'
ax[0].axhline(20,color='gray',ls=':');ax[0].set(ylabel='Material drift in wrist frame (mm)');ax[0].legend()
ax[1].plot(np.arange(1,len(surface)+1)*m['dt'],surface.max(1)*1000);ax[1].axhline(10,color='gray',ls=':')
ax[1].set(xlabel='Physical time (s)',ylabel='Max collider target error (mm)')
fig.savefig(root/'report/grasp_diagnostics.png',dpi=140);plt.close(fig)
(root/'report/grasp_diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
