"""Append a slow Cartesian transfer to an exact saved grasp-pilot prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from shirt_demo.assets import Robot
from build_shirt_grasp_pilot import smooth, TCP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--translation-mm',type=float,nargs=3,default=[100.,0.,0.])
    p.add_argument('--move-frames',type=int,default=180)
    p.add_argument('--hold-frames',type=int,default=90)
    a=p.parse_args()
    assert a.move_frames>0 and a.hold_frames>0
    a.output.mkdir(parents=True,exist_ok=False)
    source=a.prefix/'trajectory.npz';plan=json.loads((a.prefix/'plan.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['sha256']
    robot=Robot(source);n=len(robot.q);q=np.tile(robot.q[-1],(n+a.move_frames+a.hold_frames,1));q[:n]=robot.q
    joints={j.get('name'):j for j in robot.xml.findall('joint')}
    targets=robot.poses(q[n-1]);errors=[];clearances=[];collisions=robot.collisions()
    for k in range(n,len(q)):
        q[k]=q[k-1]
        delta=np.array(a.translation_mm)*.001*smooth((k-n+1)/a.move_frames)
        for ids,link in [(np.arange(3,9),'left_link16'),(np.arange(11,17),'right_link26')]:
            target=targets[link];position=target[:3,:3]@TCP+target[:3,3]+delta
            bounds=np.array([[float(joints[robot.names[i]].find('limit').get(key)) for i in ids] for key in ['lower','upper']])
            def residual(values):
                state=q[k].copy();state[ids]=values;t=robot.poses(state)[link]
                return np.r_[t[:3,:3]@TCP+t[:3,3]-position,.1*Rotation.from_matrix(target[:3,:3].T@t[:3,:3]).as_rotvec()]
            solved=least_squares(residual,q[k-1,ids],bounds=bounds,xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100)
            assert np.max(np.abs(solved.fun))<1e-7,(k,link,solved.fun)
            q[k,ids]=solved.x;errors.append(float(np.abs(solved.fun).max()))
        poses=robot.poses(q[k])
        clearances.append(min(float((v@poses[name][:3,:3].T+poses[name][:3,3])[:,2].min()-.8) for name,v,f in collisions))
    np.testing.assert_array_equal(q[:n],robot.q)
    speed=np.max(np.abs(np.diff(q,axis=0))*60,axis=0)
    for i,name in enumerate(robot.names):
        limit=joints[name].find('limit');assert speed[i]<=float(limit.get('velocity'))+1e-9
        if limit.get('lower') is not None:
            assert q[:,i].min()>=float(limit.get('lower'))-1e-10 and q[:,i].max()<=float(limit.get('upper'))+1e-10
    assert min(clearances)>.0002
    openness=np.column_stack([q[:,[9,10]].mean(1),q[:,[17,18]].mean(1)])/.044
    np.testing.assert_array_equal(openness[:n],robot.openness)
    path=a.output/'trajectory.npz';np.savez(path,joint_q=q,joint_names=robot.names,openness=openness)
    stages=plan['stages_control_frames'].copy();stages['lift_hold_end']=stages.pop('hold_end')
    stages.update(transfer_end=n+a.move_frames-1,hold_end=len(q)-1)
    plan.update(prefix_trajectory=str(source.resolve()),prefix_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        exact_prefix_frames=n,translation_mm=a.translation_mm,move_frames=a.move_frames,hold_frames=a.hold_frames,
        frames=len(q),stages_control_frames=stages,stages_physical_steps={k:60+2*(v+1) for k,v in stages.items()},
        minimum_target_collider_table_clearance_m=min(plan['minimum_target_collider_table_clearance_m'],min(clearances)),
        maximum_weighted_ik_residual=max(plan['maximum_weighted_ik_residual'],max(errors)),joint_max_speeds=speed.tolist(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trajectory=str(path.resolve()))
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
