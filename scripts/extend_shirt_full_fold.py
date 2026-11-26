"""Append a fold arc, placement, opening, retreat and observation to a verified prefix."""
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
from build_shirt_grasp_pilot import smooth,TCP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination-x',type=float,default=.7184)
    p.add_argument('--arc-height',type=float,default=.12)
    p.add_argument('--placement-z',type=float,default=.82)
    p.add_argument('--wrist-pitch-deg',type=float,default=0.,help='World-Y wrist rotation smoothly added during the fold arc')
    p.add_argument('--laydown-forward-mm',type=float,default=0.,help='Additional forward translation while lowering')
    p.add_argument('--placement-frames',type=int,default=120)
    p.add_argument('--pre-release-hold-frames',type=int,default=30)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    src=a.prefix/'trajectory.npz';old=json.loads((a.prefix/'plan.json').read_text())
    assert hashlib.sha256(src.read_bytes()).hexdigest()==old['sha256']
    robot=Robot(src);n=len(robot.q)
    assert a.placement_frames>0 and a.pre_release_hold_frames>=0
    durations=[('fold_arc_end',300),('placement_end',a.placement_frames),('pre_release_hold_end',a.pre_release_hold_frames),('opened',120),('retreat_end',90),('observe_end',180)]
    q=np.tile(robot.q[-1],(n+sum(v for _,v in durations),1));q[:n]=robot.q
    base=robot.poses(q[n-1]);joints={j.get('name'):j for j in robot.xml.findall('joint')};collisions=robot.collisions()
    cursor=n-1;stages=old['stages_control_frames'].copy();stages['transfer_hold_end']=stages.pop('hold_end')
    for name,count in durations:cursor+=count;stages[name]=cursor
    errors=[];clearances=[];starts={link:base[link][:3,:3]@TCP+base[link][:3,3] for link in ['left_link16','right_link26']}
    for k in range(n,len(q)):
        q[k]=q[k-1]
        opening=smooth((k-stages['pre_release_hold_end'])/120)
        q[k,[9,10,17,18]]=.044*opening
        for ids,link in [(np.arange(3,9),'left_link16'),(np.arange(11,17),'right_link26')]:
            start=starts[link];position=start.copy();rotation=base[link][:3,:3]
            u=smooth((k-n+1)/300)
            rotation=Rotation.from_rotvec(np.array([0.,np.deg2rad(a.wrist_pitch_deg)*u,0.])).as_matrix()@rotation
            position[0]=start[0]+u*(a.destination_x-start[0])
            position[2]=start[2]+a.arc_height*np.sin(np.pi*u)**2
            if k>stages['fold_arc_end']:
                t=smooth((k-stages['fold_arc_end'])/a.placement_frames)
                position[0]=a.destination_x+t*a.laydown_forward_mm*.001
                position[2]=start[2]+t*(a.placement_z-start[2])
            if k>stages['opened']:
                position[2]=a.placement_z+.1*smooth((k-stages['opened'])/90)
            bounds=np.array([[float(joints[robot.names[i]].find('limit').get(key)) for i in ids] for key in ['lower','upper']])
            def residual(values):
                state=q[k].copy();state[ids]=values;t=robot.poses(state)[link]
                return np.r_[t[:3,:3]@TCP+t[:3,3]-position,.1*Rotation.from_matrix(rotation.T@t[:3,:3]).as_rotvec()]
            solved=least_squares(residual,q[k-1,ids],bounds=bounds,xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=100)
            assert np.max(np.abs(solved.fun))<1e-7,(k,link,solved.fun)
            q[k,ids]=solved.x;errors.append(float(np.abs(solved.fun).max()))
        poses=robot.poses(q[k]);clearances.append(min(float((v@poses[name][:3,:3].T+poses[name][:3,3])[:,2].min()-.8) for name,v,f in collisions))
    np.testing.assert_array_equal(q[:n],robot.q)
    speed=np.max(np.abs(np.diff(q,axis=0))*60,axis=0)
    for i,name in enumerate(robot.names):
        limit=joints[name].find('limit');assert speed[i]<=float(limit.get('velocity'))+1e-9,(name,speed[i])
        if limit.get('lower') is not None:assert q[:,i].min()>=float(limit.get('lower'))-1e-10 and q[:,i].max()<=float(limit.get('upper'))+1e-10
    assert min(clearances)>.0002,min(clearances)
    openness=np.column_stack([q[:,[9,10]].mean(1),q[:,[17,18]].mean(1)])/.044
    np.testing.assert_array_equal(openness[:n],robot.openness)
    dest=a.output/'trajectory.npz';np.savez(dest,joint_q=q,joint_names=robot.names,openness=openness)
    plan=dict(old);plan.update(scope='Complete modified first-fold trajectory; geometry/solver unchanged; no claim of successful fold before simulation.',
        prefix_trajectory=str(src.resolve()),prefix_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),exact_prefix_frames=n,
        frames=len(q),destination_x_m=a.destination_x,arc_extra_height_m=a.arc_height,placement_tcp_z_m=a.placement_z,
        wrist_pitch_deg=a.wrist_pitch_deg,laydown_forward_mm=a.laydown_forward_mm,placement_frames=a.placement_frames,pre_release_hold_frames=a.pre_release_hold_frames,
        stages_control_frames=stages,stages_physical_steps={k:60+2*(v+1) for k,v in stages.items()},
        minimum_target_collider_table_clearance_m=min(old['minimum_target_collider_table_clearance_m'],min(clearances)),
        maximum_weighted_ik_residual=max(old['maximum_weighted_ik_residual'],max(errors)),joint_max_speeds=speed.tolist(),
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),trajectory=str(dest.resolve()))
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
