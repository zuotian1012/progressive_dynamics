"""Build a joint-limited L0 grasp/lift/hold boundary-condition experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from shirt_demo.assets import Robot, LINKS

TCP = np.array([.155, .001786, .014])


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10+t*(-15+6*t))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--right-offset-mm', type=float, nargs=3, default=[10., 0., -4.])
    p.add_argument('--lift-mm', type=float, default=60.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    robot = Robot()
    q = np.tile(robot.q[0], (508, 1))
    q[:148] = robot.q[:148]
    # Keep the established left orientation; mirror it for the right jaw tilt.
    base = robot.poses(robot.q[147])
    mirror = np.diag([1., -1., 1.])
    right_rotation = mirror @ base['left_link16'][:3, :3] @ mirror
    original_right_rotation = base['right_link26'][:3, :3]
    rotate = Slerp([0, 1], Rotation.from_matrix([original_right_rotation, right_rotation]))
    joints = {j.get('name'): j for j in robot.xml.findall('joint')}
    errors, clearances = [], []
    collisions = robot.collisions()
    hand_ids = [('left', np.arange(3, 9), 'left_link16'), ('right', np.arange(11, 17), 'right_link26')]
    for k in range(1, len(q)):
        seed_index = min(k, 147)
        q[k] = robot.q[seed_index]
        if k > 147:
            q[k,[9,10,17,18]] = .044*(1-smooth((k-147)/120))
        lift = a.lift_mm*.001*smooth((k-297)/120)
        for side, ids, link in hand_ids:
            source = robot.poses(robot.q[seed_index])[link]
            position = source[:3,:3]@TCP+source[:3,3]
            target_rotation = source[:3,:3]
            if side == 'right':
                blend = smooth(k/75)
                position = position + np.array(a.right_offset_mm)*.001*blend
                # Apply a relative rotation to preserve the original approach path.
                target_rotation = rotate(blend).as_matrix() @ original_right_rotation.T @ target_rotation
            position = position + [0, 0, lift]
            if side == 'left' and k <= 297:
                continue
            bounds = np.array([[float(joints[robot.names[i]].find('limit').get(key)) for i in ids] for key in ['lower','upper']])
            def residual(values):
                state = q[k].copy(); state[ids] = values
                pose = robot.poses(state)[link]
                return np.r_[pose[:3,:3]@TCP+pose[:3,3]-position,
                    .1*Rotation.from_matrix(target_rotation.T@pose[:3,:3]).as_rotvec()]
            solved = least_squares(residual, np.clip(q[k-1,ids], bounds[0]+1e-9, bounds[1]-1e-9),
                bounds=bounds, xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=100)
            if np.max(np.abs(solved.fun)) > 1e-7:
                raise RuntimeError(f'IK residual at frame {k}, {side}: {solved.fun}')
            q[k,ids] = solved.x
            errors.append(float(np.max(np.abs(solved.fun))))
        poses = robot.poses(q[k])
        clearances.append(min(float((v@poses[name][:3,:3].T+poses[name][:3,3])[:,2].min()-.8) for name,v,f in collisions))
    openness = np.column_stack([q[:,[9,10]].mean(1), q[:,[17,18]].mean(1)])/.044
    np.testing.assert_array_equal(q[0], robot.q[0])
    speed = np.max(np.abs(np.diff(q,axis=0))*60,axis=0)
    for i, name in enumerate(robot.names):
        limit = joints[name].find('limit')
        assert speed[i] <= float(limit.get('velocity'))+1e-9, (name, speed[i])
        if limit.get('lower') is not None:
            assert q[:,i].min() >= float(limit.get('lower'))-1e-10
            assert q[:,i].max() <= float(limit.get('upper'))+1e-10
    assert min(clearances) > .0002, min(clearances)
    path=a.output/'trajectory.npz'
    np.savez(path,joint_q=q,joint_names=np.asarray(robot.names),openness=openness)
    stages={'approach_end':75,'grasp_position':147,'closed':267,'close_hold_end':297,'lift_end':417,'hold_end':507}
    result=dict(scope='Robot joint trajectory only; unchanged cloth/contact/PD, original initial state.',
        right_offset_mm=a.right_offset_mm,right_rotation=right_rotation.tolist(),lift_mm=a.lift_mm,
        frames=len(q),stages_control_frames=stages,stages_physical_steps={k:60+2*(v+1) for k,v in stages.items()},
        minimum_target_collider_table_clearance_m=min(clearances),maximum_weighted_ik_residual=max(errors),
        joint_max_speeds=speed.tolist(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trajectory=str(path.resolve()),note='Kinematic checks do not establish grasp or collision-free motion. Native contact simulation is required.')
    (a.output/'plan.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
