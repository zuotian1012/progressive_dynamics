"""Retain the Scene527 first fold, check a deeper right-hand grasp, and append a hold."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


TCP_LOCAL = np.array([.155, .001786, .014])
GRASP_FRAMES = np.arange(130, 261)
RIGHT_ARM = slice(11, 17)
TABLE_Z = .8
TOOL_CLEARANCE = .0015
HOLD_START = 620
HOLD_FRAMES = 180


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def clearance(links):
    return min(float(array(link.get_AABB()).reshape(2, 3)[0, 2] - TABLE_Z) for link in links)


def pairs(robot):
    return {tuple(sorted(pair)) for pair in array(robot.detect_collision()).reshape(-1, 2).tolist()}


def baseline_poses(robot, q, wrist, links):
    records = []
    for frame in GRASP_FRAMES:
        robot.set_qpos(q[frame], zero_velocity=True)
        quat = array(wrist.get_quat(relative=False)).reshape(4)
        position = array(wrist.get_pos(relative=False)).reshape(3)
        tcp = position + Rotation.from_quat(quat[[1, 2, 3, 0]]).apply(TCP_LOCAL)
        records.append((tcp, quat, clearance(links), pairs(robot)))
    return records


def correct_motion(robot, q, wrist, links, depth, baseline):
    result = q.copy()
    records = []
    ramp = np.clip(np.minimum((GRASP_FRAMES - 130) / 30, (260 - GRASP_FRAMES) / 35), 0, 1)
    ramp = ramp**2 * (3 - 2 * ramp)
    for frame, amount, (tcp, quat, old_clearance, old_pairs) in zip(GRASP_FRAMES, ramp, baseline, strict=True):
        desired = tcp - [0., 0., depth * amount]
        robot.set_qpos(q[frame], zero_velocity=True)
        if depth * amount:
            solved = robot.inverse_kinematics(
                link=wrist, pos=desired, quat=quat, local_point=TCP_LOCAL,
                init_qpos=q[frame], dofs_idx_local=np.arange(RIGHT_ARM.start, RIGHT_ARM.stop),
                max_samples=1, max_solver_iters=70, damping=.04, max_step_size=.1,
                pos_tol=.00012, rot_tol=.001, rot_mask=[True, True, True], respect_joint_limit=True,
            )
            result[frame, RIGHT_ARM] = array(solved).reshape(-1)[RIGHT_ARM]
        robot.set_qpos(result[frame], zero_velocity=True)
        actual_quat = array(wrist.get_quat(relative=False)).reshape(4)
        relative_rotation = 2 * np.arccos(np.clip(abs(np.dot(quat, actual_quat)), 0, 1))
        actual = (array(wrist.get_pos(relative=False)).reshape(3)
                  + Rotation.from_quat(actual_quat[[1, 2, 3, 0]]).apply(TCP_LOCAL))
        gap = clearance(links)
        record = dict(frame=int(frame), tcp_z_m=float(actual[2]), clearance_m=gap,
                      baseline_clearance_m=old_clearance, position_error_m=float(np.linalg.norm(actual - desired)),
                      rotation_error_rad=float(relative_rotation), new_collisions=len(pairs(robot) - old_pairs))
        records.append(record)
    position_ok = all(row['position_error_m'] < .00035 and row['rotation_error_rad'] < .0175 for row in records)
    contact_ok = all(row['clearance_m'] >= min(row['baseline_clearance_m'], TOOL_CLEARANCE) - .00005
                     and not row['new_collisions'] for row in records)
    input_change = result[:, RIGHT_ARM] - q[:, RIGHT_ARM]
    step_ok = np.max(np.abs(np.diff(input_change, axis=0))) < .04
    return result, records, bool(position_ok and contact_ok and step_ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('handoff', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--urdf', type=Path, required=True)
    args = parser.parse_args()
    import genesis as gs

    with np.load(args.handoff / 'controls/trajectory.npz', allow_pickle=False) as source:
        q = source['joint_q'][:HOLD_START+1].copy()
        openness = source['openness'][:HOLD_START+1].copy()
        names = source['joint_names']
    assert np.all(openness[-1] == 1)
    gs.init(backend=gs.cpu, logging_level='warning', seed=0)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/60), show_viewer=False,
                     rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True))
    robot = scene.add_entity(morph=gs.morphs.URDF(file=str(args.urdf.resolve()), pos=(0, 0, .17), fixed=True,
                                               visualization=False, collision=True, convexify=False, decimate=False))
    for joint in robot.joints:
        if joint.n_qs:
            joint._init_qpos = np.zeros(joint.n_qs)
    scene.build()
    if [joint.name for joint in robot.joints if joint.n_qs] != names.tolist():
        raise ValueError('Unexpected X5 source joint order')
    wrist = robot.get_link(name='right_link26')
    links = [robot.get_link(name=name) for name in ('right_link26', 'right_link27', 'right_link28')]
    baseline = baseline_poses(robot, q, wrist, links)
    trials = []
    for depth in (.005, .004, .003, .002, .001, .00075, .0005, .00025, .0):
        corrected, rows, ok = correct_motion(robot, q, wrist, links, depth, baseline)
        trials.append(dict(depth_m=depth, accepted=ok, minimum_tool_clearance_m=min(row['clearance_m'] for row in rows),
                           max_ik_error_m=max(row['position_error_m'] for row in rows),
                           new_collision_frames=sum(bool(row['new_collisions']) for row in rows)))
        print(trials[-1], flush=True)
        if ok:
            break
    else:
        raise ValueError('Scene527 reference motion failed the rigid preflight')
    q = np.concatenate((corrected, np.repeat(corrected[-1:], HOLD_FRAMES, axis=0)))
    openness = np.concatenate((openness, np.repeat(openness[-1:], HOLD_FRAMES, axis=0)))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'trajectory.npz', joint_q=q, openness=openness, joint_names=names)
    (args.output / 'trajectory_check.json').write_text(json.dumps(dict(
        fps=60, actions=len(q), original_last_source_frame=HOLD_START, hold_frames=HOLD_FRAMES,
        adjusted_hand='right', depth_m=depth, trials=trials, rig_check=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
