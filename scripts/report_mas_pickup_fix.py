"""Measure saved MAS repair pickup; tracking points do not constrain simulation."""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from progressive.diagnostics import nearest_material_point
from shirt_demo.report import load_run, draw, plt

root = Path('output/shirt_L2_mas_fix_20261119')
probe = root/'pickup_536_full_budget'
source = Path('output/shirt_L2_feasibility_fix_20261118/L2/run')
data = load_run(source)
data['x'] = np.load(probe/'probe_positions.npy')
data['robot'] = np.load(probe/'probe_robot.npy')
info = json.loads((probe/'probe.json').read_text())
history = json.loads((probe/'history.json').read_text())
assert info['completed_probe_steps'] == 540 and info['all_probe_steps_converged']
counts = [int(s) for s in re.findall(r'Iterative linear solver iteration count: (\d+)', probe.with_suffix('.log').read_text(errors='replace'))]
metrics = dict(start=536, end=1076, steps=540, seconds=info['wall_seconds'],
               all_steps_passed=True, pcg_caps=sum(n>=51690 for n in counts),
               maximum_single_pcg_iterations=max(counts),
               maximum_newton_iterations=max(h['stats']['newton_iterations'] for h in history),
               final_origin_error_mm=history[-1]['max_robot_origin_target_error_mm'],
               maximum_origin_error_mm=max(h['max_robot_origin_target_error_mm'] for h in history), hands={})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
start = 596-536
for ax, (side, wrist_id) in zip(axes, [('left',1), ('right',5)]):
    wrist = data['robot'][start:, wrist_id]
    tip = np.einsum('nij,j->ni', wrist[:,:3,:3], [.155,.001786,.014]) + wrist[:,:3,3]
    face, weights, distance = nearest_material_point(data['x'][start], data['f'], tip[0])
    ids = data['f'][face]
    point = np.einsum('j,njk->nk',weights,data['x'][start:][:,ids])
    local = np.einsum('nij,nj->ni',np.linalg.inv(wrist[:,:3,:3]),point-wrist[:,:3,3])
    lift = (point[:,2]-point[0,2])*1000
    hand_lift = (tip[:,2]-tip[0,2])*1000
    poses = data['robot'][-1,wrist_id+1:wrist_id+3]
    rotations = [np.linalg.svd(t[:3,:3])[0]@np.linalg.svd(t[:3,:3])[2] for t in poses]
    angle = np.degrees(np.arccos(np.clip((np.trace(rotations[0].T@rotations[1])-1)/2,-1,1)))
    metrics['hands'][side] = dict(material_face=int(face), material_weights=np.asarray(weights).tolist(),
        initial_material_distance_mm=float(distance*1000), final_material_lift_mm=float(lift[-1]),
        actual_tip_lift_mm=float(hand_lift[-1]), maximum_relative_drift_mm=float(np.linalg.norm(local-local[0],axis=1).max()*1000),
        material_height_change_last_second_mm=float(lift[-1]-lift[-121]), final_finger_relative_angle_deg=float(angle))
    times=np.arange(596,1077)/120
    ax.plot(times,lift,label='Tracked cloth material point')
    ax.plot(times,hand_lift,'--',label='Actual gripper tip')
    ax.set(title=side, xlabel='Simulation time (s)', ylabel='Lift from closure (mm)')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(root/'material_tracking.png',dpi=150);plt.close(fig)
fig=plt.figure(figsize=(12,8))
for i,(k,title) in enumerate([(536,'Start'),(596,'Closed'),(956,'Lifted'),(1076,'Hold end')]):
    draw(fig.add_subplot(2,2,i+1,projection='3d'),data,k-536,f'{title} | step {k}')
fig.suptitle('L2 repaired MAS | saved-state continuation, physical gripper contacts',fontsize=12)
fig.tight_layout();fig.savefig(root/'pickup_stages.png',dpi=150);plt.close(fig)
metrics['limitation']='Nearest material point at closure is a tracking measurement; drift may include rolling/sliding. This is not a contact-force or no-slip proof, nor a full-fold validation.'
(root/'pickup_measurements.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
