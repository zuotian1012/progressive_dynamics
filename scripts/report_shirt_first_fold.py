"""Inspect the complete source first-fold segment using saved physical states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import ConvexHull
from shirt_demo.report import load_run, draw, plt, Image
from shirt_demo.assets import BUNDLE, Robot, LINKS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--skip-animation', action='store_true')
a = p.parse_args()
root = a.run
out = root / 'report'
data = load_run(root)
m = data['meta']
n = m['completed_steps']
assert m['completed'] and n == 1540 and m['parameters']['frames'] == 740
audit = json.loads((out / 'audit.json').read_text())
records = json.loads((root / 'steps.json').read_text())
source_manifest = BUNDLE / 'source_records/01_first_fold/manifest.json'
source = json.loads(source_manifest.read_text())
assert source['frame_range'] == [0, 739] and source['frames_to_execute'] == 740
assert m['parameters']['settle_steps'] == 60 and m['parameters']['substeps'] == 2
step = lambda frame: 60 + 2 * (frame + 1)
# The compiled trajectory contains initial frame 0, followed by the action durations.
plan = json.loads((BUNDLE / 'source_records/01_first_fold/plan.json').read_text())
cursor = 0
action_ends = []
for action in plan['actions']:
    cursor += action['duration_frames']
    action_ends.append(dict(action=action['id'], control_frame=cursor, step=step(cursor)))
assert cursor == 739
stages = [('Settled', 60), ('Closed', step(219)), ('Lift / pull end', step(363)),
          ('Transfer end', step(507)), ('Opened', step(579)), ('Released / observed', step(739))]
robot = Robot()
assert np.all(robot.openness[219:508] < 1e-8)
assert np.all(robot.openness[579:740] > .999)
x = data['x'][:n+1]
surface = np.load(out / 'body_surface_tracking_error.npy')
stretch = np.load(out / 'maximum_stretch.npy')
base_area = ConvexHull(x[60, :, :2]).volume
base_span = np.ptp(x[60, :, :2], axis=0)
stage_metrics = []
for name, k in stages:
    points = x[k]
    stage_metrics.append(dict(stage=name, step=k, time_s=k*m['dt'],
        xy_span_m=np.ptp(points[:, :2], axis=0).tolist(),
        convex_hull_xy_area_m2=float(ConvexHull(points[:, :2]).volume),
        convex_hull_xy_area_ratio_vs_settled=float(ConvexHull(points[:, :2]).volume/base_area),
        bbox_xy_area_ratio_vs_settled=float(np.prod(np.ptp(points[:, :2], axis=0))/np.prod(base_span)),
        vertex_mean_shift_from_settled_m=(points.mean(0)-x[60].mean(0)).tolist(),
        height_above_table_percentiles_m=np.percentile(points[:, 2]-.8, [0, 50, 90, 100]).tolist()))
phases = []
start = 1
for name, end in stages:
    rows = records[start-1:end]
    phases.append(dict(phase=name, first_step=start, last_step=end,
        native_converged_steps=sum(r['native']['converged'] for r in rows),
        preview_steps=sum(r['native']['preview_non_descent'] for r in rows),
        newton_limit_frames=[r['step'] for r in rows if r['native']['hit_newton_limit']],
        non_preview_nonconverged_frames=[r['step'] for r in rows if not (r['native']['converged'] or r['native']['preview_non_descent'])],
        peak_surface_target_error_mm=float(surface[start-1:end].max()*1000)))
    start = end+1
grip = {}
for hand, j in [('left', 1), ('right', 5)]:
    ids = audit['grip_region_tracking'][hand]['tracked_vertex_ids']
    ks = np.arange(498, step(507)+1)
    poses = data['robot'][ks, j]
    center = x[ks][:, ids].mean(1)
    local = np.einsum('nij,nj->ni', np.linalg.inv(poses[:, :3, :3]), center-poses[:, :3, 3])
    drift = np.linalg.norm(local-local[0], axis=1)
    subsets = {}
    for count in [1, 2, 4, 8]:
        points = x[ks][:, ids[:count]].mean(1)
        relative = np.einsum('nij,nj->ni', np.linalg.inv(poses[:, :3, :3]), points-poses[:, :3, 3])
        delta = np.linalg.norm(relative-relative[0], axis=1)
        subsets[str(count)] = dict(material_vertex_ids=ids[:count],
            maximum_wrist_relative_drift_mm=float(delta.max()*1000), peak_step=int(ks[delta.argmax()]))
    grip[hand] = dict(material_vertex_ids=ids,
        first_step=int(ks[0]), last_step=int(ks[-1]),
        maximum_wrist_relative_drift_mm=float(drift.max()*1000),
        peak_drift_step=int(ks[drift.argmax()]),
        mean_region_displacement_at_transfer_end_m=(center[-1]-center[0]).tolist(),
        subsets=subsets,
        note='Same nearest eight material vertices as the pilot, tracked until opening begins. No post-release holding requirement.')
peak_step, peak_link = np.unravel_index(surface.argmax(), surface.shape)
last_velocity = np.diff(x[-61:], axis=0)/m['dt']
result = dict(scope=f'L{m["level"]} PSQ preview; complete source first-fold segment 0..739; not all folds or finest-level convergence',
    completed_steps=n, physical_seconds=n*m['dt'], physics_seconds=m['physics_seconds'],
    source_manifest=str(source_manifest), source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
    source_action_endpoints=action_ends, stages=stage_metrics, phases=phases,
    native_converged_steps=audit['native_converged_steps'], preview_steps=audit['preview_stopped_steps'],
    rejected_infinite_trial_count=audit.get('rejected_infinite_trial_count',0),
    all_preview_evidence_verified=audit['preview_audit']['all_preview_evidence_verified'],
    newton_limit_frames=audit['newton_limit_frames'],
    non_preview_nonconverged_frames=audit['preview_audit']['unclassified_nonconverged_frames'],
    maximum_principal_stretch=audit['maximum_principal_stretch'], minimum_singular_value=audit['minimum_singular_value'],
    peak_surface_tracking_error=dict(step=int(peak_step+1), link=LINKS[peak_link], error_mm=float(surface.max()*1000)),
    grip_through_transfer=grip,
    last_half_second_vertex_rms_speed_mps=float(np.sqrt(np.mean(np.sum(last_velocity**2, axis=-1)))),
    last_half_second_vertex_rms_displacement_m=float(np.sqrt(np.mean(np.sum((x[-1]-x[-61])**2, axis=-1)))),
    prefix_grasp_screening=audit['grip_region_tracking'],
    numerical_and_tracking_screening=audit['screening'],
    fold_quality_passed=None,
    limitations=['Footprint reduction alone does not establish a correctly layered fold.',
                'The existing grip screening covers only the original lift pilot, not the complete fold.',
                'No independent self-intersection, contact-force, or semantic layer audit.',
                'One run; preview exits are not Newton convergence.'])
(out / 'first_fold_audit.json').write_text(json.dumps(result, indent=2))

def overhead_view(ax):
    ax.view_init(elev=90, azim=-90)
    ax.set_proj_type('ortho')
    ax.set_zticks([])
    ax.set_zlabel('')

for overhead in [False, True]:
    fig = plt.figure(figsize=(13, 8))
    for i, (name, k) in enumerate(stages):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        draw(ax, data, k, f'{name} | step {k} | {k*m["dt"]:.2f}s', True)
        if overhead: overhead_view(ax)
    fig.suptitle(f'L{m["level"]} first fold | Physical cloth and robot states | 1x drive, non-descent preview')
    fig.tight_layout()
    fig.savefig(out / ('first_fold_overhead.png' if overhead else 'first_fold_stages.png'), dpi=140)
    plt.close(fig)
fig = plt.figure(figsize=(10, 8))
for i, (label, elev, azim) in enumerate([('Overview',48,-125), ('Overhead',90,-90), ('Side',10,-90), ('Opposite',25,55)]):
    ax = fig.add_subplot(2,2,i+1,projection='3d')
    draw(ax,data,n,f'Final state | {label}',True)
    ax.view_init(elev=elev,azim=azim)
    if elev == 90: overhead_view(ax)
fig.tight_layout(); fig.savefig(out/'first_fold_final_views.png',dpi=150); plt.close(fig)
material_x = x[0, data['f'], 0].mean(1)
norm = plt.Normalize(float(material_x.min()), float(material_x.max()))
fig = plt.figure(figsize=(11,5), layout='constrained')
axes_material = []
for i, (label, k) in enumerate([('Settled',60), ('Final after release',n)]):
    ax = fig.add_subplot(1,2,i+1,projection='3d'); axes_material.append(ax)
    draw(ax,data,k,label,True)
    ax.collections[0].set_facecolor(plt.cm.viridis(norm(material_x)))
    overhead_view(ax)
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'), ax=axes_material,
             shrink=.65, label='Original material x coordinate (m)')
fig.suptitle('Fixed material colors: settled versus final physical cloth')
fig.savefig(out/'first_fold_material_tracking.png',dpi=150);plt.close(fig)
fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
axes[0].plot(np.arange(n+1)*m['dt'],stretch)
axes[0].axhline(1.1,color='red',ls='--');axes[0].set_ylabel('Maximum principal stretch')
axes[1].plot(np.arange(1,n+1)*m['dt'],surface.max(1)*1000)
axes[1].axhline(10,color='red',ls='--');axes[1].set(xlabel='Physical time (s)',ylabel='Peak surface target error (mm)')
for ax in axes:
    for name,k in stages[1:-1]:ax.axvline(k*m['dt'],color='grey',alpha=.3)
fig.savefig(out/'first_fold_metrics.png',dpi=150);plt.close(fig)
images=[]
for k in ([] if a.skip_animation else np.unique(np.linspace(0,n,128).astype(int))):
    fig=plt.figure(figsize=(10,4.5),dpi=85)
    for i,overhead in enumerate([False,True]):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        draw(ax,data,k,f'{"Overhead" if overhead else "Overview"} | step {k}',True)
        if overhead:overhead_view(ax)
    fig.suptitle(f'L{m["level"]} first fold | t={k*m["dt"]:.2f}s | Saved physical states')
    fig.tight_layout();fig.canvas.draw()
    images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
if images:
    duration=max(10,round(n*m['dt']*1000/len(images)/10)*10)
    images[0].save(out/'first_fold.gif',save_all=True,append_images=images[1:],duration=duration,loop=0)
print(json.dumps(result,indent=2))
