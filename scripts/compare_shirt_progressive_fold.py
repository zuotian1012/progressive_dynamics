"""Compare a completed L0 first fold and its actual PSQ/VelPro L1 simulation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz
from progressive.prolongation import ShellProlongation
from shirt_demo.report import load_run,draw,plt,Image
from trimesh.triangles import closest_point

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('lower',type=Path);p.add_argument('fine',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
data=[load_run(a.lower),load_run(a.fine)]
audits=[json.loads((path/'report/first_fold_audit.json').read_text()) for path in [a.lower,a.fine]]
metas=[d['meta'] for d in data];assert all(m['completed'] and m['completed_steps']==1540 for m in metas)
assert Path(metas[1]['lower_run']).resolve()==a.lower.resolve()
assert metas[0]['dt']==metas[1]['dt']
assets=Path(metas[1]['parameters']['assets'])
P=ShellProlongation(data[0]['x'][0],data[1]['x'][0],data[0]['f'],load_npz(assets/'map_0_1.npz'),np.load(assets/'anchors_0_1.npy'))
errors=[]
for k in range(1541):
    delta=data[1]['x'][k]-P.position(data[0]['x'][k])
    errors.append(float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))))
np.save(a.output/'L1_vs_psq_prolongated_L0_rms.npy',errors)
closure_surface=[]
for j,d in enumerate(data):
    for k in [498,500]:
        for hand,link in [('left',1),('right',5)]:
            pose=d['robot'][k,link];tcp=pose[:3,:3]@np.array([.155,.001786,.014])+pose[:3,3]
            nearest=closest_point(d['x'][k][d['f']],np.broadcast_to(tcp,(len(d['f']),3)))
            distance=np.linalg.norm(nearest-tcp,axis=1)
            closure_surface.append(dict(level=j,step=k,hand=hand,tcp_to_cloth_surface_mm=float(distance.min()*1000)))
rows=[]
for m,v in zip(metas,audits):
    timing=np.load(Path(m['parameters']['output'])/'timing.npz')
    rows.append(dict(level=m['level'],vertices=m['vertices'],faces=m['faces'],completed_steps=m['completed_steps'],
        recorded_step_seconds=m['physics_seconds'],predictor_and_audit_seconds=float(timing['predictor_seconds'].sum()),
        advance_retrieve_seconds_excluding_predictor=float((timing['step_seconds']-timing['predictor_seconds']).sum()),
        native_converged_steps=v['native_converged_steps'],preview_steps=v['preview_steps'],newton_limit_frames=v['newton_limit_frames'],
        rejected_infinite_trial_count=v.get('rejected_infinite_trial_count',0),
        maximum_principal_stretch=v['maximum_principal_stretch'],minimum_singular_value=v['minimum_singular_value'],
        final_convex_hull_area_ratio=v['stages'][-1]['convex_hull_xy_area_ratio_vs_settled'],
        final_height_percentiles_m=v['stages'][-1]['height_above_table_percentiles_m'],
        last_half_second_rms_displacement_mm=v['last_half_second_vertex_rms_displacement_m']*1000,
        peak_surface_target_error=v['peak_surface_tracking_error'],
        grip_through_transfer=v['grip_through_transfer']))
result=dict(scope='One L1 dynamic refinement of the approved L0 first-fold trajectory, not a direct-L1 benchmark',
    physical_seconds=1540*metas[0]['dt'],runs=rows,
    closure_tcp_to_surface=closure_surface,
    L1_vs_psq_prolongated_L0_uniform_vertex_rms_mm=dict(mean=float(np.mean(errors)*1000),maximum=float(max(errors)*1000),final=float(errors[-1]*1000)),
    velocity_transfer_audit=json.loads((a.fine/'report/velocity_transfer_audit.json').read_text()),
    timing_note='L1 recorded step time includes velocity transfer, native readback audit and injected-velocity memmap flush. The separate advance/retrieve column subtracts recorded predictor time. No direct L1 speedup claim.',
    limitations=['L0 contains two Newton-limit steps, inherited as input states.',
                 'L1 is an intermediate preview layer; non-descent exits do not establish strict Newton convergence.',
                 'Geometric difference is not error against ground truth; no independent collision or layer-quality certification.'])
(a.output/'comparison.json').write_text(json.dumps(result,indent=2))

def overhead(ax):
    ax.view_init(elev=90,azim=-90);ax.set_proj_type('ortho');ax.set_zticks([]);ax.set_zlabel('')

norm=plt.Normalize(min(d['x'][0,:,0].min() for d in data),max(d['x'][0,:,0].max() for d in data))
fig=plt.figure(figsize=(12,10),layout='constrained');colored=[]
for j,d in enumerate(data):
    label=f'L{j}: {len(d["f"]):,} triangles'
    ax=fig.add_subplot(2,2,j+1,projection='3d');draw(ax,d,1540,f'{label} | final',True)
    ax=fig.add_subplot(2,2,j+3,projection='3d');draw(ax,d,1540,f'{label} | material colors',True);overhead(ax)
    ax.collections[0].set_facecolor(plt.cm.viridis(norm(d['x'][0,d['f'],0].mean(1))));colored.append(ax)
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=colored,shrink=.6,label='Original material x (m)')
fig.suptitle('Actual first-fold states | L0 input and progressive L1')
fig.savefig(a.output/'final_comparison.png',dpi=140);plt.close(fig)

fig=plt.figure(figsize=(14,8))
for j,d in enumerate(data):
    for i,(label,k) in enumerate([('Lift end',788),('Transfer end',1076),('After release',1540)]):
        ax=fig.add_subplot(2,3,j*3+i+1,projection='3d');draw(ax,d,k,f'L{j} | {label} | {k*metas[0]["dt"]:.2f}s',True)
fig.tight_layout();fig.savefig(a.output/'stage_comparison.png',dpi=140);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained');ax.plot(np.arange(1541)*metas[0]['dt'],np.asarray(errors)*1000)
ax.set(xlabel='Physical time (s)',ylabel='Uniform-vertex RMS difference (mm)',title='Simulated L1 versus nonlinear PSQ prolongation of saved L0')
fig.savefig(a.output/'geometric_difference.png',dpi=140);plt.close(fig)
images=[]
for k in np.unique(np.linspace(0,1540,100).astype(int)):
    fig=plt.figure(figsize=(11,5),dpi=85)
    for j,d in enumerate(data):
        ax=fig.add_subplot(1,2,j+1,projection='3d');draw(ax,d,k,f'L{j} | {len(d["f"]):,} triangles',True)
    fig.suptitle(f'L0 input -> Progressive L1 | t={k*metas[0]["dt"]:.2f}s | Actual saved states')
    fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
duration=round(1540*metas[0]['dt']*1000/len(images)/10)*10
images[0].save(a.output/'comparison.gif',save_all=True,append_images=images[1:],duration=duration,loop=0)
print(json.dumps(result,indent=2))
