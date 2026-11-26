"""Compare actual saved pickup states for any number of hierarchy levels."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import ipctk
from trimesh.triangles import closest_point, points_to_barycentric
from shirt_demo.report import load_run, draw, plt, Image


def measure(d, end):
    x, f, poses = d['x'], d['f'], d['robot']
    hands = {}
    if end >= 596:
        ks = np.arange(596, end+1)
        for side, j in [('left', 1), ('right', 5)]:
            w = poses[ks, j]
            tip = w[0, :3, :3] @ np.array([.150, .001786, 0]) + w[0, :3, 3]
            cp = closest_point(x[596][f], np.broadcast_to(tip, (len(f), 3)))
            dist = np.linalg.norm(cp-tip, axis=1); face = int(dist.argmin())
            bary = points_to_barycentric(x[596][f[face]][None], cp[face][None])[0]
            point = np.einsum('j,njk->nk', bary, x[:, f[face]][ks])
            local = np.einsum('nij,nj->ni', np.linalg.inv(w[:, :3, :3]), point-w[:, :3, 3])
            drift = np.linalg.norm(local-local[0], axis=1)*1000
            lift = (point[:, 2]-point[0, 2])*1000
            hold_drop = float(lift[896-596]-lift[-1]) if end >= 896 else None
            hands[side] = dict(face=face, barycentric=bary.tolist(),
                pinch_center_surface_distance_at_closure_mm=float(dist[face]*1000),
                final_material_lift_mm=float(lift[-1]), maximum_wrist_relative_drift_mm=float(drift.max()),
                hold_height_drop_mm=hold_drop,
                geometric_hold_passed=bool(end==1076 and lift[-1]>=45 and drift.max()<=10 and abs(hold_drop)<=3))
    edges=np.unique(np.sort(f[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(x[0],edges,f.astype(np.int32))
    samples=[]
    for k in sorted({min(i,end) for i in [0,60,356,596,656,896,1076]}):
        samples.append(dict(step=k, self_intersection=bool(ipctk.has_intersections(cm,x[k])),
            height_percentiles_mm=(np.percentile(x[k,:,2]-.8,[0,50,90,100])*1000).tolist()))
    return dict(hands=hands, samples=samples, both_geometric_holds_passed=bool(len(hands)==2 and all(h['geometric_hold_passed'] for h in hands.values())))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--skip-animation',action='store_true');a=p.parse_args();root=a.root
    selection=json.loads((root/'selection.json').read_text());assets=Path(selection['assets'])
    count=len(json.loads((assets/'manifest.json').read_text())['levels'])
    data=[load_run(root/f'L{k}/run') for k in range(count)]
    assert all(d['meta']['status']!='running' for d in data)
    end=min(d['meta']['completed_steps'] for d in data);assert end>0
    baseline=load_run(Path('output/shirt_L2_mesh_improvement_20261117/full_fold/L2/run'))
    out=root/'comparison';out.mkdir(exist_ok=True);rows=[]
    for k,d in enumerate(data):
        m=d['meta'];audit=json.loads((root/f'L{k}/run/report/audit.json').read_text())
        for key in ['config','material_parameters','robot_q_sha256','trajectory_sha256','runtime_hashes']:
            assert m[key]==baseline['meta'][key],key
        if k:assert audit['velocity_transfer_audit']['all_injected_velocities_verified']
        row=dict(level=k,vertices=m['vertices'],faces=m['faces'],status=m['status'],
            completed_steps=m['completed_steps'],physics_seconds=m['physics_seconds'],
            native_steps=m['native_converged_steps'],preview_steps=m['preview_stopped_steps'],
            newton_limit_steps=m['newton_limit_steps'],maximum_stretch=audit['maximum_principal_stretch'],
            robot_surface_tracking_error_mm=audit['maximum_body_surface_tracking_error_m']*1000,
            common_prefix=measure(d,end))
        rows.append(row)
    assert data[-1]['meta']['finest_requires_convergence'] and data[-1]['meta']['preview_stopped_steps']==0
    np.testing.assert_array_equal(data[-1]['x'][0],baseline['x'][0]);np.testing.assert_array_equal(data[-1]['f'],baseline['f'])
    result=dict(common_steps=end,requested_steps=1076,all_completed=all(d['meta']['completed'] and d['meta']['completed_steps']==1076 for d in data),
        levels=rows,baseline_finest=measure(baseline,end),unchanged_finest_geometry_physics_trajectory_runtime=True,
        limitations=['Fixed barycentric tracking is geometric evidence, not a contact-force measurement.',
            'Geometric hold screening: lift >=45mm, drift <=10mm, hold height change <=3mm.',
            'Only sampled self intersections are checked; continuous collision freedom is not certified.',
            'Preview exits are not Newton convergence. Only actual saved positions are rendered.',
            'The pilot ends with jaws closed; no release or full fold is simulated.'])
    (out/'pickup_comparison.json').write_text(json.dumps(result,indent=2))
    for d in data+[baseline]:d['plot_bounds']=[(.15,1.25),(-.55,.55),(.77,1.26)]
    stages=[('Settled',60),('Closed',596),('Lifted',896),('Hold end',1076)]
    stages=[(n,k) for n,k in stages if k<=end]
    fig=plt.figure(figsize=(4*len(stages),3.7*count))
    for l,d in enumerate(data):
        for c,(name,k) in enumerate(stages):
            draw(fig.add_subplot(count,len(stages),l*len(stages)+c+1,projection='3d'),d,k,f'L{l} | {name} | {k/120:.2f}s',False)
    fig.tight_layout(h_pad=3);fig.savefig(out/'four_levels_stages.png',dpi=120);plt.close(fig)
    fig=plt.figure(figsize=(12,5))
    for i,(label,d) in enumerate([('Previous 3-level finest',baseline),('New 4-level finest',data[-1])]):
        draw(fig.add_subplot(1,2,i+1,projection='3d'),d,end,f'{label} | {end/120:.2f}s',False)
    fig.tight_layout();fig.savefig(out/'finest_hold_comparison.png',dpi=140);plt.close(fig)
    if not a.skip_animation:
        images=[]
        for k in np.unique(np.linspace(0,end,90).astype(int)):
            fig=plt.figure(figsize=(12,9),dpi=85)
            for l,d in enumerate(data):draw(fig.add_subplot(2,2,l+1,projection='3d'),d,k,f'L{l} | {d["meta"]["faces"]:,} triangles',False)
            fig.suptitle(f'Actual progressive pickup states | {k/120:.2f}s');fig.tight_layout(h_pad=2)
            fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        images[0].save(out/'four_levels_pickup.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
