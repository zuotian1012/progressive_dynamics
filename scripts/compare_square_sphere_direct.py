"""Compare same-mesh material trajectories to an independent L3 reference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import report_square_sphere_demo as renderer
from run_square_sphere_demo import ROOT, write_json


def main(root):
    out = root/'direct_comparison'
    out.mkdir(exist_ok=True)
    scene = json.loads((root/'scene.json').read_text())
    dt, steps, prefix = scene['dt'], scene['steps'], scene['analytic_freefall_steps']
    mesh = dict(np.load(root/'assets/mesh_3.npz'))
    sphere = dict(np.load(root/'assets/sphere.npz'))
    rest, faces = mesh['vertices'], mesh['faces']
    all_faces = np.vstack((faces, sphere['faces']+len(rest))).astype(np.int32)
    edges = np.unique(np.sort(all_faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    collision = ipctk.CollisionMesh(np.vstack((rest,sphere['vertices'])),edges,all_faces)
    data, audit, metas = {}, {}, {}
    hashes = json.loads((root/'direct_L3/input_hashes.json').read_text())
    for filename, expected in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Direct input changed: {filename}')
    identity = json.loads((root/'assets/L3.json').read_text())['identity']
    assert identity
    for method in ('semi','velpro','direct'):
        path=root/f'{method}_L3'
        meta=json.loads((path/'run.json').read_text())
        records=json.loads((path/'steps.json').read_text())
        x=np.load(path/'positions.npy')
        assert meta['status']=='completed' and x.shape==(steps+1,len(rest),3) and len(records)==steps
        assert np.isfinite(x).all()
        native=[r['native'] for r in records if r['native'] is not None]
        assert len(native)==steps-prefix
        assert all(s['completed'] and s['converged'] and not s['hit_newton_limit'] and not s['hit_line_search_limit'] and not s['preview_non_descent'] for s in native)
        assert max(r['stretch'] for r in records)<1.068000001
        assert max(r['sphere_error_m'] for r in records)==0.
        assert 'clamped' not in (path/'native.log').read_text(errors='replace')
        names={s['name'].split('::')[-1] for s in json.loads((path/'backend/systems.json').read_text())['sim_systems'] if s['valid']}
        assert {'IPCSimplexNormalContact','IPCSimplexFrictionalContact'}<=names
        assert 'SoftPositionConstraint' not in names
        bad=[i for i,y in enumerate(x) if ipctk.has_intersections(collision,np.vstack((y,sphere['vertices'])))]
        assert not bad, (method,bad)
        audit[method]=dict(accepted_steps=steps,analytic_freefall_steps=prefix,native_converged_steps=len(native),
            coarse_preview_frames=0,intersecting_saved_frames=bad,max_stretch=max(r['stretch'] for r in records),
            fixed_sphere_error_m=0.,elapsed_s=meta['elapsed_s'])
        data[method,3]=dict(x=x,mesh=mesh)
        metas[method]=meta
        print(f'Validated {method}: {len(native)} native converged steps; no intersections',flush=True)
    for key in ('config','dt','dll_sha256','friction','requested_contact_stiffness','consistency_weight','pinned_vertices'):
        assert all(metas[m][key]==metas['direct'][key] for m in ('semi','velpro')),key
    assert metas['direct']['coarse_trajectory_used'] is False
    assert metas['direct']['velocity_injection_each_step'] is False
    for method in ('semi','velpro'):
        np.testing.assert_array_equal(data[method,3]['x'][:prefix+1],data['direct',3]['x'][:prefix+1])
    write_json(out/'audit.json',audit)
    tri=rest[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    mass=np.zeros(len(rest))
    np.add.at(mass,faces.ravel(),np.repeat(area*scene['density_kg_m3']*scene['full_thickness_m']/3,3))
    weights=mass/mass.sum()
    reference=data['direct',3]['x']
    time=np.arange(steps+1)*dt
    curves, summary, rows={}, {}, []
    for method in ('semi','velpro'):
        delta=data[method,3]['x']-reference
        rms=np.sqrt(np.einsum('v,tvc,tvc->t',weights,delta,delta))*1000
        com=np.einsum('v,tvc->tc',weights,delta)
        centered=delta-com[:,None,:]
        shape=np.sqrt(np.einsum('v,tvc,tvc->t',weights,centered,centered))*1000
        com_distance=np.linalg.norm(com,axis=1)*1000
        worst=np.linalg.norm(delta,axis=2).max(axis=1)*1000
        curves[method]=dict(rms=rms,shape=shape,com=com_distance,max=worst)
        summary[method]=dict(final_rms_mm=float(rms[-1]),peak_rms_mm=float(rms.max()),
            peak_rms_time_s=float(time[rms.argmax()]),post_freefall_spacetime_rms_mm=float(np.sqrt(np.mean(rms[prefix+1:]**2))),
            final_com_distance_mm=float(com_distance[-1]),final_centered_shape_rms_mm=float(shape[-1]),
            final_max_vertex_distance_mm=float(worst[-1]))
        for i in range(steps+1):
            rows.append((method,i,time[i],rms[i],shape[i],com_distance[i],worst[i]))
    n=steps-prefix
    win=curves['velpro']['rms'][prefix+1:]<curves['semi']['rms'][prefix+1:]
    result=dict(reference='Independent L3 using native implicit Euler; no coarse trajectory or per-step velocity injection',
        identity_proxy=True,dt=dt,steps=steps,analytic_freefall_steps=prefix,cloth_mass_kg=float(mass.sum()),
        metric='same material vertex correspondence, lumped-rest-mass-weighted RMS; fixed world coordinates; no temporal or rigid alignment',
        shape_metric='remove mass-weighted translation only; no rotation alignment',
        aggregate_window_s=[float(time[prefix+1]),float(time[-1])],
        methods=summary,velpro_lower_rms_frames=int(win.sum()),compared_native_frames=n,
        scope='One scene and one discretization; Direct is a numerical reference, not ground truth.')
    write_json(out/'summary.json',result)
    with (out/'per_frame.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('method','frame','time_s','rms_to_direct_mm','translation_removed_rms_mm','com_distance_mm','max_vertex_distance_mm'))
        writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for method,color,label in [('semi','#c87542','Semi-diagonal'),('velpro','#14888f','VelPro')]:
        axes[0].plot(time,curves[method]['rms'],color=color,label=label,lw=2)
        axes[1].plot(time,curves[method]['shape'],color=color,label=label,lw=2)
    for ax in axes:
        ax.axvspan(0,prefix*dt,color='#dce1e8',alpha=.5)
        ax.set(xlabel='Physical time (s)',ylabel='Mass-weighted RMS distance (mm)')
        ax.grid(alpha=.2); ax.legend()
    axes[0].set_title('Distance to Direct L3 / fixed world coordinates')
    axes[1].set_title('Shape difference / translation removed')
    fig.savefig(out/'errors_to_direct.png',dpi=160)
    plt.close(fig)
    renderer.STEPS=steps
    renderer.render(data,sphere,out,audit,True,methods=('semi','velpro','direct'))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path,nargs='?',default=ROOT/'output/square_sphere_demo')
    main(p.parse_args().root.resolve())
