"""Audit matched L1 direct/VelPro settling runs, separating strict and diagnostic states."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz
from progressive.prolongation import frames
from shirt_demo.report import load_run,draw,plt,Image
from audit_shirt_velocity_transfer import audit as transfer_audit


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(a):return float(np.sqrt(np.mean(np.sum(a*a,axis=-1))))


def edge_pairs(f):
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);owners=np.tile(np.arange(len(f)),3)
    edges=np.sort(edges,axis=1);order=np.lexsort((edges[:,1],edges[:,0]));edges=edges[order];owners=owners[order]
    start=np.r_[0,1+np.flatnonzero(np.any(edges[1:]!=edges[:-1],axis=1)),len(edges)]
    ii=start[:-1][np.diff(start)==2]
    return edges[ii],np.stack([owners[ii],owners[ii+1]],axis=1)


def angles(x,f,edges,adj):
    tri=x[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    e=x[edges[:,1]]-x[edges[:,0]];e/=np.linalg.norm(e,axis=1)[:,None]
    a,b=n[adj[:,0]],n[adj[:,1]]
    return np.arctan2(np.sum(np.cross(a,b)*e,axis=1),np.sum(a*b,axis=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);a=parser.parse_args();root=a.root
    names=['direct','velpro','direct_repeat','velpro_repeat','direct_diagnostic','velpro_diagnostic']
    data={n:load_run(root/n) for n in names};baseline=data['direct']['meta'];results={};tracks={}
    assets=Path(baseline['parameters']['assets']);U=load_npz(assets/'map_0_1.npz')
    inv=frames(data['direct']['x'][0],data['direct']['f'])[2]
    edges,adj=edge_pairs(data['direct']['f']);theta0=angles(data['direct']['x'][0],data['direct']['f'],edges,adj)
    for name,d in data.items():
        m=d['meta'];n=m['completed_steps'];assert m['status']!='running' and n>0
        for key in ['config','material_parameters','mesh_sha256','psq_bundle_sha256','robot_q_sha256','runtime_hashes','dt']:
            assert m[key]==baseline[key],(name,key)
        assert m['level']==1 and m['steps']==60 and m['parameters']['frames']==0 and m['parameters']['settle_steps']==60
        assert not m.get('coarse_preview') and not m.get('force_updates')
        assert m['config']['newton']['semi_implicit']['enable']==0
        defaults=json.loads((root/name/'native_default_config.json').read_text());expected=copy.deepcopy(defaults)
        expected['newton']['semi_implicit']['enable']=0
        for key in ['newton','line_search','linear_system','extras','cfl','integrator']:assert m['config'][key]==expected[key]
        assert m['solver_overrides']=={'newton.semi_implicit.enable':0}
        assert digest(assets/'mesh_1.npz')==m['mesh_sha256'] and digest(m['psq_bundle'])==m['psq_bundle_sha256']
        assert all(digest(path)==h for path,h in m['runtime_hashes'].items())
        np.testing.assert_array_equal(d['f'],data['direct']['f']);np.testing.assert_array_equal(d['x'][0],data['direct']['x'][0])
        np.testing.assert_array_equal(d['robot'][0],data['direct']['robot'][0])
        x=d['x'][:n+1];assert np.isfinite(x).all() and np.isfinite(d['robot'][:n+1]).all()
        rows=json.loads((root/name/'steps.json').read_text());assert len(rows)==n
        assert all(r['native']['completed'] and r['native']['frame']==k+1 and not r['native']['preview_non_descent'] for k,r in enumerate(rows))
        clean=[r['native']['converged'] and not r['native']['hit_newton_limit'] and not r['native']['hit_line_search_limit'] for r in rows]
        diagnostic=name.endswith('_diagnostic')
        if not diagnostic:assert m['requires_convergence'] and all(clean)
        else:assert m['diagnostic_only'] and not m.get('requires_convergence')
        velocity=np.load(root/name/'input_velocities.npy',mmap_mode='r')[:n]
        audit=None
        if name.startswith('velpro'):
            audit=transfer_audit(root/name);low=np.load(Path(m['lower_run'])/'positions.npy',mmap_mode='r')
            expected_v=np.stack([np.zeros_like(x[0]) if k==0 else U@((low[k]-low[k-1])/m['dt']) for k in range(n)])
        else:
            assert not m.get('lower_run') and not (root/name/'injected_velocities.npy').exists()
            expected_v=np.concatenate([np.zeros_like(x[:1]),np.diff(x[:n],axis=0)/m['dt']])
        velocity_error=float(np.abs(velocity-expected_v).max());assert velocity_error<=1e-12
        per_step=[]
        for k,z in enumerate(x):
            tri=z[d['f']];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)@inv
            sigma=np.linalg.svd(F,compute_uv=False)
            theta=angles(z,d['f'],edges,adj);change=np.degrees(np.abs(np.arctan2(np.sin(theta-theta0),np.cos(theta-theta0))))
            per_step.append(dict(step=k,time_s=k*m['dt'],max_stretch=float(sigma[:,0].max()),min_sigma=float(sigma[:,1].min()),
                near_limit_triangles=int(np.count_nonzero(sigma[:,0]>1.099)),height_above_table_p50_mm=float(np.median(z[:,2]-.8)*1000),
                height_above_table_p95_mm=float(np.percentile(z[:,2]-.8,95)*1000),
                dihedral_change_p95_deg=float(np.percentile(change,95)),edges_changed_over_30deg=int(np.count_nonzero(change>30)),
                clean_step=True if k==0 else bool(clean[k-1])))
        tracks[name]=per_step
        log=(root/name).with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace')
        assert 'PSQ_PREVIEW_STOP frame=' not in log and 'PSQ_FORCE_ACCEPT frame=' not in log
        fail=json.loads((root/name/'failed_step.json').read_text()) if (root/name/'failed_step.json').exists() else None
        searches=[dict(frame=int(f),newton=int(it),relative_energy_increase=float(inc)) for f,it,inc in re.findall(r'Line Search Exits with Max Iteration: \d+ \(Frame=(\d+), Newton=(\d+),.*?rel_E_increase=([^,]+),',log)]
        results[name]=dict(status=m['status'],requested_steps=60,saved_steps=n,completed=m['completed'],diagnostic_only=diagnostic,
            native_converged_steps=sum(r['native']['converged'] for r in rows),clean_steps=sum(clean),
            exhausted_search_steps=[r['step'] for r in rows if r['native']['hit_line_search_limit']],
            newton_limit_steps=[r['step'] for r in rows if r['native']['hit_newton_limit']],failed_step=fail,
            physics_seconds=m['physics_seconds'],native_input_velocity_max_error=velocity_error,velocity_transfer_audit=audit,
            maximum_principal_stretch=max(t['max_stretch'] for t in per_step),strict_stretch_screen_passed=all(t['max_stretch']<1.1 for t in per_step),
            exhausted_search_events=searches,measurements=per_step)
    pairs=[]
    for first,second in [('direct','velpro'),('direct_repeat','velpro_repeat'),('direct','direct_repeat'),('velpro','velpro_repeat'),('direct_diagnostic','velpro_diagnostic')]:
        n=min(data[first]['meta']['completed_steps'],data[second]['meta']['completed_steps'])
        pairs.append(dict(first=first,second=second,common_steps=n,samples=[dict(step=k,rms_vertex_difference_mm=1000*rms(data[first]['x'][k]-data[second]['x'][k])) for k in sorted({0,min(1,n),min(10,n),n})]))
    result=dict(runs=results,pairs=pairs,all_matched_controls_verified=True,all_native_input_velocities_verified=True,
        primary_question='Same L1 dynamics with own velocity versus U times existing L0 velocity; both L1 branches disable preview.',
        limitations=['The reused L0 input has preview exits, first at step 15; this does not test an entirely preview-free hierarchy.',
          'Both strict branches stop early. Native-continue trajectories are diagnostic only and contain solver-quality failures.',
          'Direct here retains the SAME PSQ proxy energy and modified native backend; it is not an unmodified libuipc per-mesh-material baseline.',
          'Dihedral changes describe folding on the same mesh, not an automatic artifact classification.',
          'This experiment does not isolate energy gradients, contact, discretization or the upstream L0 error.'])
    rest=data['direct']['x'][0];faces=data['direct']['f'];tri=rest[faces]
    lengths=np.linalg.norm(tri[:,[1,2,0]]-tri,axis=2)
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    quality=4*np.sqrt(3)*area/np.sum(lengths**2,axis=1)
    near={}
    for name in names[:4]:
        tx=data[name]['x'][1][faces];F=np.stack([tx[:,1]-tx[:,0],tx[:,2]-tx[:,0]],axis=-1)@inv
        stretch=np.linalg.svd(F,compute_uv=False)[:,0]
        near[name]=np.flatnonzero(stretch>1.099).tolist()
        assert np.max(np.abs(np.load(root/name/'input_velocities.npy',mmap_mode='r')[0]))==0
    worst=np.argsort(quality)[:3]
    result['first_step_common_evidence']=dict(near_limit_face_ids=near,all_first_input_velocities_zero=True,
        three_lowest_quality_face_ids=worst.tolist(),all_four_hit_exactly_three_worst=all(set(ids)==set(worst) for ids in near.values()),
        quality_formula='4 sqrt(3) area / sum(edge_length squared); equilateral=1, degenerate=0',
        rest_face_quality=[dict(face=int(i),quality=float(quality[i]),edge_lengths_mm=(1000*lengths[i]).tolist()) for i in worst],
        interpretation='Shared first-step behavior before velocity predictors differ; correlation with slivers is not yet a causal isolation.')
    (root/'comparison.json').write_text(json.dumps(result,indent=2));(root/'measurement_tracks.json').write_text(json.dumps(tracks,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,i in zip(axs,worst):
        # Display each triangle in its own plane with millimetre units and equal axes.
        t=tri[i];longest=int(np.argmax(lengths[i]));order=[longest,(longest+1)%3,(longest+2)%3];t=t[order]
        e=t[1]-t[0];e/=np.linalg.norm(e);normal=np.cross(t[1]-t[0],t[2]-t[0]);normal/=np.linalg.norm(normal)
        B=np.stack([e,np.cross(normal,e)],axis=1);xy=(t-t[0])@B*1000;xy=np.vstack([xy,xy[0]])
        ax.plot(xy[:,0],xy[:,1],'-o',color='crimson',markersize=3);ax.set_aspect('equal',adjustable='box')
        ax.set(xlabel='Local x (mm)',ylabel='Local y (mm)',title=f'Face {i} | quality {quality[i]:.4f}',ylim=(-2,2))
    fig.suptitle('The same three rest-mesh slivers approach stretch limit at the first step in all four strict runs')
    fig.savefig(root/'initial_sliver_triangles.png',dpi=145);plt.close(fig)
    selected=['direct_diagnostic','velpro_diagnostic'];n=min(data[nm]['meta']['completed_steps'] for nm in selected)
    labels=['Direct L1 (own velocity)','VelPro L1 (L0 velocity)']
    fig=plt.figure(figsize=(12,5))
    for i,(name,label) in enumerate(zip(selected,labels)):
        draw(fig.add_subplot(1,2,i+1,projection='3d'),data[name],n,f'{label} | step {n} | {n/120:.3f} s',False)
    fig.suptitle('DIAGNOSTIC ONLY: native continuation includes exhausted line searches | preview OFF')
    fig.tight_layout();fig.savefig(root/'diagnostic_final.png',dpi=145,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained',sharex=True)
    for name,label in zip(selected,labels):
        r=results[name];t=tracks[name];steps=[p['step'] for p in t]
        axes[0].plot(steps,[p['height_above_table_p50_mm'] for p in t],label=label)
        axes[1].plot(steps,[p['dihedral_change_p95_deg'] for p in t],label=label)
        rr=json.loads((root/name/'steps.json').read_text());axes[2].plot(steps[1:],[v['native']['newton_iterations'] for v in rr],label=label)
        bad=r['exhausted_search_steps'];axes[2].scatter(bad,[rr[k-1]['native']['newton_iterations'] for k in bad],s=18,marker='x')
    axes[0].set(ylabel='Median height over table (mm)',title='Diagnostic trajectories: geometric differences are not validated accuracy')
    axes[1].set(ylabel='95th percentile dihedral change (deg)')
    axes[2].set(xlabel='Step',ylabel='Newton iterations',title='Crosses mark exhausted line search within the step')
    for ax in axes:ax.legend()
    fig.savefig(root/'diagnostic_metrics.png',dpi=140);plt.close(fig)
    images=[]
    for k in np.unique(np.linspace(0,n,min(n+1,45)).astype(int)):
        fig=plt.figure(figsize=(12,5))
        for i,(name,label) in enumerate(zip(selected,labels)):
            bad=any(not p['clean_step'] for p in tracks[name][:k+1])
            draw(fig.add_subplot(1,2,i+1,projection='3d'),data[name],k,f'{label} | {k/120:.3f} s'+(' | solver failures present' if bad else ''),False)
        fig.suptitle('DIAGNOSTIC ONLY | preview OFF | native relaxed continuation | slowed playback')
        fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    images[0].save(root/'diagnostic_comparison.gif',save_all=True,append_images=images[1:],duration=85,loop=0)
    print(json.dumps({name:{k:v for k,v in r.items() if k not in ('measurements','velocity_transfer_audit','exhausted_search_events')} for name,r in results.items()},indent=2))


if __name__=='__main__':main()
