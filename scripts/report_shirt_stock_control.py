"""Compare genuine stock independent L1 dynamics with a fresh PD L1 run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import load_run,draw,plt,Image
from audit_shirt_velocity_transfer import audit as audit_transfer


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rest_inverse(x,f):
    t=x[f];a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];length=np.linalg.norm(a,axis=1)
    area2=np.linalg.norm(np.cross(a,b),axis=1);dm=np.zeros((len(f),2,2))
    dm[:,0,0]=length;dm[:,0,1]=np.sum(a*b,axis=1)/length;dm[:,1,1]=area2/length
    return np.linalg.inv(dm)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    names=['stock_default','stock_semi_off','pd_l1_nopreview'];data={name:load_run(root/name) for name in names}
    stock=data['stock_semi_off']['meta'];pd=data['pd_l1_nopreview']['meta'];assert stock['config']==pd['config']
    m0=data['stock_default']['meta'];expected=copy.deepcopy(m0['config']);expected['newton']['semi_implicit']['enable']=0
    assert expected==stock['config']
    for key in ['young','poisson','density','full_thickness','bending']:assert stock['material_parameters'][key]==pd['material_parameters'][key]
    inverse=rest_inverse(data['stock_semi_off']['x'][0],data['stock_semi_off']['f']);results={};slivers=[30,2673,3413]
    for name,d in data.items():
        m=d['meta'];n=m['completed_steps'];r=root/name
        assert m['status']=='completed' and m['completed'] and n==60
        assert m['mesh_sha256']==stock['mesh_sha256'] and m['robot_q_sha256']==stock['robot_q_sha256'] and m['dt']==stock['dt']
        assert m['config']['contact']==stock['config']['contact'] and m['robot_drive_strength']==stock['robot_drive_strength']
        assert m['config']['gravity']==stock['config']['gravity']
        np.testing.assert_array_equal(d['x'][0],data['stock_semi_off']['x'][0]);np.testing.assert_array_equal(d['f'],data['stock_semi_off']['f'])
        np.testing.assert_array_equal(d['robot'][0],data['stock_semi_off']['robot'][0])
        assert all(digest(path)==h for path,h in m['runtime_hashes'].items())
        for i in range(8):
            for k in ['vertices','faces']:np.testing.assert_array_equal(np.load(r/f'collider_{i}.npz')[k],np.load(root/'stock_semi_off'/f'collider_{i}.npz')[k])
        x=d['x'][:n+1];assert np.isfinite(x).all() and np.isfinite(d['robot'][:n+1]).all()
        rows=json.loads((r/'steps.json').read_text());assert len(rows)==n
        assert all(q['native']['completed'] and q['native']['frame']==i and not q['native'].get('preview_non_descent',False) for i,q in enumerate(rows,1))
        log=r.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace')
        assert 'PSQ_PREVIEW_STOP frame=' not in log and 'PSQ_FORCE_ACCEPT frame=' not in log
        transfer=None;wheel=None
        if name.startswith('stock'):
            assert m['stock_wheel'] and not m['psq_enabled'] and not m['pd_enabled'] and not m['velocity_injection'] and not m['hard_strain_barrier']
            assert not m.get('lower_run') and not m.get('psq_bundle')
            assert 'PSQ_' not in log
            wheel=json.loads((r/'wheel_integrity.json').read_text());assert wheel['all_record_hashes_match']
            package=Path(wheel['package_path']);assert Path(m['backend_path']).is_relative_to(package)
            assert all(Path(path).is_relative_to(package/'_native') and wheel['files'][path]==h for path,h in m['runtime_hashes'].items())
            assert all(digest(path)==h for path,h in wheel['files'].items())
            default=json.loads((r/'native_default_config.json').read_text());expected=copy.deepcopy(default)
            if name=='stock_semi_off':expected['newton']['semi_implicit']['enable']=0
            for key in ['newton','line_search','linear_system','extras','cfl','integrator']:assert expected[key]==m['config'][key]
            actual=np.load(r/'input_velocities.npy');expected_v=np.concatenate([np.zeros_like(x[:1]),np.diff(x[:n],axis=0)/m['dt']])
            velocity_error=float(np.abs(actual-expected_v).max());assert velocity_error<=1e-12
        else:
            assert m['method']=='psq_velpro_L1' and not m.get('coarse_preview')
            transfer=audit_transfer(r);velocity_error=transfer['maximum_native_velocity_readback_error']
            assert m['runtime_hashes']!=stock['runtime_hashes']
        measurements=[]
        for k,z in enumerate(x):
            tri=z[d['f']];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)@inverse
            s=np.linalg.svd(F,compute_uv=False)
            measurements.append(dict(step=k,time_s=k*m['dt'],max_principal_stretch=float(s[:,0].max()),
                stretch_p99=float(np.percentile(s[:,0],99)),faces_above_1_1=int(np.count_nonzero(s[:,0]>1.1)),
                sliver_face_stretch={str(i):float(s[i,0]) for i in slivers},minimum_singular_value=float(s[:,1].min()),
                height_above_table_p50_mm=float(np.median(z[:,2]-.8)*1000)))
        results[name]=dict(status=m['status'],steps=n,physical_seconds=n*m['dt'],stock_wheel=bool(wheel),
            module_version=m.get('package_version',m.get('pyuipc')),runtime_hashes=m['runtime_hashes'],
            native_converged_steps=sum(q['native']['converged'] for q in rows),
            line_search_limit_steps=[q['step'] for q in rows if q['native']['hit_line_search_limit']],
            newton_limit_steps=[q['step'] for q in rows if q['native']['hit_newton_limit']],preview_steps=0,
            semi_implicit_enabled=bool(m['config']['newton']['semi_implicit']['enable']),
            native_input_velocity_formula_error=velocity_error,velocity_transfer_audit=transfer,
            maximum_principal_stretch=max(t['max_principal_stretch'] for t in measurements),
            first_step=measurements[1],last_step=measurements[-1],measurements=measurements,physics_seconds=m['physics_seconds'])
    result=dict(scope='Replacement of the rejected PSQ-independent control; all three runs are newly executed.',
        matched_stock_off_pd_scene_and_solver_config=True,matched_nominal_material_constants=True,
        matched_initial_L1_mesh_and_robot_colliders=True,stock_loaded_wheel_binaries_verified=True,
        runs=results,
        limitations=['Stock computes elastic energy on L1; PD uses the finest PSQ proxy and lower-level velocities.',
          'Stock has no custom 1.1 hard strain barrier. This is not an otherwise-identical constrained model comparison.',
          'Stock wheel 0.0.28 and modified source runtime differ; absence of PSQ in stock is verified by loaded DLL paths and wheel RECORD hashes.',
          'PD L1 preview is disabled; reused L0 includes preview exits. PD line-search failures remain marked.',
          'Native-default convergence can include semi-implicit early termination; matched stock control disables it.',
          'Rendering uses flat triangle shading on the same mesh. Not every triangular highlight is a numerical artifact.',
          'No independent self-intersection audit; convergence is not proof of physical accuracy.'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    selected=['stock_semi_off','pd_l1_nopreview'];labels=['Stock libuipc: independent L1','PD: PSQ + VelPro L1']
    fig=plt.figure(figsize=(12,5))
    for i,(name,label) in enumerate(zip(selected,labels)):
        draw(fig.add_subplot(1,2,i+1,projection='3d'),data[name],60,label+' | 0.500 s',False)
    fig.suptitle('Same L1 mesh and solver settings; different elastic evaluation and strain barrier')
    fig.tight_layout();fig.savefig(root/'comparison_final.png',dpi=145,bbox_inches='tight');plt.close(fig)
    fig=plt.figure(figsize=(12,9))
    for row,k in enumerate([1,10]):
        for i,(name,label) in enumerate(zip(selected,labels)):
            draw(fig.add_subplot(2,2,row*2+i+1,projection='3d'),data[name],k,f'{label} | step {k}',False)
    fig.suptitle('Actual early states; no upsampled surrogate geometry');fig.tight_layout();fig.savefig(root/'early_states.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained',sharex=True)
    for name in names:
        mm=results[name]['measurements'];axes[0].plot([t['step'] for t in mm],[t['max_principal_stretch'] for t in mm],label=name)
        axes[1].plot([t['step'] for t in mm],[t['height_above_table_p50_mm'] for t in mm],label=name)
    axes[0].axhline(1.1,color='gray',ls='--');axes[0].set(ylabel='Maximum principal stretch',title='The stock model has no custom 1.1 strain barrier')
    axes[1].set(xlabel='Step',ylabel='Median height above table (mm)')
    for ax in axes:ax.legend()
    fig.savefig(root/'metrics.png',dpi=140);plt.close(fig)
    images=[]
    for k in np.unique(np.linspace(0,60,45).astype(int)):
        fig=plt.figure(figsize=(12,5))
        for i,(name,label) in enumerate(zip(selected,labels)):
            suffix=' | exhausted searches recorded' if any(t<=k for t in results[name]['line_search_limit_steps']) else ''
            draw(fig.add_subplot(1,2,i+1,projection='3d'),data[name],k,f'{label} | {k/120:.3f} s'+suffix,False)
        fig.suptitle('Actual trajectories | semi-implicit OFF | stock has no custom strain barrier | slowed playback')
        fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    images[0].save(root/'comparison.gif',save_all=True,append_images=images[1:],duration=85,loop=0)
    summary={name:{k:v for k,v in row.items() if k not in ('measurements','velocity_transfer_audit','runtime_hashes')} for name,row in results.items()}
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
