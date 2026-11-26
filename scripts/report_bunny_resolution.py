"""Audit independent, strictly converged native Bunny resolution controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    a=p.parse_args();root=a.root
    cases=[(i,f'L{i}') for i in range(3)]
    if (root/'L2_repeat/run.json').exists():cases.append((2,'L2_repeat'))
    metas=[json.loads((root/name/'run.json').read_text()) for _,name in cases]
    bundles=[json.loads((root/f'bundles/L{i}.json').read_text()) for i,_ in cases]
    common=min(m['completed_steps'] for m in metas)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    result=dict(requested_seconds=metas[0]['requested_steps']*metas[0]['dt'],
        common_accepted_steps=common,common_seconds=common*metas[0]['dt'],
        independent_collision_audit=False,
        alpha_scope='Last line-search alpha of each accepted frame only, NOT the minimum alpha over every Newton iteration. Failed-frame alpha is available only when logged.',
        timing_scope='Single serial runs; accepted-state advance/retrieve time and whole process wall time reported separately; no statistically established performance claim.',
        cases=[],source_hashes={},library_hashes={})
    for (level,name),m,b in zip(cases,metas,bundles):
        run=root/name
        assert m['config']==metas[0]['config'] and m['material']==metas[0]['material']
        assert b['parameters']==bundles[0]['parameters']
        assert m['assets']['bunny.npz']==metas[0]['assets']['bunny.npz']
        assert m['hard_strain_barrier'] and not m['psq_enabled'] and m['psq_mode']=='reference'
        assert m['method']=='independent' and not m.get('lower_run')
        assert not m.get('coarse_preview') and not m.get('force_updates')
        records=json.loads((run/'steps.json').read_text())
        states=np.load(run/'accepted_positions.npy')
        assert len(states)==m['completed_steps']+1==len(records)+1 and np.isfinite(states).all()
        assert all(r['native']['completed'] and r['native']['converged'] and not r['native']['preview_non_descent'] for r in records)
        assert all(r['fixed_error']==0 for r in records)
        mesh_path=Path(f'output/bunny_paper_assets/mesh_{level}.npz')
        assert hashlib.sha256(mesh_path.read_bytes()).hexdigest()==m['assets'][mesh_path.name]
        mesh=np.load(mesh_path);rest=mesh['vertices'];faces=mesh['faces'];t=rest[faces]
        np.testing.assert_array_equal(states[0],rest)
        D=np.stack([t[:,1]-t[:,0],t[:,2]-t[:,0]],axis=-1)
        B=np.linalg.inv(np.linalg.cholesky(D.transpose(0,2,1)@D).transpose(0,2,1))
        stretch=[]
        for x in states:
            t=x[faces];F=np.stack([t[:,1]-t[:,0],t[:,2]-t[:,0]],axis=-1)@B
            stretch.append(float(np.linalg.svd(F,compute_uv=False)[:,0].max()))
        assert max(stretch)<m['strain_limit']
        log=(root/f'{name}.log').read_text(encoding='utf-8',errors='replace')
        assert 'replace=false' in log and 'PSQ_FORCE_ACCEPT' not in log and 'PSQ_PREVIEW_STOP' not in log
        nr=np.array([r['native']['newton_iterations'] for r in records])
        trials=np.array([r['native']['line_search_trials'] for r in records])
        alphas=np.array([r['native']['last_line_search_alpha'] for r in records])
        per_step=[]
        for j,r in enumerate(records):
            per_step.append(dict(step=j+1,time=(j+1)*m['dt'],newton_iterations=int(nr[j]),
                line_search_trials=int(trials[j]),backtracks=int(trials[j]-nr[j]),
                final_alpha=float(alphas[j]),max_principal_stretch=stretch[j+1],seconds=r['seconds']))
        with (run/'resolution_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(per_step[0]));w.writeheader();w.writerows(per_step)
        fail_alpha=re.findall(r'alpha_last=([^,]+)',log)
        residual=re.findall(r'Residual/AbsTol: ([^/\s]+)/([^\s]+)',log)
        case=dict(name=name,level=level,vertices=len(rest),triangles=len(faces),status=m['status'],
            accepted_steps=len(records),physical_seconds=len(records)*m['dt'],failed_step=m.get('failed_step'),
            failure=m.get('error'),strict_steps=m['strictly_converged_steps'],preview_steps=m['preview_stopped_steps'],
            mean_newton=float(nr.mean()),max_newton=int(nr.max()),total_newton=int(nr.sum()),
            common_mean_newton=float(nr[:common].mean()),common_max_newton=int(nr[:common].max()),
            total_backtracks=int((trials-nr).sum()),common_backtracks=int((trials[:common]-nr[:common]).sum()),
            min_final_alpha=float(alphas.min()),common_min_final_alpha=float(alphas[:common].min()),
            last_accepted_alpha=float(alphas[-1]),failed_alpha_last=float(fail_alpha[-1]) if fail_alpha else None,
            last_logged_residual=[float(v) for v in residual[-1]] if residual else None,
            max_principal_stretch=max(stretch),common_max_principal_stretch=max(stretch[:common+1]),
            accepted_physics_seconds=m['physics_seconds'],process_wall_seconds=m['process_wall_seconds'],
            common_physics_seconds=sum(r['seconds'] for r in records[:common]),
            trajectory_sha256=hashlib.sha256((run/'accepted_positions.npy').read_bytes()).hexdigest())
        result['cases'].append(case)
        time=np.arange(1,len(records)+1)*m['dt'];label=f'{name}: {len(rest)} vertices'
        axes[0,0].plot(time,nr,label=label)
        axes[0,1].plot(time,trials-nr,label=label)
        axes[1,0].plot(time,alphas,label=label)
        axes[1,1].plot(np.arange(len(stretch))*m['dt'],stretch,label=label)
    for ax,title in zip(axes.flat,['Newton iterations per accepted step','Backtracks per accepted step','Final line-search alpha per accepted step','Maximum principal stretch']):
        ax.set_title(title);ax.set_xlabel('Physical time (s)');ax.grid(alpha=.3);ax.legend(fontsize=8)
        ax.axvline(common*metas[0]['dt'],ls=':',color='gray')
    axes[1,0].set_yscale('log')
    axes[1,1].axhline(metas[0]['strain_limit'],color='red',ls='--',label='strain limit')
    fig.suptitle('Independent native Bunny runs | identical strain barrier and solver settings\nAccepted frames only; dotted line = common comparison interval')
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    for path in ['scripts/run_psq_probe.py','scripts/probe_libuipc_bunny.py','scripts/pack_psq_native.py','scripts/report_bunny_resolution.py']:
        result['source_hashes'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for name,path in metas[0]['loaded_libraries'].items():
        result['library_hashes'][name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['cases'],indent=2))


if __name__=='__main__':main()
