"""Compare actual accepted level trajectories, with explicitly geometric metrics.

The PSQ positional discrepancy here is NOT PD++ Equation 12 continuity error.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from progressive.prolongation import ShellProlongation

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root;out=root/'comparison';out.mkdir(exist_ok=True)
    paths=sorted(root.glob('L[0-9]'),key=lambda q:int(q.name[1:]));window_steps=None
    if (root/'chain.json').exists():
        chain=json.loads((root/'chain.json').read_text());paths=[Path(r['run']) for r in chain['chain']]
        window_steps=chain['requested_steps']
        if 'failed_level' in chain:paths.append(root/f'L{chain["failed_level"]}')
    rows=[];tracks={};metas={};native_records={}
    for path in paths:
        if not (path/'run.json').exists():continue
        m=json.loads((path/'run.json').read_text())
        if m['status']=='running':continue
        k=m['level'];metas[k]=m;count=m['completed_steps'] if window_steps is None else min(m['completed_steps'],window_steps)
        x=np.load(path/'positions.npy',mmap_mode='r')[:count+1];tracks[k]=x
        step=json.loads((path/'steps.json').read_text())[:count];native_records[k]=step;speed=np.array([np.linalg.norm(x[i]-x[i-1],axis=1).max()/m['dt'] for i in range(1,len(x))])
        rows.append(dict(level=k,vertices=m['vertices'],faces=m['faces'],completed=m['completed'] if window_steps is None else count==window_steps,physical_seconds=count*m['dt'],
            source_status=m['status'],source_completed_steps=m['completed_steps'],
            native_steps=sum(bool(s['native']['converged']) for s in step),
            preview_steps=sum(bool(s['native'].get('preview_small_step') or s['native'].get('preview_non_descent')) for s in step),
            small_step_preview_steps=sum(bool(s['native'].get('preview_small_step')) for s in step),
            maximum_stretch=max((s['maximum_stretch'] for s in step),default=1.),zero_motion_steps=int(np.count_nonzero(speed<1e-10)),
            maximum_velocity_readback_error=max((s['velocity_readback_error'] for s in step),default=0.),
            maximum_position_injection_error=max((s['position_injection_error'] for s in step),default=0.),wall_seconds=m.get('wall_seconds')))
    if not rows:raise ValueError('No finished or failed trajectories to compare')
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True);gaps=[]
    for row in rows:
        k=row['level'];m=metas[k];x=tracks[k];t=np.arange(len(x))*m['dt']
        speed=np.r_[0,[np.sqrt(np.mean(np.sum((x[i]-x[i-1])**2,axis=1)))/m['dt'] for i in range(1,len(x))]]
        axes[0].plot(t,speed,label=f'L{k}: {row["vertices"]:,} vertices')
        if k-1 in tracks:
            assets=Path(m['assets']);low=np.load(assets/f'mesh_{k-1}.npz');fine=np.load(assets/f'mesh_{k}.npz')
            transfer=ShellProlongation(low['vertices'],fine['vertices'],low['faces'],load_npz(assets/f'map_{k-1}_{k}.npz'),np.load(assets/f'anchors_{k-1}_{k}.npy'))
            ids=np.unique(np.r_[np.arange(0,min(len(x),len(tracks[k-1])),5),min(len(x),len(tracks[k-1]))-1]).astype(int)
            error=np.array([np.sqrt(np.mean(np.sum((x[i]-transfer.position(tracks[k-1][i]))**2,axis=1))) for i in ids])
            gaps.append(dict(level=k,time_seconds=(ids*m['dt']).tolist(),position_gap_rms_m=error.tolist()))
            axes[1].plot(ids*m['dt'],error*1000,label=f'L{k} vs PSQ(L{k-1})')
    axes[0].set_ylabel('RMS vertex speed (m/s)');axes[1].set_ylabel('Geometric RMS gap (mm)');axes[1].set_xlabel('Physical time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'{metas[rows[0]["level"]]["scene"].upper()} | actual level dynamics; geometric diagnostics only')
    fig.savefig(out/'level_metrics.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7.5),constrained_layout=True)
    for row in rows:
        k=row['level'];meta=metas[k];records=native_records[k]
        axes[0].plot(np.arange(1,len(records)+1)*meta['dt'],
            [r['native']['newton_iterations'] for r in records],lw=.8,label=f'L{k}')
        rejected=meta.get('rejected_step',{})
        if rejected and (window_steps is None or rejected['step']<=window_steps):
            iters=rejected.get('native',{}).get('newton_iterations')
            if iters:
                axes[0].scatter([rejected['step']*meta['dt']],[iters],marker='x',s=70,c='#ae3838',zorder=5)
                axes[0].annotate(f'L{k} rejected',(rejected['step']*meta['dt'],iters),fontsize=8,
                    xytext=(-4,-13),textcoords='offset points',ha='right',color='#ae3838')
    axes[0].set_yscale('log');axes[0].set_ylabel('Newton iterations (log scale)');axes[0].set_xlabel('Physical time (s)')
    axes[0].legend(ncol=5,fontsize=8);axes[0].grid(alpha=.2)
    labels=[f'L{r["level"]}' for r in rows];newton=np.array([r['native_steps'] for r in rows]);small=np.array([r['small_step_preview_steps'] for r in rows]);other=np.array([r['preview_steps']-r['small_step_preview_steps'] for r in rows])
    axes[1].bar(labels,newton,label='Native Newton convergence',color='#287c74')
    axes[1].bar(labels,small,bottom=newton,label='Coarse small-step stop',color='#d69644')
    axes[1].bar(labels,other,bottom=newton+small,label='Verified non-descent stop',color='#8670a8')
    axes[1].set_ylabel('Accepted steps in the comparison window');axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Solver diagnostics | preview acceptance is separate from Newton convergence')
    fig.savefig(out/'solver_diagnostics.png',dpi=140);plt.close(fig)
    result=dict(levels=rows,adjacent_geometric_gaps=gaps,
        scope='Accepted trajectories only; position gaps and speeds are not PD++ Equation 12 and do not prove stability or temporal continuity',
        paper_scale_complete=len(rows)==(10 if m['scene']=='flag' else 8) and all(r['completed'] for r in rows))
    (out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
