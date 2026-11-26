"""Scientific plots from saved trajectories; never synthesizes simulation states."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


def report(root,all_levels=False,accepted_prefix=False,compare_with=None):
    root=Path(root)
    cfg=json.loads((root/'experiment.json').read_text())
    if not cfg['completed'] and not accepted_prefix: raise ValueError('Use --accepted-prefix to inspect an incomplete experiment')
    available=[dict(r,source_root=root) for r in cfg['runs']]
    if compare_with:
        for other_root in map(Path,compare_with):
            other=json.loads((other_root/'experiment.json').read_text())
            ignored={'output','config','methods','resume_from','resume_interrupted','checkpoint_every','threads'}
            if {k:v for k,v in cfg['parameters'].items() if k not in ignored}!={k:v for k,v in other['parameters'].items() if k not in ignored}:
                raise ValueError('Comparison requires matching physical and solver parameters')
            if cfg.get('asset_files_sha256')!=other.get('asset_files_sha256') or cfg['mesh_hashes']!=other['mesh_hashes']:
                raise ValueError('Comparison geometry/mapping mismatch')
            for kernel in ('affine.py','contact.py','integrators.py','prolongation.py','shell.py','solver.py','differentials.py'):
                if cfg['code_sha256'][kernel]!=other['code_sha256'][kernel]:raise ValueError(f'Comparison kernel mismatch: {kernel}')
            if accepted_prefix:raise ValueError('Cross-experiment comparison currently requires completed runs')
            if not other['completed']:raise ValueError('Comparison experiment is incomplete')
            available.extend(dict(r,source_root=other_root) for r in other['runs'])
    if accepted_prefix:
        available=[dict(json.loads(p.read_text()),folder=p.parent.name) for p in sorted(root.glob('*_L*/run.json'))]
    level=cfg['parameters']['levels']-1
    runs=[]
    for r in available:
        if not all_levels and r['level']!=level:continue
        # Shared coarse states remain visible for each progressive method.
        if 'reused_from' in r:
            source=next(s for s in available if s.get('folder')==r['reused_from'] and s.get('source_root')==r.get('source_root'))
            r=dict(source,method=r['method'])
        runs.append(r)
    if not runs:raise ValueError('No saved levels selected; use --all-levels for coarse-only checkpoints')
    frames=min(r['completed_steps'] for r in runs)
    if frames<1:raise ValueError('No accepted steps available')
    records=[];summary=[]
    for r in runs:
        folder=r.get('source_root',root)/r['folder']
        track=np.load(folder/'positions.npy',mmap_mode='r')[:frames+1]
        faces=np.load(folder/'faces.npy')
        metrics=json.loads((folder/'metrics.json').read_text())[:frames]
        poses=np.load(folder/'robot.npy',mmap_mode='r')[:frames+1]
        if len(metrics)!=frames or len(track)!=frames+1 or len(poses)!=frames+1:
            raise ValueError('Incomplete accepted checkpoint')
        colliders=[]
        for p in sorted(folder.glob('collider_*.npz')):
            with np.load(p) as d:colliders.append((d['vertices'],d['faces']))
        for p in sorted(folder.glob('fixed_collider_*.npz')):
            with np.load(p) as d:colliders.append((d['vertices'],d['faces']))
            # Rendering-only identity pose; fixed geometry is already world-space.
            poses=np.concatenate((poses,np.broadcast_to(np.eye(4),(frames+1,1,4,4))),axis=1)
        records.append((r,track,faces,metrics,poses,colliders))
        summary.append({'method':r['method'],'level':r['level'],'source_folder':str(folder.resolve()),'physics_seconds':r['physics_seconds'],
                        'vertices':r['vertices'],'faces':r['faces'],'displayed_steps':frames,
                        'experiment_completed':cfg['completed'],
                        'preview_stopped_steps':r.get('preview_stopped_steps',0),
                        'mean_consistency_rms_m':float(np.mean([m['coarse_rms_m'] for m in metrics])) if r['level'] else None,
                        'max_stretch':max(m['max_stretch'] for m in metrics),
                        'max_continuity_rms_m':max((m.get('midpoint_rms_m',0) for m in metrics),default=0)})
    out=root/('report_comparison' if compare_with else ('report_prefix' if accepted_prefix else ('report_all_levels' if all_levels else 'report')));out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for r,x,f,metrics,poses,c in records:
        t=np.arange(1,len(metrics)+1)*cfg['dt']
        label=f'{r["method"]} L{r["level"]}'
        if r['level']:axes[0].plot(t,[m['coarse_rms_m']*1000 for m in metrics],label=label)
        axes[1].plot(t,[m['max_stretch'] for m in metrics],label=label)
        axes[2].plot(t,[m.get('midpoint_rms_m',np.nan)*1000 for m in metrics],label=label)
    for ax,title in zip(axes,['Coarse-space consistency RMS (mm)','Maximum principal stretch','Force-corrected continuity RMS (mm)']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.25)
        if ax.lines:ax.legend()
    axes[1].axhline(cfg['parameters']['strain_limit'],ls='--',c='red',alpha=.6)
    fig.savefig(out/'metrics.png',dpi=150);plt.close(fig)
    # Common fixed camera and axes across methods and frames.
    bounds=[np.asarray(x)[::max(1,len(x)//20)].reshape(-1,3) for _,x,*_ in records]
    if cfg['parameters']['scene']=='bunny':
        bounds.extend(v for v,_ in records[0][-1])
    points=np.concatenate(bounds)
    lo,hi=points.min(axis=0),points.max(axis=0)
    center=(lo+hi)/2;radius=max((hi-lo).max()*.6,.04)
    images=[]
    for frame in np.unique(np.linspace(0,frames,min(61,frames+1)).astype(int)):
        fig=plt.figure(figsize=(max(6,4*len(records)),4.5))
        for i,(r,x,f,metrics,poses,colliders) in enumerate(records):
            ax=fig.add_subplot(1,len(records),i+1,projection='3d')
            color={'semi':'#cb8074','full':'#9fbd53','velpro':'#74b7ee','direct':'#c4b4d5'}[r['method']]
            ax.add_collection3d(Poly3DCollection(x[frame][f],facecolor=color,edgecolor='#42515b',linewidth=.2,alpha=.96))
            for j,(v,cf) in enumerate(colliders):
                p=v @ poses[frame,j,:3,:3].T+poses[frame,j,:3,3]
                bunny=cfg['parameters']['scene']=='bunny'
                color='#c99b66'
                if bunny:
                    triangle=p[cf]
                    normal=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
                    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-20)
                    light=np.array([-.5,-.7,1.]);light/=np.linalg.norm(light)
                    color=np.clip((.5+.4*np.maximum(normal @ light,0))[:,None]*np.array([.8,.82,.85]),0,1)
                ax.add_collection3d(Poly3DCollection(p[cf],facecolor=color,
                                                    edgecolor='none' if bunny else '#695039',linewidth=.2,alpha=1 if bunny else .35))
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
            ax.view_init(elev=15,azim=-65);ax.set_box_aspect((1,1,1),zoom=1.2);ax.set_axis_off()
            status=f'\nPreview stops: {r["preview_stopped_steps"]}' if r.get('preview_stopped_steps',0) else ''
            ax.set_title(f'{r["method"]} | L{r["level"]} | {r["vertices"]} vertices\n{r["physics_seconds"]:.2f}s compute'+status,fontsize=11)
        status='\nINCOMPLETE: accepted states only' if not cfg['completed'] else ''
        fig.suptitle(f'{cfg["parameters"]["scene"]}: physical time {frame*cfg["dt"]:.3f}s'+status,fontsize=12,y=.99)
        fig.subplots_adjust(top=.78,left=.01,right=.99,bottom=.01)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        plt.close(fig)
    images[-1].save(out/'final.png')
    images[0].save(out/'comparison.gif',save_all=True,append_images=images[1:],duration=max(30,int(frames*cfg['dt']*1000/len(images))),loop=0)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--all-levels',action='store_true',help='Show coarse-to-fine trajectories side by side')
    p.add_argument('--accepted-prefix',action='store_true',help='Inspect only saved accepted states of an incomplete run; explicitly labeled')
    p.add_argument('--compare-with',nargs='+',type=Path,help='Additional completed experiments with identical inputs and solver settings')
    args=p.parse_args();report(args.root,args.all_levels,args.accepted_prefix,args.compare_with)
