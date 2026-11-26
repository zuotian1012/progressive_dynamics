"""Render saved PSQ states with explicit forced-update labels; no simulation."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import load_run,draw,plt,Image
from progressive.prolongation import frames
from shirt_demo.assets import Robot,LINKS


def report(root):
    data=load_run(root);meta=data['meta'];n=meta['completed_steps']
    label=f'L{meta["level"]}: {meta["vertices"]:,} vertices / {meta["faces"]:,} faces'
    records={};current=0
    with root.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as stream:
        for line in stream:
            match=re.search(r'>>> Begin Frame: (\d+)',line)
            if match:current=int(match[1])
            match=re.search(r'Newton Iteration Converged with Iteration Count: (\d+)',line)
            if match:records[current]=dict(forced=False,iterations=int(match[1])+1)
            match=re.search(r'PSQ_FORCE_ACCEPT frame=(\d+) reason=(\w+) newton=(\d+)',line)
            if match:records[int(match[1])]=dict(forced=True,reason=match[2],iterations=int(match[3])+1)
    assert set(range(1,n+1))<=set(records),'Missing convergence/force evidence'
    records={k:records[k] for k in range(1,n+1)}
    x=data['x'][:n+1];f=data['f'];assert np.isfinite(x).all()
    inv=frames(x[0],f)[2];stretch=[]
    for v in x:
        tri=v[f];F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)@inv
        stretch.append(float(np.linalg.svd(F,compute_uv=False)[:,0].max()))
    forced=[k for k,r in records.items() if r['forced']]
    robot=Robot();tracking=[]
    settle=meta['parameters']['settle_steps'];substeps=meta['parameters']['substeps']
    for step in range(n):
        if step<settle:q=robot.q[0]
        else:
            frame,sub=divmod(step-settle,substeps)
            previous=robot.q[max(0,frame-1)]
            q=previous+(sub+1)/substeps*(robot.q[frame]-previous)
        target=robot.poses(q)
        tracking.append([float(np.linalg.norm(data['robot'][step+1,j,:3,3]-target[name][:3,3])) for j,name in enumerate(LINKS)])
    audit=dict(completed=meta['completed'],completed_steps=n,requested_steps=meta['steps'],
               physical_seconds=n*meta['dt'],physics_seconds=meta.get('physics_seconds'),
               forced_steps=forced,forced_count=len(forced),strictly_converged_count=n-len(forced),
               maximum_principal_stretch=max(stretch),strain_limit=meta['material_parameters']['strain_limit'],
               finite_states=True,independent_collision_audit=False,
               maximum_body_origin_tracking_error_m=float(np.max(tracking)) if tracking else None,
               final_body_origin_tracking_errors_m=tracking[-1] if tracking else [],
               scope=f'Exploratory physical L{meta["level"]} trajectory; forced updates are not Newton convergence',
               records=records)
    out=root/'report';out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    np.save(out/'maximum_stretch.npy',stretch)
    np.save(out/'body_origin_tracking_error.npy',tracking)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axes[0].plot(np.arange(n+1)*meta['dt'],stretch);axes[0].axhline(audit['strain_limit'],color='red',ls='--')
    axes[0].set_ylabel('Maximum stretch')
    if tracking:axes[1].plot(np.arange(1,n+1)*meta['dt'],np.asarray(tracking)*1000)
    axes[1].set(xlabel='Physical time (s)',ylabel='Gripper body origin error (mm)')
    fig.suptitle(f'Exploratory PSQ L{meta["level"]}: strain and actual/target gripper mismatch')
    fig.tight_layout();fig.savefig(out/'metrics.png',dpi=130);plt.close(fig)
    fig=plt.figure(figsize=(8,7));draw(fig.add_subplot(projection='3d'),data,n,
        f'{label} | t={n*meta["dt"]:.3f}s\nEXPLORATORY: {len(forced)}/{n} forced steps',True)
    fig.tight_layout();fig.savefig(out/'final.png',dpi=150);plt.close(fig)
    if n:
        images=[]
        for k in np.unique(np.linspace(0,n,min(150,n+1),dtype=int)):
            fig=plt.figure(figsize=(7,6),dpi=95)
            flag='INITIAL' if k==0 else ('FORCED / NOT CONVERGED' if records[k]['forced'] else 'NEWTON CONVERGED')
            draw(fig.add_subplot(projection='3d'),data,k,
                 f'{label} | t={k*meta["dt"]:.3f}s\n{flag}',True)
            fig.text(.5,.015,f'Exploratory run | forced steps so far: {sum(j<=k for j in forced)}',ha='center',color='#a43122')
            fig.tight_layout(rect=[0,.03,1,1]);fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        duration=max(40,int(1000*n*meta['dt']/len(images)))
        images[0].save(out/'motion.gif',save_all=True,append_images=images[1:],duration=duration,loop=0)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('records','forced_steps')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    report(p.parse_args().run)
