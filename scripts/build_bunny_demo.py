"""Build a portable local HTML demo from verified L0-L5 saved trajectories."""
from pathlib import Path
import csv, hashlib, json, shutil, sys
import numpy as np
import imageio.v2 as imageio
from scipy.sparse import csr_matrix, load_npz
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from progressive.prolongation import ShellProlongation, frames
from report_bunny_pcg_no_ground_l1 import read_audited, digest
from report_bunny_ground_fine import rasterize

BASE=ROOT/'output/bunny_pcg_no_ground_20261124'
SITE=BASE/'demo'
LEVELS=range(1,6)

def weights(mesh):
    areas=frames(mesh['vertices'],mesh['faces'])[3]
    return np.bincount(mesh['faces'].ravel(),weights=np.repeat(areas/3,3),minlength=len(mesh['vertices']))/areas.sum()

def stats(delta,w):
    return dict(rms=(1000*np.sqrt(np.einsum('vi,vi,v->',delta,delta,w))).item(),
                maximum=float(1000*np.linalg.norm(delta,axis=1).max()),
                centroid=float(1000*np.linalg.norm(w@delta)))

def main():
    SITE.mkdir(exist_ok=True);(SITE/'frames').mkdir(exist_ok=True);(SITE/'media').mkdir(exist_ok=True)
    data=dict(dt=.01,frames=101,seconds=1,groups={},runs={},generated='2026-11-25',
              metrics={'parent':'当前层与 PSQ 延拓后的直接父层，按当前层静止面积加权。',
                       'l0':'各层分别用其实际弹性代理算子映射至共同 L7（90,000 顶点），与 L0 映射结果比较；按 L7 静止面积加权。'})
    csvrows=[];manifest={}
    for label,mu in [('mu0',0.),('mu03',.3)]:
        runs={}
        for key in ['L0']+[f'L{level}_{method}' for method in ['semi','velpro'] for level in LEVELS]:
            level=int(key[1]);method=key[3:] if level else None
            path=BASE/label if not level else BASE/f'L{level}'/f'{label}_{method}'
            m,x,mesh,bunny=read_audited(path)
            if level:
                parentpath=BASE/label if level==1 else BASE/f'L{level-1}'/f'{label}_{method}'
                parent=json.loads((parentpath/'run.json').read_text())
                assert Path(m['lower_run']).resolve()==parentpath.resolve() and m['lower_sha256']==digest(parentpath/'positions.npy')
                assert m['config']==parent['config'] and m['material']==parent['material'] and m['ground']==parent['ground']
                assert m['contact_friction']==parent['contact_friction'] and m['library_sha256']==parent['library_sha256']
                assert m['method']==method and m['consistency_weight']==0
            rows=json.loads((path/'steps.json').read_text())
            with np.load(path/'operator/operator.npz') as op:
                proxy=ShellProlongation(mesh['vertices'],op['fine_rest'],mesh['faces'],
                    csr_matrix((op['U_data'],op['U_indices'],op['U_indptr']),shape=tuple(op['U_shape'])),op['anchors'])
                fw=weights(dict(vertices=op['fine_rest'],faces=op['fine_faces']))
            runid=f'{label}_{key}'
            data['runs'][runid]=dict(level=level,method=method or 'shared',friction=mu,vertices=m['vertices'],faces=m['faces'],
                converged=m['native_steps'],preview=m['preview_steps'],wall=m['wall_seconds'],
                maxStretch=max(r['maximum_stretch'] for r in rows),
                termination=['initial']+[r['native']['termination_reason'] for r in rows],
                source=str(path.relative_to(BASE)).replace('\\','/'),sha256=digest(path/'positions.npy'))
            manifest[str(path/'positions.npy')]=data['runs'][runid]['sha256']
            runs[key]=dict(meta=m,x=x,mesh=mesh,proxy=proxy,weights=weights(mesh))
            print('Verified',runid,flush=True)
        targets={}
        for method in ['semi','velpro']:
            group=dict(friction=mu,method=method,runIds=[f'{label}_L0']+[f'{label}_L{level}_{method}' for level in LEVELS],metrics={})
            for level in LEVELS:
                child=runs[f'L{level}_{method}'];parent=runs['L0' if level==1 else f'L{level-1}_{method}']
                assets=Path(child['meta']['assets'])
                transfer=ShellProlongation(parent['mesh']['vertices'],child['mesh']['vertices'],parent['mesh']['faces'],
                    load_npz(assets/f'map_{level-1}_{level}.npz'),np.load(assets/f'anchors_{level-1}_{level}.npy'))
                targets[(method,level)]=transfer
                group['metrics'][str(level)]={'parent':[],'l0':[]}
            data['groups'][f'{label}_{method}']=group
        for frame in range(101):
            ref=runs['L0']['proxy'].position(runs['L0']['x'][frame])
            for method in ['semi','velpro']:
                group=data['groups'][f'{label}_{method}']
                for level in LEVELS:
                    child=runs[f'L{level}_{method}'];parent=runs['L0' if level==1 else f'L{level-1}_{method}']
                    metrics=group['metrics'][str(level)]
                    metrics['parent'].append(stats(child['x'][frame]-targets[(method,level)].position(parent['x'][frame]),child['weights']))
                    metrics['l0'].append(stats(child['proxy'].position(child['x'][frame])-ref,fw))
                    for basis in ['parent','l0']:
                        csvrows.append(dict(friction=mu,method=method,level=level,frame=frame,time_s=frame*.01,basis=basis,**metrics[basis][-1]))
        render(label,runs,bunny)
    for group in data['groups'].values():
        for metrics in group['metrics'].values():
            for basis in ['parent','l0']:
                values=np.array([r['rms'] for r in metrics[basis]])
                metrics[basis+'Summary']=dict(mean=float(values[1:].mean()),peak=float(values.max()),final=float(values[-1]),
                    maxVertex=max(r['maximum'] for r in metrics[basis]))
    (SITE/'data.js').write_text('window.BUNNY_DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';',encoding='utf-8')
    (SITE/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (SITE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (SITE/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=csvrows[0].keys());w.writeheader();w.writerows(csvrows)
    for source,name in [(BASE/'report/comparison.mp4','L0_friction.mp4'),
                        (BASE/'L1/report/comparison.mp4','L1_comparison.mp4')]:
        shutil.copy2(source,SITE/'media'/name)
    print('Built demo assets:',SITE,flush=True)

def render(label,runs,bunny):
    az,el=np.deg2rad([-55,28]);cam=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0]);basis=np.stack([right,np.cross(cam,right),cam],axis=1)
    width,height=480,420;bt=bunny['vertices'][bunny['faces']]
    for method in ['semi','velpro']:(SITE/'frames'/f'{label}_{method}').mkdir(exist_ok=True)
    writers={method:imageio.get_writer(SITE/'media'/f'{label}_{method}.mp4',fps=25,codec='libx264',macro_block_size=1)
             for method in ['semi','velpro']}
    try:
        for frame in range(101):
            bounds=np.vstack([bt.reshape(-1,3)@basis[:,:2]]+[r['x'][frame]@basis[:,:2] for r in runs.values()])
            low,high=bounds.min(0),bounds.max(0);center=(low+high)/2
            span=np.array([width,height])*max((high-low)/[width,height])*1.1
            images={}
            for key,r in runs.items():
                x,faces=r['x'],r['mesh']['faces'];tri=np.vstack([x[frame][faces],bt])
                n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
                shade=.55+.45*np.abs(n@np.array([.3,-.7,.64]))
                colors=np.vstack([np.tile([.11,.45,.62],(len(faces),1)),np.tile([.79,.76,.70],(len(bt),1))])
                rgba=rasterize(tri@basis,colors*shade[:,None],center-span/2,center+span/2,width,height,
                    np.ascontiguousarray(x[0][faces][:,:,:2]),shade[:len(faces),None]*np.array([.14,.63,.74]))[::-1]
                images[key]=np.where(rgba[:,:,3:]!=0,rgba[:,:,:3],246).astype(np.uint8)
            for method in ['semi','velpro']:
                canvas=np.hstack([images['L0']]+[images[f'L{level}_{method}'] for level in LEVELS])
                imageio.imwrite(SITE/'frames'/f'{label}_{method}'/f'{frame:03d}.jpg',canvas,quality=90)
                # Standalone export has permanent labels; HTML supplies its own captions.
                import cv2
                video=np.full((height+40,width*(len(LEVELS)+1),3),246,np.uint8);video[40:]=canvas
                for i in range(len(LEVELS)+1):cv2.putText(video,f'{label} | {method} | L{i} | {frame*.01:.2f}s | 4x slow',
                    (i*480+12,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,50,65),1,cv2.LINE_AA)
                writers[method].append_data(video)
            if frame%25==0:print('Render',label,frame,flush=True)
    finally:
        for writer in writers.values():writer.close()

if __name__=='__main__':main()
