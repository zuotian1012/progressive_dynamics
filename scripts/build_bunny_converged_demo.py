"""Audit and render the 75-step converged hierarchy against matched original prefixes."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import imageio.v2 as imageio
from scipy.sparse import csr_matrix, load_npz
from run_bunny_velpro_chain import validate, EVENT_RUNTIME, digest
from bunny_runtime import runtime_provenance, check_run_sources
from run_bunny_preview_pilot import audit, ROOT, ASSETS
from report_bunny_event_fallback_pilot import event_audit
from report_bunny_preview_pilot import weights, rms, summary, ShellProlongation
from report_bunny_ground_fine import rasterize

BASE = ROOT/'output/bunny_newton_converged_mu03_velpro_75_20261125'
OLD = ROOT/'output/bunny_pcg_no_ground_20261124'
SITE = BASE/'demo'


def metric(delta, w):
    return dict(rms=rms(delta,w).tolist(), maximum=(1000*np.linalg.norm(delta,axis=2).max(axis=1)).tolist(),
                centroid=(1000*np.linalg.norm(np.einsum('v,tvi->ti',w,delta),axis=1)).tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-render',action='store_true');parser.add_argument('--reuse-audits',action='store_true');a=parser.parse_args()
    SITE.mkdir(exist_ok=True)
    datasets={}; report=dict(frames=76,dt=.01,seconds=.75,friction=.3,generated='2026-11-26',runs={},metrics={},changes={},comparability={},audit={})
    if (SITE/'summary.json').exists():
        previous=json.loads((SITE/'summary.json').read_text(encoding='utf-8'))
        if 'continuation' in previous:report['continuation']=previous['continuation']
    manifest={}; dll=runtime_provenance(EVENT_RUNTIME)['dll_sha256']
    for mode in ('improved','original'):
        datasets[mode]={};report['runs'][mode]={};report['metrics'][mode]={}
        for level in range(7 if mode=='improved' else 6):
            run=BASE/f'L{level}' if mode=='improved' else (OLD/'mu03' if not level else OLD/f'L{level}'/'mu03_velpro')
            if mode=='improved':
                m=validate(run,level,SimpleNamespace(steps=75,friction=.3,event_fallback=True,output=BASE),dll)
                if not a.reuse_audits:audit(run,75)
            else:
                m=json.loads((run/'run.json').read_text());assert m['completed'] and m['completed_steps']==100
                assert m['frozen_offset_only'] and not m.get('event_fallback_enabled',False)
                check_run_sources(run,m['source_files_sha256'])
            saved=json.loads((run/'no_ground_audit.json').read_text())
            if mode=='original' and level==0:
                assert saved['no_ground'] and saved['pcg_confirmed'] and saved['fixed_bunny_error_m']==0
            else:assert saved['passed']
            assert not saved['intersecting_states'] and saved['saved_states_checked']==m['completed_steps']+1
            assert saved['positions_sha256']==digest(run/'positions.npy')
            assert json.loads((run/'audit.json').read_text())['accepted_prefix_checks_passed']
            assert json.loads((run/'line_search_audit.json').read_text())['passed']
            assert json.loads((run/'report/non_descent_audit.json').read_text())['all_accepted_preview_evidence_verified']
            rows=json.loads((run/'steps.json').read_text())[:75]
            events=event_audit(run,m,rows)['event_frames'] if mode=='improved' else {}
            if level:
                parent=datasets[mode][level-1]
                assert Path(m['lower_run']).resolve()==parent['path'].resolve()
                assert m['lower_sha256']==digest(parent['path']/'positions.npy')
                for field in ('config','material','ground','contact_friction','library_sha256'):
                    assert m[field]==parent['meta'][field],field
            mesh=np.load(ASSETS/f'mesh_{level}.npz');x=np.load(run/'positions.npy')[:76]
            with np.load(run/'operator/operator.npz') as op:
                proxy=ShellProlongation(mesh['vertices'],op['fine_rest'],mesh['faces'],
                    csr_matrix((op['U_data'],op['U_indices'],op['U_indptr']),shape=tuple(op['U_shape'])),op['anchors'])
                fw=weights(dict(vertices=op['fine_rest'],faces=op['fine_faces']))
            datasets[mode][level]=dict(path=run,meta=m,mesh=mesh,x=x,proxy=proxy,w=weights(mesh))
            report['runs'][mode][level]=dict(vertices=m['vertices'],faces=m['faces'],
                converged=sum(r['native']['converged'] for r in rows),preview=sum(r['native']['preview_non_descent'] for r in rows),
                events={int(k):v for k,v in events.items()},event_count=len(events),
                simulation_seconds=sum(r['seconds'] for r in rows),full_run_wall_seconds=m['wall_seconds'],
                newton_total=sum(r['native']['newton_iterations'] for r in rows),
                newton_max=max(r['native']['newton_iterations'] for r in rows),
                newton_by_frame=[0]+[r['native']['newton_iterations'] for r in rows],
                termination=['initial']+[r['native']['termination_reason'] for r in rows],
                max_stretch=max(r['maximum_stretch'] for r in rows),source=str(run),sha256=saved['positions_sha256'])
            manifest[str(run/'positions.npy')]=saved['positions_sha256']
            print('Verified',mode,level,flush=True)
    for level in range(6):
        old,new=datasets['original'][level],datasets['improved'][level]
        for field in ('config','material','ground','contact_friction','bundle_sha256'):
            assert old['meta'][field]==new['meta'][field],field
        np.testing.assert_array_equal(old['x'][0],new['x'][0])
        report['changes'][level]=metric(new['x']-old['x'],new['w'])
    csvrows=[]
    for mode,runs in datasets.items():
        for level in range(1,len(runs)):
            child,parent=runs[level],runs[level-1]
            transfer=ShellProlongation(parent['mesh']['vertices'],child['mesh']['vertices'],parent['mesh']['faces'],
                load_npz(ASSETS/f'map_{level-1}_{level}.npz'),np.load(ASSETS/f'anchors_{level-1}_{level}.npy'))
            delta=child['x']-np.array([transfer.position(x) for x in parent['x']])
            report['metrics'][mode][level]={'parent':metric(delta,child['w']),'l0':{'rms':[],'maximum':[],'centroid':[]}}
        for f in range(76):
            ref=runs[0]['proxy'].position(runs[0]['x'][f])
            for level in range(1,len(runs)):
                child=runs[level];delta=child['proxy'].position(child['x'][f])-ref
                v=metric(delta[None],fw)
                for k in v:report['metrics'][mode][level]['l0'][k].append(v[k][0])
        for level in range(1,len(runs)):
            for basis in ('parent','l0'):
                v=report['metrics'][mode][level][basis]
                v['summary']=summary(np.array(v['rms']))
                v['summary']['max_vertex_mm']=max(v['maximum'])
                for f in range(76):csvrows.append(dict(mode=mode,level=level,basis=basis,frame=f,time_s=f*.01,
                    rms_mm=v['rms'][f],max_vertex_mm=v['maximum'][f],centroid_mm=v['centroid'][f]))
    report['comparability']=dict(same_scene_material_solver_settings_and_proxy=True,original_prefix_steps=75,
        runtime_change='Same GPU PCG and assembly objects; event-enabled advance loop changes the preview exit behavior.',
        parent_reference='Each hierarchy uses its own parent and L0; no reference is physical ground truth.',
        timing='Simulation-step time for the same 75 steps. GPU contention was uncontrolled; wall times are not isolated benchmarks.',
        repeatability='One run per strategy; no repeated-run confidence interval.')
    report['audit']=dict(improved_steps_converged=525,improved_states_checked=532,intersections_found=0,
        event_frames_verified=sum(r['event_count'] for r in report['runs']['improved'].values()),
        strain_limit=1.068,all_hashes_and_parent_links_verified=True)
    (SITE/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (SITE/'data.js').write_text('window.BUNNY_DATA='+json.dumps(report,ensure_ascii=False,separators=(',',':'))+';',encoding='utf-8')
    (SITE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (SITE/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=csvrows[0].keys());writer.writeheader();writer.writerows(csvrows)
    for level in range(1,6):
        print('METRIC L'+str(level),json.dumps({mode:{basis:report['metrics'][mode][level][basis]['summary'] for basis in ('parent','l0')} for mode in datasets}),flush=True)
    print('METRIC L6',json.dumps({k:v['summary'] for k,v in report['metrics']['improved'][6].items()}),flush=True)
    if not a.skip_render:render(datasets)


def render(datasets):
    bunny=np.load(ASSETS/'bunny.npz');bt=bunny['vertices'][bunny['faces']]
    az,el=np.deg2rad([-55,28]);cam=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0]);basis=np.stack([right,np.cross(cam,right),cam],axis=1)
    bounds=[bt.reshape(-1,3)@basis[:,:2]]
    for runs in datasets.values():
        for r in runs.values():
            projected=r['x'].reshape(-1,3)@basis[:,:2];bounds.extend([projected.min(axis=0)[None],projected.max(axis=0)[None]])
    bounds=np.vstack(bounds);low,high=bounds.min(axis=0),bounds.max(axis=0);center=(low+high)/2
    width,height=480,420;span=np.array([width,height])*max((high-low)/[width,height])*1.10
    (SITE/'media').mkdir(exist_ok=True)
    for mode,runs in datasets.items():
        out=SITE/'frames'/mode;out.mkdir(parents=True,exist_ok=True)
        writer=imageio.get_writer(SITE/'media'/f'{mode}.mp4',fps=25,codec='libx264',macro_block_size=1)
        try:
            for frame in range(76):
                images=[]
                for level,r in runs.items():
                    faces=r['mesh']['faces'];tri=np.vstack([r['x'][frame][faces],bt]);n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                    n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30);shade=.55+.45*np.abs(n@np.array([.3,-.7,.64]))
                    colors=np.vstack([np.tile([.11,.45,.62],(len(faces),1)),np.tile([.79,.76,.70],(len(bt),1))])
                    rgba=rasterize(tri@basis,colors*shade[:,None],center-span/2,center+span/2,width,height,
                        np.ascontiguousarray(r['x'][0][faces][:,:,:2]),shade[:len(faces),None]*np.array([.14,.63,.74]))[::-1]
                    images.append(np.where(rgba[:,:,3:]!=0,rgba[:,:,:3],246).astype(np.uint8))
                canvas=np.hstack(images);imageio.imwrite(out/f'{frame:03d}.jpg',canvas,quality=91)
                import cv2
                video=np.full((height+40,width*len(runs),3),246,np.uint8);video[40:]=canvas
                for level in runs:cv2.putText(video,f'{mode} | L{level} | {frame*.01:.2f}s | 4x slow',(level*width+10,26),cv2.FONT_HERSHEY_SIMPLEX,.49,(25,50,65),1,cv2.LINE_AA)
                writer.append_data(video)
                if frame%15==0:print('Rendered',mode,frame,flush=True)
        finally:writer.close()
    print('Built',SITE,flush=True)


if __name__=='__main__':main()
