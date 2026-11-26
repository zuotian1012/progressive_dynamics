"""Verify L1 results and compare material-point trajectories with their saved L0."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import cv2
import imageio.v2 as imageio
import ipctk
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from progressive.prolongation import ShellProlongation, frames
from bunny_runtime import check_run_sources, runtime_provenance
from report_bunny_ground_fine import rasterize

BASE=ROOT/'output/bunny_pcg_no_ground_20261124'
OUT=BASE/'L1'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_audited(run):
    m=json.loads((run/'run.json').read_text())
    assert m['completed'] and m['steps']==m['completed_steps']==100 and m['dt']==.01
    assert not m['ground']['enabled'] and m['bunny_linear_solver']=='native_pcg'
    assert m['config']['linear_system']['solver']=='fused_pcg'
    assert m['frozen_offset_only'] and not m['exact_offset_gradient']
    check_run_sources(run,m['source_files_sha256'])
    assert runtime_provenance(m['native_package'])['dll_sha256']==m['library_sha256']['uipc_backend_cuda.dll']
    assert json.loads((run/'audit.json').read_text())['accepted_prefix_checks_passed']
    assert json.loads((run/'report/non_descent_audit.json').read_text())['all_accepted_preview_evidence_verified']
    assert json.loads((run/'line_search_audit.json').read_text())['passed']
    mesh=np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz')
    bunny=np.load(Path(m['assets'])/'bunny.npz')
    x=np.load(run/'positions.npy')
    assert x.shape==(101,len(mesh['vertices']),3) and np.isfinite(x).all()
    log=run.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
    assert 'BUNNY_DIRECT_SOLVE' not in log and 'LinearFusedPCG:' in log
    planes=re.findall(r'VertexHalfPlaneTrajectoryFilter PHs: (\d+)',log)
    assert planes and all(int(n)==0 for n in planes)
    if m['level']:
        rest=np.vstack([mesh['vertices'],bunny['vertices']])
        faces=np.vstack([mesh['faces'],bunny['faces']+len(mesh['vertices'])]).astype(np.int32)
        edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
        cm=ipctk.CollisionMesh(rest,edges,faces)
        intersections=[i for i,state in enumerate(x) if ipctk.has_intersections(cm,np.vstack([state,bunny['vertices']]))]
        assert not intersections,(run,intersections)
        rows=json.loads((run/'steps.json').read_text())
        assert max(r['obstacle_error'] for r in rows)==0
        assert max(r['velocity_readback_error'] for r in rows)<=1e-12
        assert max(r['position_injection_error'] for r in rows)<=1e-12
        audit=dict(no_ground=True,passed=True,saved_states_checked=101,intersecting_states=[],
                   positions_sha256=digest(run/'positions.npy'),newton_converged=m['native_steps'],
                   preview_stops=m['preview_steps'],maximum_stretch=max(r['maximum_stretch'] for r in rows),
                   fixed_bunny_error_m=0,velocity_injection_verified=True,pcg_confirmed=True,
                   limitation='Saved-state geometry checks, not an independent continuous-time clearance certificate')
        (run/'no_ground_audit.json').write_text(json.dumps(audit,indent=2))
    else:
        audit=json.loads((run/'no_ground_audit.json').read_text())
        assert audit['positions_sha256']==digest(run/'positions.npy') and not audit['intersecting_states']
    return m,x,mesh,bunny

def main():
    report=OUT/'report';report.mkdir(exist_ok=True)
    panels=[];results=[];series=[];centroids={}
    for row,label in enumerate(['mu0','mu03']):
        parent=BASE/label
        pm,px,coarse,bunny=read_audited(parent)
        assets=Path(pm['assets']);fine=np.load(assets/'mesh_1.npz')
        U=load_npz(assets/'map_0_1.npz')
        transfer=ShellProlongation(coarse['vertices'],fine['vertices'],coarse['faces'],U,np.load(assets/'anchors_0_1.npy'))
        target=np.array([transfer.position(state) for state in px])
        areas=frames(fine['vertices'],fine['faces'])[3]
        weights=np.bincount(fine['faces'].ravel(),weights=np.repeat(areas/3,3),minlength=len(fine['vertices']))/areas.sum()
        np.testing.assert_allclose(target[0],fine['vertices'],atol=1e-10,rtol=0)
        mu=pm['contact_friction']['cloth_bunny']
        bt=bunny['vertices'][bunny['faces']]
        panels.append((f'mu={mu:g} | L0',px,coarse['faces'],bt,None))
        centroids[label+'_L0']=np.einsum('tvi,v->ti',target,weights)
        for method in ['semi','velpro']:
            run=OUT/f'{label}_{method}'
            m,x,mesh,_=read_audited(run)
            assert Path(m['lower_run']).resolve()==parent.resolve() and m['lower_sha256']==digest(parent/'positions.npy')
            assert m['map_sha256']==digest(assets/'map_0_1.npz')
            assert m['config']==pm['config'] and m['ground']==pm['ground'] and m['material']==pm['material']
            assert m['contact_friction']==pm['contact_friction'] and m['library_sha256']==pm['library_sha256']
            assert m['method']==method and m['consistency_weight']==0
            delta=x-target
            rms=1000*np.sqrt(np.einsum('tvi,tvi,v->t',delta,delta,weights))
            maximum=1000*np.linalg.norm(delta,axis=2).max(axis=1)
            centroid=1000*np.linalg.norm(np.einsum('tvi,v->ti',delta,weights),axis=1)
            centroids[label+'_'+method]=np.einsum('tvi,v->ti',x,weights)
            for frame in range(101):
                series.append(dict(friction=mu,method=method,frame=frame,time_s=frame*.01,rms_mm=rms[frame],
                                   max_vertex_mm=maximum[frame],centroid_delta_mm=centroid[frame]))
            result=dict(friction=mu,method=method,mean_rms_mm=float(rms[1:].mean()),peak_rms_mm=float(rms.max()),
                        peak_time_s=float(rms.argmax()*.01),final_rms_mm=float(rms[-1]),
                        final_centroid_delta_mm=float(centroid[-1]),peak_vertex_difference_mm=float(maximum.max()),
                        newton_converged=m['native_steps'],preview_stops=m['preview_steps'],wall_seconds=m['wall_seconds'])
            results.append(result)
            panels.append((f'mu={mu:g} | L1 {method.title()}',x,mesh['faces'],bt,rms))
            print(result,flush=True)
    with (report/'trajectory_errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=series[0].keys());w.writeheader();w.writerows(series)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,mu in zip(axes,[0.,.3]):
        for method,color in [('semi','#1676a5'),('velpro','#d66a21')]:
            values=[r for r in series if r['friction']==mu and r['method']==method]
            ax.plot([r['time_s'] for r in values],[r['rms_mm'] for r in values],label=method.title(),color=color)
        ax.set(title=f'Friction = {mu:g}',xlabel='Time (s)',ylabel='L1 vs prolonged L0: area-weighted RMS (mm)')
        ax.grid(alpha=.25);ax.legend()
    fig.savefig(report/'trajectory_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,6),layout='constrained')
    for ax,label in zip(axes,['mu0','mu03']):
        for key,style in [('L0','k--'),('semi','-'),('velpro','-')]:
            c=centroids[label+'_'+key];ax.plot(c[:,0],c[:,2],style,label=key)
            ax.scatter(c[-1,0],c[-1,2],s=20)
        ax.set(title=label,xlabel='Area-weighted center x (m)',ylabel='Area-weighted center z (m)',aspect='equal')
        ax.grid(alpha=.25);ax.legend()
    fig.savefig(report/'centroid_paths.png',dpi=180);plt.close(fig)
    render(report,panels)
    summary=dict(results=results,metric='sqrt(sum_i normalized_rest_area_L1[i] * ||x_L1[i]-P_0_to_1(x_L0)[i]||^2)',
                 alignment='Same physical timestamps and world coordinates; no rigid alignment or time shifting',
                 comparison='Parent consistency, not error against physical ground truth',
                 mean_excludes_initial_state=True,physical_seconds=1.,playback='4x slow',
                 all_606_saved_states_audited=True)
    (report/'summary.json').write_text(json.dumps(summary,indent=2))

def render(report,panels):
    az,el=np.deg2rad([-55,28]);cam=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0]);basis=np.stack([right,np.cross(cam,right),cam],axis=1)
    width,height=480,400;rowh=480
    gifs=[];single=[[],[]]
    writers=[imageio.get_writer(report/name,fps=25,codec='libx264',macro_block_size=1)
             for name in ['comparison.mp4','mu0_comparison.mp4','mu03_comparison.mp4']]
    try:
        for frame in range(101):
            canvas=np.full((rowh*2,width*3,3),246,np.uint8)
            for row in range(2):
                group=panels[row*3:row*3+3]
                bounds=np.vstack([arr.reshape(-1,3)@basis[:,:2] for _,x,_,bt,_ in group for arr in [x[frame],bt]])
                low,high=bounds.min(0),bounds.max(0);center=(low+high)/2
                span=np.array([width,height])*max((high-low)/[width,height])*1.09
                for col,(title,x,faces,bt,rms) in enumerate(group):
                    tri=np.vstack([x[frame][faces],bt]);norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                    norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-30)
                    shade=.55+.45*np.abs(norm@np.array([.3,-.7,.64]))
                    colors=np.vstack([np.tile([.11,.45,.62],(len(faces),1)),np.tile([.79,.76,.70],(len(bt),1))])
                    rgba=rasterize(tri@basis,colors*shade[:,None],center-span/2,center+span/2,width,height,
                                   np.ascontiguousarray(x[0][faces][:,:,:2]),shade[:len(faces),None]*np.array([.14,.63,.74]))[::-1]
                    top,left=row*rowh,col*width
                    canvas[top+42:top+42+height,left:left+width]=np.where(rgba[:,:,3:]!=0,rgba[:,:,:3],246)
                    cv2.putText(canvas,f'{title} | {frame*.01:.2f}s',(left+12,top+27),cv2.FONT_HERSHEY_SIMPLEX,.6,(23,47,58),1,cv2.LINE_AA)
                    text='Reference | GPU PCG | No ground | 4x slow' if rms is None else f'RMS to P(L0): {rms[frame]:.3f} mm | 4x slow'
                    cv2.putText(canvas,text,(left+12,top+466),cv2.FONT_HERSHEY_SIMPLEX,.46,(50,65,75),1,cv2.LINE_AA)
            writers[0].append_data(canvas)
            for row in range(2):writers[row+1].append_data(canvas[row*rowh:(row+1)*rowh])
            if frame%2==0:
                gifs.append(cv2.resize(canvas,(1080,720),interpolation=cv2.INTER_AREA))
                for row in range(2):single[row].append(cv2.resize(canvas[row*rowh:(row+1)*rowh],(1200,400),interpolation=cv2.INTER_AREA))
            if frame in [20,40,60,80,100]:imageio.imwrite(report/f'frame_{frame:03d}.png',canvas)
    finally:
        for writer in writers:writer.close()
    imageio.mimsave(report/'comparison.gif',gifs,duration=80,loop=0)
    for row,name in enumerate(['mu0','mu03']):imageio.mimsave(report/f'{name}_comparison.gif',single[row],duration=80,loop=0)
    for name in ['comparison.mp4','mu0_comparison.mp4','mu03_comparison.mp4']:
        cap=cv2.VideoCapture(str(report/name));assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==101
        count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==101
    print('Rendered',report,flush=True)

if __name__=='__main__':main()
