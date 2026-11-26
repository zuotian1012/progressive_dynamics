"""Render eight verified bunny levels with one camera, scale, and clock."""
from pathlib import Path
import argparse,json,contextlib,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import cv2
from report_bunny_ground_fine import rasterize

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    chain=json.loads((a.root/'chain.json').read_text());assert chain['status']=='completed'
    ancestors=[Path(chain['seed'])]
    parent=json.loads((ancestors[-1]/'run.json').read_text())
    while parent['level']>0:
        path=Path(parent['lower_run'])
        with (path/'positions.npy').open('rb') as source:
            assert hashlib.file_digest(source,'sha256').hexdigest()==parent['lower_sha256']
        ancestors.append(path)
        parent=json.loads((path/'run.json').read_text())
    runs=list(reversed(ancestors))+[Path(v['run']) for v in chain['levels']]
    assert len(runs)==8
    out=a.root/'report';out.mkdir(exist_ok=True)
    font=lambda s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s)
    bg='#f5f7fa';ink='#163144';muted='#596f80'
    az,el=np.deg2rad([-55,28]);camera=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0]);up=np.cross(camera,right);basis=np.stack([right,up,camera],axis=1)
    data=[];xylo=np.full(2,np.inf);xyhi=-xylo;lo=np.full(2,np.inf);hi=-lo;summary=[]
    for k,run in enumerate(runs):
        m=json.loads((run/'run.json').read_text());audit=json.loads((run/'audit.json').read_text());ground=json.loads((run/'ground_audit.json').read_text());validation=json.loads((run/'continuation_audit.json').read_text())
        assert m['level']==k and m['completed_steps']==400 and m['completed'] and m['dt']==.01 and validation['passed'] and not audit['zero_motion_steps']
        mesh=np.load(Path(m['assets'])/f'mesh_{k}.npz');f=mesh['faces'];rest=mesh['vertices'];x=np.load(run/'positions.npy',mmap_mode='r')
        b=np.load(Path(m['assets'])/'bunny.npz');bv=b['vertices'];bf=b['faces']
        centers=rest[f].mean(1);checker=(np.floor(centers[:,0]/.08)+np.floor(centers[:,1]/.08)).astype(int)%2
        colors=np.vstack([np.where(checker[:,None]==0,[.10,.54,.73],[.25,.73,.82]),np.tile([.77,.75,.70],(len(bf),1))])
        for state in x:
            xylo=np.minimum(xylo,state[:,:2].min(0));xyhi=np.maximum(xyhi,state[:,:2].max(0))
            proj=state@basis;lo=np.minimum(lo,proj[:,:2].min(0));hi=np.maximum(hi,proj[:,:2].max(0))
        data.append(dict(m=m,x=x,f=f,colors=colors,run=run))
        summary.append(dict(level=k,method=m['method'],run=str(run.resolve()),vertices=m['vertices'],faces=m['faces'],wall_seconds=m['wall_seconds'],native_steps=m['native_steps'],preview_steps=m['preview_steps'],maximum_stretch=audit['maximum_stretch'],minimum_height_m=ground['minimum_vertex_height_m'],final_maximum_height_m=ground['final_bounds'][1][2],rescue_events=validation['rescue_events'],static_states_checked=ground['static_intersection_states_checked'],zero_motion_steps=audit['zero_motion_steps'],velocity_readback_error=audit['velocity_readback_error'],position_injection_error=audit['position_injection_error']))
    xylo=np.minimum(xylo,bv[:,:2].min(0))-.12;xyhi=np.maximum(xyhi,bv[:,:2].max(0))+.12
    floor=np.array([[xylo[0],xylo[1],0],[xyhi[0],xylo[1],0],[xyhi[0],xyhi[1],0],[xylo[0],xyhi[1],0]])
    bounds=np.vstack([floor@basis,bv@basis]);lo=np.minimum(lo,bounds[:,:2].min(0));hi=np.maximum(hi,bounds[:,:2].max(0));center=(lo+hi)/2;span=(hi-lo)*1.07
    width,height=600,470
    span[0]=max(span[0],span[1]*width/height);span[1]=span[0]*height/width
    low=center-span/2;high=center+span/2
    def pixels(v):
        p=(v@basis)[:,:2];p=(p-low)/span*np.array([width,height]);p[:,1]=height-p[:,1];return [tuple(v) for v in p]
    base=Image.new('RGB',(width,height),bg);draw=ImageDraw.Draw(base)
    draw.polygon(pixels(floor),fill='#dfe6e9',outline='#b0bfc8')
    for vx in np.arange(np.ceil(xylo[0]/.1)*.1,xyhi[0],.1):draw.line(pixels(np.array([[vx,xylo[1],0],[vx,xyhi[1],0]])),fill='#c7d2d9')
    for vy in np.arange(np.ceil(xylo[1]/.1)*.1,xyhi[1],.1):draw.line(pixels(np.array([[xylo[0],vy,0],[xyhi[0],vy,0]])),fill='#c7d2d9')
    light=np.array([.3,-.7,.64]);btri=bv[bf]
    def panel(k,step,rate):
        d=data[k];m=d['m'];tri=np.vstack([d['x'][step][d['f']],btri]);coords=tri@basis
        n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
        colors=d['colors']*(.55+.45*np.abs(n@light))[:,None]
        rgba=rasterize(coords,colors,low,high,width,height)
        layer=Image.fromarray(np.flipud(rgba));scene=base.copy();scene.paste(layer,(0,0),layer)
        tile=Image.new('RGB',(640,640),bg);tile.paste(scene,(20,100));d2=ImageDraw.Draw(tile)
        d2.text((22,10),f'L{k}  ·  {m["vertices"]:,} 顶点',font=font(27),fill=ink)
        d2.text((22,50),f'{m["faces"]:,} 三角形  |  计算 {m["wall_seconds"]:.1f} 秒',font=font(19),fill=muted)
        d2.text((22,574),f't = {step*.01:.2f} s  ·  {rate}',font=font(21),fill=ink)
        d2.text((22,607),'PSQ 粗层' if k==0 else f'{m["method"]}  L{k-1} → L{k}',font=font(18),fill=muted)
        return tile
    def mosaic(tiles,step,rate):
        canvas=Image.new('RGB',(2560,1440),bg);d=ImageDraw.Draw(canvas)
        d.text((24,15),'Bunny · 八层同步对比',font=font(36),fill=ink)
        d.text((660,25),f'同一相机与比例  ·  同一物理时刻  ·  {rate}  ·  t = {step*.01:.2f} / 4.00 s',font=font(27),fill=muted)
        for k,tile in enumerate(tiles):canvas.paste(tile,((k%4)*640,85+(k//4)*640))
        d.text((24,1384),'固定兔子与地面 z=0  |  布料—兔子摩擦 0.3，布料—地面摩擦 0  |  全部为实际仿真轨迹',font=font(25),fill=muted)
        return np.asarray(canvas)
    if a.preview_only:
        for step in [0,40,200,400]:imageio.imwrite(out/f'all_levels_{step:03}.png',mosaic([panel(k,step,'0.25× 慢放') for k in range(8)],step,'0.25× 慢放'))
    else:
        paths=[]
        with contextlib.ExitStack() as stack:
            def writer(path,frames):
                paths.append((path,frames));return stack.enter_context(imageio.get_writer(path,fps=25,codec='libx264',quality=8,macro_block_size=1))
            slow=writer(out/'bunny_L0_L7_comparison_slow.mp4',401);regular=writer(out/'bunny_L0_L7_comparison_1x.mp4',101)
            singles={}
            for k in range(3,8):
                rd=runs[k]/'report';rd.mkdir(exist_ok=True)
                singles[k]=(writer(rd/f'bunny_L{k}_ground_slow.mp4',401),writer(rd/f'bunny_L{k}_ground_1x.mp4',101))
            for step in range(401):
                tiles=[panel(k,step,'0.25× 慢放') for k in range(8)];frame=mosaic(tiles,step,'0.25× 慢放');slow.append_data(frame)
                for k,(ws,wr) in singles.items():ws.append_data(np.asarray(tiles[k]))
                if step in [0,40,200,400]:imageio.imwrite(out/f'all_levels_{step:03}.png',frame)
                if step%4==0:
                    for tile in tiles:
                        d=ImageDraw.Draw(tile);d.rectangle((20,570,639,604),fill=bg);d.text((22,574),f't = {step*.01:.2f} s  ·  1× 原速',font=font(21),fill=ink)
                    regular.append_data(mosaic(tiles,step,'1× 原速'))
                    for k,(ws,wr) in singles.items():wr.append_data(np.asarray(tiles[k]))
                if step%25==0:print(f'RENDERED {step}/400',flush=True)
        media=[]
        for path,expected in paths:
            cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                count+=1
            cap.release();assert count==expected and fps==25
            media.append(dict(file=str(path.resolve()),frames=count,fps=fps,duration_seconds=count/fps,passed=True))
        (out/'media_validation.json').write_text(json.dumps(media,indent=2))
    (a.root/'all_levels_summary.json').write_text(json.dumps(summary,indent=2))
    if not a.preview_only:
        protected=json.loads((a.root/'protected_manifest.json').read_text())
        changed=[]
        for name,expected in protected.items():
            with Path(name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected:changed.append(name)
        concurrent=json.loads((a.root/'concurrent_changes.json').read_text()) if (a.root/'concurrent_changes.json').exists() else {}
        assert set(changed)==set(concurrent),changed
        for name,record in concurrent.items():
            assert Path(name)==ROOT/'shirt_demo/simulate.py'
            with Path(name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            assert actual==record['observed_sha256'] and protected[name]==record['task_start_sha256']
        assert summary[7]['native_steps']==400 and summary[7]['preview_steps']==0
        validation=dict(passed=True,levels=8,steps=3200,saved_states_checked=sum(s['static_states_checked'] for s in summary),zero_motion_steps=sum(s['zero_motion_steps'] for s in summary),maximum_stretch=max(s['maximum_stretch'] for s in summary),strain_limit=1.068,finest_all_newton_converged=True,protected_files_checked=len(protected),protected_files_changed=changed,concurrent_changes_preserved=concurrent,file_protection_all_unchanged=not changed,media_validated=len(media),scope='Saved-state intersection and floor-height checks, not independent continuous collision certification')
        (a.root/'final_validation.json').write_text(json.dumps(validation,indent=2))
        lines=['Bunny L0-L7 with fixed ground: all eight levels completed and audited.',
            'Each level: 400 steps, dt=0.01 s, physical time 4 s. Gravity -9.81 m/s^2.',
            'Cloth-bunny/self friction 0.3; fixed frictionless ground at z=0.',
            'Dynamic mesh sizes: 1443 -> 2663 -> 4834 -> 8608 -> 15651 -> 28185 -> 50268 -> 90000.',
            'Fine-level integrators (L1-L7): '+', '.join(s['method'] for s in summary[1:]),
            'Solver: velocity increment tolerance 0.05, PCG relative tolerance 1e-3, line search limit 8, Newton budget 300.',
            'Coarse verified non-descent stopping is reported separately from Newton convergence; L7 requires Newton convergence.',
            'Isolated adaptive line-search/probe/full-gradient rescue extensions reused from repaired flag runtime. Not an unmodified author implementation.',
            'Ground and 4-second observation window are local scene choices; meshes/material constants are not authenticated author assets.',
            '', 'Level | Vertices | Wall s | Newton steps | Coarse stops | Max stretch | Rescues']
        lines.extend(f"L{s['level']} | {s['vertices']} | {s['wall_seconds']:.3f} | {s['native_steps']} | {s['preview_steps']} | {s['maximum_stretch']:.9f} | {s['rescue_events']}" for s in summary)
        lines.extend(['',f"All 3200 steps accepted; {validation['saved_states_checked']} saved states checked for static intersections and floor height; zero frozen steps.",
            f"{len(protected)-len(changed)}/{len(protected)} protected existing files unchanged. Recorded concurrent changes: {changed}. This task did not edit those files and preserved their current contents. See concurrent_changes.json.",
            f'Reused trajectory through L{len(ancestors)-1}. Simulation times exclude audits and rendering.',
            'Eight-panel videos use identical camera, projection scale, floor grid, and physical times.',
            'Slow playback: 0.25x, 401 frames / 25 fps = 16.04 s; original speed: 101 frames / 25 fps = 4.04 s.',
            'No independent continuous-time collision certification; no claim that every level has reached equilibrium or completely slid off bunny.'])
        (a.root/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        labels=[f'L{s["level"]}' for s in summary]
        axes[0,0].bar(labels,[s['wall_seconds'] for s in summary],color='#268eac');axes[0,0].set_ylabel('Simulation wall time (s)')
        axes[0,1].bar(labels,[s['native_steps'] for s in summary],label='Newton converged',color='#268eac')
        axes[0,1].bar(labels,[s['preview_steps'] for s in summary],bottom=[s['native_steps'] for s in summary],label='Verified coarse stop',color='#cfa361');axes[0,1].legend();axes[0,1].set_ylabel('Accepted steps')
        axes[1,0].plot(labels,[s['maximum_stretch'] for s in summary],'o-',color='#268eac');axes[1,0].axhline(1.068,ls='--',color='#b05546',label='Limit 1.068');axes[1,0].legend();axes[1,0].set_ylabel('Maximum stretch over 4 s')
        for k,d in enumerate(data):axes[1,1].plot(np.arange(401)*.01,np.max(d['x'][:,:,2],axis=1),label=f'L{k}')
        axes[1,1].set_xlabel('Physical time (s)');axes[1,1].set_ylabel('Highest cloth vertex (m)');axes[1,1].legend(ncol=2)
        for axis in axes.ravel():axis.grid(axis='y',alpha=.2)
        fig.suptitle('Bunny with ground | eight levels, 4 s per level | actual accepted trajectories')
        fig.savefig(out/'all_levels_metrics.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
