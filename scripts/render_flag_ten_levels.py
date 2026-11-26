"""Render all ten verified trajectories together, without rerunning simulation."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
import imageio.v2 as imageio

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    assert json.loads((a.root/'final_validation.json').read_text())['passed']
    a.output.mkdir(parents=True,exist_ok=True)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    bg='#f5f7fa';ink='#163144';muted='#596f80'
    az,el=np.deg2rad([-62,18]);camera=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0]);up=np.cross(camera,right);basis=np.stack([right,up,camera],axis=1)
    data=[];lo=np.full(2,np.inf);hi=-lo
    for k in range(10):
        run=a.root/f'L{k}';m=json.loads((run/'run.json').read_text());assert m['completed'] and m['completed_steps']==500 and m['dt']==.04
        assert json.loads((run/'continuation_audit.json').read_text())['passed']
        mesh=np.load(Path(m['assets'])/f'mesh_{k}.npz');f=mesh['faces'];rest=mesh['vertices'];x=np.load(run/'positions.npy',mmap_mode='r')
        fc=rest[f].mean(1);checker=(np.floor(fc[:,0]/.08)+np.floor(fc[:,2]/.08)).astype(int)%2
        colors=np.where(checker[:,None]==0,[.10,.54,.73],[.25,.73,.82])
        for state in x[:501]:
            projected=state@basis;lo=np.minimum(lo,projected[:,:2].min(0));hi=np.maximum(hi,projected[:,:2].max(0))
        data.append(dict(m=m,x=x,f=f,colors=colors,run=str(run.resolve())))
    pole=np.array([[0,0,.1],[0,0,1.4]])@basis
    lo=np.minimum(lo,pole[:,:2].min(0));hi=np.maximum(hi,pole[:,:2].max(0));center=(lo+hi)/2
    # Every panel uses exactly the same orthographic projection and bounds.
    span=(hi-lo)*1.09
    fig=plt.figure(figsize=(25.6,14.4),dpi=100,facecolor=bg);polys=[]
    for k,d in enumerate(data):
        row,col=divmod(k,5);left=.022+col*.195;bottom=.525 if row==0 else .105
        ax=fig.add_axes([left,bottom,.183,.34]);ax.set_facecolor(bg);ax.axis('off');ax.set_aspect('equal')
        ax.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2);ax.set_ylim(center[1]-span[1]/2,center[1]+span[1]/2)
        ax.plot(pole[:,0],pole[:,1],color=muted,lw=2.4,zorder=0)
        poly=PolyCollection([],edgecolors='none');ax.add_collection(poly);polys.append(poly)
        fig.text(left,bottom+.365,f'L{k}',fontsize=23,weight='bold',color=ink)
        fig.text(left+.040,bottom+.367,f'{d["m"]["vertices"]:,} 顶点  /  {d["m"]["faces"]:,} 三角形',fontproperties=font,fontsize=12,color=muted)
    fig.text(.022,.958,'旗帜仿真 · 十层同步对比',fontproperties=font,fontsize=29,color=ink)
    fig.text(.022,.919,'同一相机与比例   ·   同一物理时刻   ·   1× 原速',fontproperties=font,fontsize=17,color=muted)
    label=fig.text(.974,.948,'',ha='right',fontsize=23,color=ink)
    fig.text(.022,.045,'每层 20 秒 / 500 步   ·   PSQ + VelPro（含已记录修复）   ·   全部为实际仿真轨迹',fontproperties=font,fontsize=15,color=muted)
    bar=fig.add_axes([.022,.075,.952,.004]);bar.set_xlim(0,20);bar.set_ylim(0,1);bar.axis('off');bar.axhspan(0,1,color='#dbe4ea');progress=bar.barh(.5,0,height=1,color='#1b86a5')[0]
    light=np.array([.3,-.7,.64])
    def draw(k):
        for d,poly in zip(data,polys):
            tri=d['x'][k][d['f']];projected=tri@basis;order=np.argsort(projected[:,:,2].mean(1))
            normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
            colors=d['colors']*(.72+.28*np.abs(normal@light))[:,None]
            poly.set_verts(projected[order,:,:2]);poly.set_facecolor(colors[order])
        label.set_text(f't = {k*.04:05.2f} / 20.00 s');progress.set_width(k*.04);fig.canvas.draw()
    if a.preview_only:
        for k in [0,250,500]:draw(k);fig.savefig(a.output/f'frame_{k:03}.png',dpi=100)
        plt.close(fig);return
    movie=a.output/'flag_L0_L9_comparison.mp4'
    with imageio.get_writer(movie,fps=25,codec='libx264',macro_block_size=1,quality=8,pixelformat='yuv420p') as writer:
        for k in range(501):
            draw(k);writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
            if k in [0,250,500]:fig.savefig(a.output/f'frame_{k:03}.png',dpi=100)
            if k%50==0:print(f'RENDERED {k}/500',flush=True)
    plt.close(fig)
    manifest=dict(levels=list(range(10)),layout='2 rows x 5 columns, row-major L0-L4 / L5-L9',size=[2560,1440],fps=25,frames=501,physical_seconds=20,display_seconds=20.04,playback_rate=1,
        initial_state_included=True,simulation_rerun=False,shared_camera=True,shared_scale=True,source_runs=[d['run'] for d in data],sha256=hashlib.sha256(movie.read_bytes()).hexdigest())
    (a.output/'render_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
