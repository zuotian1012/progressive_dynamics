"""Render the verified coarse bunny trajectory with its actual ground plane."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.font_manager import FontProperties
import imageio.v2 as imageio
from numba import njit

@njit
def rasterize(coords, colors, low, high, width, height, material_uv=None, alternate_colors=None):
    rgba=np.zeros((height,width,4),dtype=np.uint8)
    depth=np.full((height,width),-np.inf)
    for t in range(len(coords)):
        p=coords[t].copy()
        p[:,0]=(p[:,0]-low[0])/(high[0]-low[0])*width-.5
        p[:,1]=(p[:,1]-low[1])/(high[1]-low[1])*height-.5
        den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den)<1e-14:continue
        for iy in range(max(0,int(np.ceil(p[:,1].min()))),min(height-1,int(np.floor(p[:,1].max())))+1):
            for ix in range(max(0,int(np.ceil(p[:,0].min()))),min(width-1,int(np.floor(p[:,0].max())))+1):
                a=((p[1,1]-p[2,1])*(ix-p[2,0])+(p[2,0]-p[1,0])*(iy-p[2,1]))/den
                b=((p[2,1]-p[0,1])*(ix-p[2,0])+(p[0,0]-p[2,0])*(iy-p[2,1]))/den
                c=1-a-b
                if min(a,b,c)<-1e-10:continue
                z=a*p[0,2]+b*p[1,2]+c*p[2,2]
                if z>depth[iy,ix]:
                    depth[iy,ix]=z
                    color=colors[t]
                    if material_uv is not None and alternate_colors is not None and t<len(material_uv):
                        uv=material_uv[t]
                        u=a*uv[0,0]+b*uv[1,0]+c*uv[2,0]
                        v=a*uv[0,1]+b*uv[1,1]+c*uv[2,1]
                        if (int(np.floor(u/.08))+int(np.floor(v/.08)))%2:
                            color=alternate_colors[t]
                    for ch in range(3):rgba[iy,ix,ch]=int(255*min(1.,max(0.,color[ch])))
                    rgba[iy,ix,3]=255
    return rgba

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    m=json.loads((a.run/'run.json').read_text());assert m['completed'] and json.loads((a.run/'continuation_audit.json').read_text())['passed']
    out=a.run/'report';out.mkdir(exist_ok=True)
    d=np.load(Path(m['assets'])/f'mesh_{m["level"]}.npz');b=np.load(Path(m['assets'])/'bunny.npz');f=d['faces'];bf=b['faces'];bv=b['vertices']
    x=np.load(a.run/'positions.npy',mmap_mode='r')[:401];rows=json.loads((a.run/'steps.json').read_text())
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');bg='#f5f7fa';ink='#163144';muted='#596f80'
    az,el=np.deg2rad([-55,28]);camera=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0]);up=np.cross(camera,right);basis=np.stack([right,up,camera],axis=1)
    lo=x.min(axis=(0,1));hi=x.max(axis=(0,1));lo[:2]=np.minimum(lo[:2],bv[:,:2].min(0))-.12;hi[:2]=np.maximum(hi[:2],bv[:,:2].max(0))+.12
    floor=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0]])
    projected=np.vstack([s@basis for s in x]);bounds=np.vstack([projected,floor@basis,bv@basis]);low=bounds[:,:2].min(0);high=bounds[:,:2].max(0);center=(low+high)/2;span=(high-low)*1.08
    fig=plt.figure(figsize=(16,10),dpi=100,facecolor=bg);ax=fig.add_axes([.025,.14,.95,.72]);ax.set_aspect('equal');ax.axis('off')
    ax.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2);ax.set_ylim(center[1]-span[1]/2,center[1]+span[1]/2)
    ax.add_collection(PolyCollection([(floor@basis)[:,:2]],facecolors='#dfe6e9',edgecolors='#b0bfc8',linewidths=1,zorder=0))
    grid=[]
    for vx in np.arange(np.ceil(lo[0]/.1)*.1,hi[0],.1):grid.append([[vx,lo[1],0],[vx,hi[1],0]])
    for vy in np.arange(np.ceil(lo[1]/.1)*.1,hi[1],.1):grid.append([[lo[0],vy,0],[hi[0],vy,0]])
    ax.add_collection(LineCollection((np.array(grid)@basis)[:,:,:2],colors='#c7d2d9',linewidths=.5,zorder=.1))
    raster_low=center-span/2;raster_high=center+span/2
    surface=ax.imshow(np.zeros((720,1520,4),dtype=np.uint8),origin='lower',extent=[raster_low[0],raster_high[0],raster_low[1],raster_high[1]],interpolation='bilinear',zorder=1)
    centers=d['vertices'][f].mean(1);checker=(np.floor(centers[:,0]/.08)+np.floor(centers[:,1]/.08)).astype(int)%2
    base=np.vstack([np.where(checker[:,None]==0,[.10,.54,.73],[.25,.73,.82]),np.tile([.77,.75,.70],(len(bf),1))])
    fig.text(.035,.945,f'Bunny L{m["level"]} · 布料落到兔子与地面',fontproperties=font,fontsize=25,color=ink)
    fig.text(.035,.895,f'{m["vertices"]:,} 个动态顶点  /  90,000 顶点弹性代理   ·   固定兔子   ·   固定地面 z = 0',fontproperties=font,fontsize=14,color=muted)
    label=fig.text(.035,.10,'',fontproperties=font,fontsize=16,color=ink)
    mu=m['contact_friction']
    fig.text(.035,.045,f'{m["method"]}   ·   布料—兔子摩擦 {mu["cloth_bunny"]:g}   ·   布料自摩擦 {mu["cloth_self"]:g}   ·   布料—地面摩擦 {mu["cloth_ground"]:g}',fontproperties=font,fontsize=13,color=muted)
    light=np.array([.3,-.7,.64])
    def draw(k,rate):
        tri=np.vstack([x[k][f],bv[bf]]);coords=tri@basis;order=np.argsort(coords[:,:,2].mean(1));n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
        color=base*(.55+.45*np.abs(n@light))[:,None];surface.set_data(rasterize(coords,color,raster_low,raster_high,1520,720))
        label.set_text(f'物理时间 {k*.01:.2f} / 4.00 秒   ·   {rate}   ·   地面网格间距 0.1 m');fig.canvas.draw()
    samples=[0,20,40,60,100,200,400]
    if a.preview_only:
        for k in samples:draw(k,'1× 原速');fig.savefig(out/f'ground_{k:03}.png',dpi=100)
    else:
        with imageio.get_writer(out/f'bunny_L{m["level"]}_ground_1x.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1) as regular, imageio.get_writer(out/f'bunny_L{m["level"]}_ground_slow.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1) as slow:
            for k in range(401):
                draw(k,'0.25× 慢放');slow.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
                if k%4==0:
                    draw(k,'1× 原速');regular.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
                if k in samples:fig.savefig(out/f'ground_{k:03}.png',dpi=100)
                if k%100==0:print(f'RENDERED {k}/400',flush=True)
    plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True);t=np.arange(401)*.01
    axs[0].plot(t,x[:,:,2].min(1),label='Lowest cloth vertex');axs[0].plot(t,x[:,:,2].max(1),label='Highest cloth vertex');axs[0].axhline(0,color='k',lw=.7);axs[0].set_ylabel('Height (m)');axs[0].legend()
    axs[1].plot(t[1:],[r['maximum_stretch'] for r in rows]);axs[1].axhline(1.068,color='#b05546',ls='--');axs[1].set_ylabel('Maximum stretch')
    speed=np.linalg.norm(np.diff(x,axis=0),axis=2).max(1)/.01;axs[2].plot(t[1:],speed);axs[2].set_ylabel('Max vertex speed (m/s)');axs[2].set_xlabel('Physical time (s)')
    for axis in axs:axis.grid(alpha=.2)
    fig.suptitle(f'Bunny L{m["level"]} with frictionless ground | actual accepted states');fig.savefig(out/'ground_metrics.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
