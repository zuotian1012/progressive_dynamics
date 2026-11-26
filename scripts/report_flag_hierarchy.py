"""Depth-buffered renders and matched-time comparisons for the complete flag levels."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'output/flag_to_L9_20261122'
METHODS=['semi','velpro','direct']
RENDER_VERSION=2
COLORS=np.array([[.91,.59,.28],[.48,.45,.76],[.30,.67,.46]])
CAM=np.array([.3868,-.8688,.3090]);RIGHT=np.array([.9135,.4067,0]);UP=np.cross(CAM,RIGHT);UP/=np.linalg.norm(UP)


@njit(cache=True)
def raster(base,uv,z,faces,colors,wire):
    h,w,_=base.shape;depth=np.full((h,w),-1e30)
    for fidx in range(len(faces)):
        a,b,c=faces[fidx];x0,y0=uv[a];x1,y1=uv[b];x2,y2=uv[c]
        den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(den)<1e-12:continue
        for py in range(max(0,int(np.floor(min(y0,y1,y2)))),min(h-1,int(np.ceil(max(y0,y1,y2))))+1):
            for px in range(max(0,int(np.floor(min(x0,x1,x2)))),min(w-1,int(np.ceil(max(x0,x1,x2))))+1):
                u=((y1-y2)*(px+.5-x2)+(x2-x1)*(py+.5-y2))/den
                v=((y2-y0)*(px+.5-x2)+(x0-x2)*(py+.5-y2))/den;q=1-u-v
                if min(u,v,q)<0:continue
                zz=u*z[a]+v*z[b]+q*z[c]
                if zz<=depth[py,px]:continue
                depth[py,px]=zz
                dark=.75 if wire and min(u,v,q)<.018 else 1.
                for ch in range(3):base[py,px,ch]=int(min(255,max(0,colors[fidx,ch]*255*dark)))
    return base


def project(x):
    centered=x-np.array([.75,0,.30])
    return np.column_stack([240+centered@RIGHT*220,235-centered@UP*220]),centered@CAM


def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('arialbd.ttf' if bold else 'arial.ttf'),size)


def render(level,entries):
    out=SUITE/'report'/f'L{level}';out.mkdir(parents=True,exist_ok=True)
    if (out/'verified.json').exists() and json.loads((out/'verified.json').read_text()).get('render_version')==RENDER_VERSION:return
    paths=[Path(entries[m]['run']) for m in METHODS];roots=[Path(entries[m]['root']) for m in METHODS]
    metas=[json.loads((p/'run.json').read_text()) for p in paths]
    audits=[json.loads((r/'report'/p.name/'audit.json').read_text()) for r,p in zip(roots,paths)]
    assert all(a['all_native_steps_converged'] for a in audits)
    assert all(m['config']==metas[0]['config'] for m in metas)
    assert metas[2]['elastic_model']=='native_current_level' and not metas[2]['coarse_trajectory_used']
    data=np.load(roots[0]/f'assets/mesh_{level}.npz');rest,faces=data['vertices'],data['faces']
    xs=[np.load(p/'positions.npy',mmap_mode='r') for p in paths]
    assert all(x.shape==(201,len(rest),3) for x in xs)
    for x in xs:np.testing.assert_array_equal(x[0],xs[0][0])
    wind=[json.loads((p/'applied_wind.json').read_text()) for p in paths];assert wind[0]==wind[1]==wind[2]
    mass=np.zeros(len(rest));tri=rest[faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    np.add.at(mass,faces.ravel(),np.repeat(area/3,3));mass/=mass.sum()
    rms=[np.sqrt(np.einsum('i,tij,tij->t',mass,x-xs[2],x-xs[2])) for x in xs[:2]]
    edge=np.isclose(rest[:,0],1.5,atol=1e-12);heights=[x[:,edge,2].mean(axis=1) for x in xs]
    metrics=dict(level=level,vertices=len(rest),faces=len(faces),methods=METHODS,
        mean_area_weighted_rms_vs_direct_m=[float(v[1:].mean()) for v in rms],
        max_area_weighted_rms_vs_direct_m=[float(v.max()) for v in rms],
        final_area_weighted_rms_vs_direct_m=[float(v[-1]) for v in rms],
        max_stretch=[a['max_principal_stretch'] for a in audits],
        mean_free_edge_height_last_2s_m=[float(v[-51:].mean()) for v in heights],
        elapsed_s=[m['elapsed_s'] for m in metas],all_converged=True,
        rms_definition='Rest-area-weighted node position RMS versus same-level native Direct; not exact physical ground truth.')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    t=np.arange(201)*.04
    np.savetxt(out/'metrics.csv',np.column_stack([t,*rms,*heights]),delimiter=',',header='time_s,semi_rms_m,velpro_rms_m,semi_free_z,velpro_free_z,direct_free_z',comments='')
    base=Image.new('RGB',(1440,800),'#f3f6fa');draw=ImageDraw.Draw(base)
    draw.text((35,18),f'FLAG / L{level} COMPARISON',font=font(30,True),fill='#163347')
    draw.text((35,61),f'{len(rest):,} vertices / {len(faces):,} triangles | Two fixed corners | +x wind | dt = 0.04 s',font=font(18),fill='#416476')
    for i,name in enumerate(['Semi-diagonal','VelPro','Direct (native local)']):draw.text((30+480*i,103),name,font=font(24,True),fill=tuple((COLORS[i]*190).astype(int)))
    draw.text((35,760),'Actual mesh / displacement scale 1:1 | Direct uses this level only | 200 converged steps per run',font=font(18),fill='#647788')
    panel=Image.new('RGB',(480,450),'#f3f6fa');pd=ImageDraw.Draw(panel)
    uv,_=project(np.array([[0,0,-.65],[0,0,1.42]]));pd.line([tuple(v) for v in uv],fill='#4a6072',width=4)
    uv,_=project(np.array([[0,0,.25],[1.5,0,.25],[1.5,0,1.25],[0,0,1.25],[0,0,.25]]));pd.line([tuple(v) for v in uv],fill='#ccd6e0',width=1)
    uv,_=project(np.array([[.2,0,1.4],[1.,0,1.4],[.9,0,1.44],[1.,0,1.4],[.9,0,1.36]]));pd.line([tuple(v) for v in uv],fill='#bd852d',width=3)
    pins,_=project(np.array([[0,0,.25],[0,0,1.25]]));gifs=[]
    chartmax=max(v.max()*1000 for v in rms)*1.15 or 1
    def linepoints(v,k):return [(70+j/200*1300,720-float(v[j])*1000/chartmax*85) for j in range(k+1)]
    with imageio.get_writer(out/'comparison.mp4',fps=25,codec='libx264',macro_block_size=1,quality=8) as writer:
        for k in range(201):
            im=base.copy();dr=ImageDraw.Draw(im)
            dr.text((1190,23),f'{t[k]:.2f} / 8.00 s',font=font(24,True),fill='#167e99')
            for i,x in enumerate(xs):
                tri=x[k][faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
                shades=(.68+.32*np.abs(n@np.array([.3,-.7,.64])))[:,None]*COLORS[i]
                uv,z=project(x[k]);pix=raster(np.array(panel),uv,z,faces,shades,False)
                tile=Image.fromarray(pix);td=ImageDraw.Draw(tile)
                for a,b in pins:td.ellipse((a-4,b-4,a+4,b+4),fill='#cf5158')
                im.paste(tile,(i*480,140));dr.text((30+480*i,582),f'Free-edge height: {heights[i][k]:.3f} m',font=font(18),fill='#416476')
            dr.text((70,612),'Position RMS vs Direct (mm, rest-area weighted)',font=font(16),fill='#416476')
            dr.line([(70,635),(70,720),(1370,720)],fill='#92a4b3',width=1)
            for i,v in enumerate(rms):
                dr.line(linepoints(v,200),fill=tuple((COLORS[i]*70+180).astype(int)),width=1)
                if k:dr.line(linepoints(v,k),fill=tuple((COLORS[i]*230).astype(int)),width=3)
            for sec in range(9):dr.text((65+sec/8*1300,725),str(sec),font=font(13),fill='#647788')
            dr.text((12,633),str(round(chartmax)),font=font(13),fill='#647788');dr.text((29,707),'0',font=font(13),fill='#647788')
            rgb=np.array(im);writer.append_data(rgb)
            if k%2==0:gifs.append(im.resize((1080,600)))
            if k in [0,40,100,200]:im.save(out/f'frame_{k:03}.png')
    gifs[0].save(out/'comparison.gif',save_all=True,append_images=gifs[1:],duration=80,loop=0)
    reader=imageio.get_reader(out/'comparison.mp4');assert reader.count_frames()==201;reader.get_data(200);reader.close()
    with Image.open(out/'comparison.gif') as im:assert im.n_frames==101
    (out/'verified.json').write_text(json.dumps(dict(mp4_frames=201,gif_frames=101,all_frames_real=True,render_version=RENDER_VERSION)),encoding='utf-8')
    print('RENDERED L'+str(level),flush=True)


def gallery():
    rows=[]
    for file in sorted((SUITE/'report').glob('L*/metrics.json'),key=lambda p:int(p.parent.name[1:])):
        if (file.parent/'verified.json').exists():rows.append(json.loads(file.read_text()))
    data=json.dumps(rows)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>旗帜逐层对比</title>
<style>body{max-width:1440px;margin:32px auto;padding:0 24px;background:#f3f6fa;color:#163347;font:17px system-ui}video{width:100%;border-radius:12px;background:#e7edf3}select{font:inherit;padding:8px}table{border-collapse:collapse;width:100%;margin-top:24px}td,th{text-align:left;padding:12px;border-bottom:1px solid #d3dce3}p{color:#416476}a{color:#147e9a}</style>
<h1>旗帜：Semi-diagonal / VelPro / Direct</h1><p>两端固定 · +x 周期体力 · 初始扰动 1 mm · 每组 8 秒 / 200 步。Direct 使用本层弹性能，不使用 PD / PSQ 细层代理。</p>
<label>选择层级 <select id="level"></select></label><p id="info"></p><video id="video" controls loop playsinline></video><p><a id="download">下载 MP4</a> · <a id="gif">GIF 预览</a></p>
<p>位置差采用静止面积加权 RMS，参照同层 Direct；它衡量结果差异，不等于精确物理真值误差。</p>
<table><thead><tr><th>层级</th><th>顶点 / 三角形</th><th>Semi 平均 RMS</th><th>VelPro 平均 RMS</th><th>状态</th></tr></thead><tbody id="table"></tbody></table>
<script>const data=DATA;const s=document.getElementById('level');for(const r of data){s.add(new Option('L'+r.level,r.level));document.getElementById('table').insertAdjacentHTML('beforeend',`<tr><td>L${r.level}</td><td>${r.vertices} / ${r.faces}</td><td>${(r.mean_area_weighted_rms_vs_direct_m[0]*1000).toFixed(2)} mm</td><td>${(r.mean_area_weighted_rms_vs_direct_m[1]*1000).toFixed(2)} mm</td><td>3 × 200 步收敛并通过审计</td></tr>`)}function show(){let r=data.find(x=>x.level==s.value);let p='L'+r.level+'/comparison';document.getElementById('video').src=p+'.mp4';document.getElementById('download').href=p+'.mp4';document.getElementById('gif').href=p+'.gif';document.getElementById('info').textContent=`L${r.level}：${r.vertices} 个顶点，${r.faces} 个三角形。颜色依次为 Semi（橙）、VelPro（紫）、Direct（绿）。`}s.onchange=show;if(data.length){s.value=data[data.length-1].level;show()}</script></html>'''.replace('DATA',data)
    (SUITE/'report/index.html').write_text(html,encoding='utf-8')
    (SUITE/'report/summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--level',type=int);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        index=json.loads((SUITE/'index.json').read_text())
        for key,entries in index['levels'].items():
            if args.level is not None and int(key)!=args.level:continue
            if len(entries)==3 and all(e['status']=='completed' for e in entries.values()):render(int(key),entries)
        gallery()
        if not args.watch:break
        if (SUITE/'summary.json').exists():
            # Completion can race the index snapshot read at the loop start.
            final_index=json.loads((SUITE/'index.json').read_text())
            for key,entries in final_index['levels'].items():
                if args.level is not None and int(key)!=args.level:continue
                if len(entries)==3 and all(e['status']=='completed' for e in entries.values()):render(int(key),entries)
            gallery();break
        time.sleep(20)


if __name__=='__main__':main()
