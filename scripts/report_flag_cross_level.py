"""Ten-level, matched-time views of each completed flag chain, plus coherence metrics."""
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import load_npz

from report_flag_hierarchy import ROOT, SUITE, COLORS, project, raster, font
sys.path.insert(0, str(ROOT))
from progressive.prolongation import ShellProlongation

OUT = SUITE / 'report/cross_level'
BG = '#f3f6fa'
NAMES = {'semi': 'Semi-diagonal', 'velpro': 'VelPro'}


def load_chain(index, method):
    chain = []
    for level in range(10):
        entry = index['L0'] if level == 0 else index['levels'][str(level)][method]
        root, run = Path(entry['root']), Path(entry['run'])
        meta = json.loads((run / 'run.json').read_text())
        audit_path = root / 'report' / ('audit.json' if level == 0 else f'{run.name}/audit.json')
        audit = json.loads(audit_path.read_text())
        assert meta['status'] == 'completed' and meta['accepted_frames'] == 200
        assert meta['dt'] == .04 and meta['wind_scale'] == 3 and meta['pin_mode'] == 'corners'
        assert audit['all_native_steps_converged'] and audit['fixed_vertices'] == 2
        mesh = np.load(root / f'assets/mesh_{level}.npz')
        rest, faces = mesh['vertices'], mesh['faces']
        x = np.load(run / 'positions.npy', mmap_mode='r')
        assert x.shape == (201, len(rest), 3) and np.isfinite(x).all()
        if level:
            assert Path(meta['lower_run']).resolve() == chain[-1]['run'].resolve()
            assert meta['lower_positions_sha256'] == chain[-1]['sha256']
        tri = rest[faces]
        area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
        weight = np.zeros(len(rest))
        np.add.at(weight, faces.ravel(), np.repeat(area/3, 3))
        weight /= weight.sum()
        chain.append(dict(level=level, root=root, run=run, rest=rest, faces=faces, x=x,
                          weight=weight, sha256=hashlib.sha256((run/'positions.npy').read_bytes()).hexdigest()))
    return chain


def coherence(chain, method):
    curves = []
    rows = []
    for level in range(1, 10):
        low, high = chain[level-1], chain[level]
        assets = high['root']/'assets'
        U = load_npz(assets/f'map_{level-1}_{level}.npz')
        P = ShellProlongation(low['rest'], high['rest'], low['faces'], U,
                              np.load(assets/f'anchors_{level-1}_{level}.npy'))
        np.testing.assert_allclose(P.position(low['rest']), high['rest'], rtol=0, atol=1e-12)
        rms = []
        for t in range(201):
            delta = high['x'][t] - P.position(low['x'][t])
            rms.append(np.sqrt(np.einsum('i,ij,ij->', high['weight'], delta, delta)))
        rms = np.array(rms)*1000
        curves.append(rms)
        rows.append(dict(level=level, mean_adjacent_rms_mm=float(rms[1:].mean()),
                         max_adjacent_rms_mm=float(rms[1:].max()), final_adjacent_rms_mm=float(rms[-1]),
                         initial_adjacent_rms_mm=float(rms[0]), psq_zero_rest_offset=P.zero_offset))
    curves = np.array(curves)
    np.savetxt(OUT/f'{method}_adjacent_rms.csv', np.column_stack([np.arange(201)*.04, curves.T]),
               delimiter=',', header='time_s,'+','.join(f'L{k}_vs_P_L{k-1}_mm' for k in range(1,10)), comments='')
    return rows, curves


def render_chain(chain, method):
    color = COLORS[0 if method == 'semi' else 1]
    base = Image.new('RGB', (2400,1240), BG)
    draw = ImageDraw.Draw(base)
    draw.text((32,18), f'{NAMES[method]} / L0 - L9 / SAME PHYSICAL TIME', font=font(34,True), fill='#163347')
    draw.text((32,64), 'Top row: L0-L4 | Bottom row: L5-L9 | Shared L0 driver | Identical camera and scale', font=font(23), fill='#416476')
    draw.text((32,1200), 'Actual simulation meshes | Two pinned corners | No displacement magnification | 200 steps / 8 seconds', font=font(22), fill='#647788')
    for level, data in enumerate(chain):
        col, row = level%5, level//5
        draw.text((col*480+26,row*540+105), f'L{level}'+('  (shared)' if level==0 else ''), font=font(28,True), fill=tuple((color*190).astype(int)))
        draw.text((col*480+26,row*540+139), f"{len(data['rest']):,} vertices / {len(data['faces']):,} triangles", font=font(18), fill='#416476')
    panel = Image.new('RGB',(480,450),BG)
    pd = ImageDraw.Draw(panel)
    uv,_ = project(np.array([[0,0,-.65],[0,0,1.42]]))
    pd.line([tuple(v) for v in uv],fill='#4a6072',width=4)
    uv,_ = project(np.array([[0,0,.25],[1.5,0,.25],[1.5,0,1.25],[0,0,1.25],[0,0,.25]]))
    pd.line([tuple(v) for v in uv],fill='#ccd6e0',width=1)
    pins,_ = project(np.array([[0,0,.25],[0,0,1.25]]))
    gifs = []
    with imageio.get_writer(OUT/f'{method}_L0_L9.mp4',fps=25,codec='libx264',macro_block_size=1,quality=8) as writer:
        for k in range(201):
            im = base.copy()
            ImageDraw.Draw(im).text((2110,23),f'{k*.04:.2f} / 8.00 s',font=font(28,True),fill='#167e99')
            for level,data in enumerate(chain):
                x,faces = data['x'][k],data['faces']
                tri=x[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
                shades=(.68+.32*np.abs(normal@np.array([.3,-.7,.64])))[:,None]*color
                uv,z=project(x)
                assert (uv>=0).all() and (uv<np.array([480,450])).all(), (method,level,k,'clipping')
                tile=Image.fromarray(raster(np.array(panel),uv,z,faces,shades,False))
                td=ImageDraw.Draw(tile)
                for a,b in pins:td.ellipse((a-4,b-4,a+4,b+4),fill='#cf5158')
                im.paste(tile,(level%5*480,level//5*540+180))
            writer.append_data(np.asarray(im))
            if k%2==0:gifs.append(im.resize((1440,744),Image.Resampling.LANCZOS))
            if k in (0,50,100,150,200):im.save(OUT/f'{method}_frame_{k:03}.png')
            if k%50==0:print(method,'render',k,flush=True)
    gifs[0].save(OUT/f'{method}_L0_L9.gif',save_all=True,append_images=gifs[1:],duration=80,loop=0,optimize=False,disposal=2)
    reader=imageio.get_reader(OUT/f'{method}_L0_L9.mp4')
    assert reader.count_frames()==201
    assert reader.get_data(200).shape[:2]==(1240,2400)
    reader.close()
    with Image.open(OUT/f'{method}_L0_L9.gif') as check:assert check.n_frames==101


def reports(stats, curves, chains):
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(19,5.3),dpi=140,facecolor=BG)
    for ax in axes:ax.set_facecolor(BG);ax.grid(alpha=.18)
    levels=np.arange(1,10)
    for i,method in enumerate(NAMES):
        axes[0].plot(levels,[r['mean_adjacent_rms_mm'] for r in stats[method]],'o-',color=COLORS[i],label=NAMES[method])
    axes[0].set_title('Adjacent-level coherence / time mean',loc='left',weight='bold')
    axes[0].set_xticks(levels,[f'{k-1}/{k}' for k in levels]);axes[0].set_xlabel('Adjacent levels')
    axes[0].set_ylabel('Area-weighted position RMS (mm)');axes[0].set_ylim(bottom=0);axes[0].legend(frameon=False)
    ymax=max(float(c.max()) for c in curves.values())*1.05
    colors=plt.get_cmap('viridis')(np.linspace(.05,.90,9))
    for ax,method in zip(axes[1:],NAMES):
        for i,c in enumerate(curves[method]):ax.plot(np.arange(201)*.04,c,color=colors[i],lw=1,label=f'L{i} / L{i+1}')
        ax.set_title(NAMES[method]+' / matched-time differences',loc='left',weight='bold')
        ax.set_xlim(0,8);ax.set_ylim(0,ymax);ax.set_xlabel('Physical time (s)')
    axes[2].legend(ncol=3,fontsize=8,frameon=False,loc='upper left')
    fig.text(.04,.025,'Compare x[Lk](t) with PSQ(x[Lk-1](t)), on the Lk mesh. Lower means more similar to the parent, not greater physical accuracy.',fontsize=11,color='#647788')
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(OUT/'adjacent_consistency.png');plt.close(fig)
    manifest={'completed':True,'mp4_frames':201,'gif_frames':101,'duration_s':8,'levels':list(range(10)),
              'all_meshes_actual':True,'all_frames_in_bounds':True,'same_camera_and_scale':True,
              'metric':'Rest-area-weighted RMS of x[Lk](t) - P[Lk-1,Lk](x[Lk-1](t)); time mean excludes initial state.',
              'stats':stats,'source_runs':{m:[{'level':d['level'],'run':str(d['run']),'positions_sha256':d['sha256']} for d in chain] for m,chain in chains.items()}}
    (OUT/'verified.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=''.join(f"<tr><td>L{k-1} → L{k}</td><td>{stats['semi'][k-1]['mean_adjacent_rms_mm']:.3f}</td><td>{stats['velpro'][k-1]['mean_adjacent_rms_mm']:.3f}</td></tr>" for k in range(1,10))
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>旗帜十层一致性</title>
<style>body{max-width:1800px;margin:24px auto;padding:0 20px;background:#f3f6fa;color:#163347;font:17px system-ui}video,img{width:100%;border-radius:10px}button,select{font:inherit;padding:8px 14px}input{width:min(600px,50vw)}.controls{position:sticky;top:0;background:#edf2f7;padding:14px;z-index:2;display:flex;gap:14px;align-items:center;flex-wrap:wrap}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ccd6e0}a{color:#147e9a}p{color:#416476}</style>
<h1>旗帜：各方法内部的 L0–L9 一致性</h1>
<p>每张动画上排 L0–L4，下排 L5–L9。两组共享同一个 L0；所有层使用相同物理时刻、相机与空间比例，展示实际网格。完整视频 2400 × 1240，可全屏查看褶皱。</p>
<div class="controls"><button id="play">同步播放</button><button id="pause">暂停</button><label>时间步 <input id="seek" type="range" min="0" max="200" step="1" value="0"></label><output id="time">0.00 / 8.00 s</output><label>速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label><span id="status"></span></div>
<h2>Semi-diagonal</h2><video id="semi" muted playsinline preload="metadata" src="semi_L0_L9.mp4"></video><p><button data-full="semi">全屏</button> <a href="semi_L0_L9.mp4">完整 MP4</a> · <a href="semi_L0_L9.gif">GIF</a> · <a href="semi_frame_150.png">6 秒高清截图</a></p>
<h2>VelPro</h2><video id="velpro" muted playsinline preload="metadata" src="velpro_L0_L9.mp4"></video><p><button data-full="velpro">全屏</button> <a href="velpro_L0_L9.mp4">完整 MP4</a> · <a href="velpro_L0_L9.gif">GIF</a> · <a href="velpro_frame_150.png">6 秒高清截图</a></p>
<h2>相邻层的位置一致性</h2><p>先把父层同一时刻的位置通过 PSQ 延拓到子层，再与子层的实际仿真位置比较；按子层静止面积加权计算 RMS，对 200 个时间步取平均，单位 mm。数值越小表示越贴近父层，也可能意味着新细节更少，并不等于物理结果更准确。该指标没有去除整体运动或相位差。初始微扰在各网格独立采样，所以初态的层间差可能非零。</p>
<img src="adjacent_consistency.png" alt="相邻层一致性曲线"><table><thead><tr><th>相邻层</th><th>Semi 平均 RMS / mm</th><th>VelPro 平均 RMS / mm</th></tr></thead><tbody>ROWS</tbody></table>
<p><a href="semi_adjacent_rms.csv">Semi 逐帧指标</a> · <a href="velpro_adjacent_rms.csv">VelPro 逐帧指标</a> · <a href="verified.json">验证与源轨迹哈希</a> · <a href="../index.html">同层三方法对比</a></p>
<script>
const a=document.getElementById('semi'),b=document.getElementById('velpro'),videos=[a,b],seek=document.getElementById('seek'),time=document.getElementById('time');
function pause(){videos.forEach(v=>v.pause())}
document.getElementById('play').onclick=async()=>{if(a.ended||a.currentTime>=8)videos.forEach(v=>v.currentTime=0);b.currentTime=a.currentTime;try{await Promise.all(videos.map(v=>v.play()));document.getElementById('status').textContent=''}catch(e){pause();document.getElementById('status').textContent='请使用下方 MP4 链接播放'}};
document.getElementById('pause').onclick=pause;
seek.oninput=()=>{pause();videos.forEach(v=>v.currentTime=Number(seek.value)*.04)};
document.getElementById('speed').onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));
document.querySelectorAll('[data-full]').forEach(el=>el.onclick=()=>document.getElementById(el.dataset.full).requestFullscreen());
function tick(){if(!a.paused&&Math.abs(a.currentTime-b.currentTime)>.06)b.currentTime=a.currentTime;seek.value=Math.min(200,Math.round(a.currentTime/.04));time.textContent=Math.min(8,a.currentTime).toFixed(2)+' / 8.00 s';requestAnimationFrame(tick)}tick();
</script></html>'''.replace('ROWS',rows)
    (OUT/'index.html').write_text(html,encoding='utf-8')
    gallery=SUITE/'report/index.html'
    page=gallery.read_text(encoding='utf-8')
    if 'id="cross-level-link"' not in page:
        page=page.replace('<label>选择层级','<p id="cross-level-link"><a href="cross_level/index.html">分别查看 Semi / VelPro 的 L0–L9 十层同步对比与层间一致性</a></p>\n<label>选择层级')
        gallery.write_text(page,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    index=json.loads((SUITE/'index.json').read_text())
    assert json.loads((SUITE/'summary.json').read_text())['complete']
    chains={m:load_chain(index,m) for m in NAMES}
    assert chains['semi'][0]['sha256']==chains['velpro'][0]['sha256']
    stats={};curves={}
    for method,chain in chains.items():
        stats[method],curves[method]=coherence(chain,method)
        render_chain(chain,method)
    reports(stats,curves,chains)
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
