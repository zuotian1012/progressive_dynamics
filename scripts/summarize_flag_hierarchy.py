"""Final cross-level evidence and overview, only after all 27 runs pass."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from report_flag_hierarchy import project,RENDER_VERSION

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'output/flag_to_L9_20261122'


def main():
    summary=json.loads((SUITE/'summary.json').read_text())
    assert summary['complete'] and not summary['failed']
    index=json.loads((SUITE/'index.json').read_text())
    base=json.loads((Path(index['L0']['run'])/'run.json').read_text())
    base_audit=json.loads((Path(index['L0']['root'])/'report/audit.json').read_text())
    assert base['status']=='completed' and base['accepted_frames']==200
    assert base_audit['all_native_steps_converged'] and base_audit['fixed_vertices']==2
    report=SUITE/'report';rows=[];manifest=[]
    for level in range(1,10):
        folder=report/f'L{level}'
        verified=json.loads((folder/'verified.json').read_text())
        assert verified['mp4_frames']==201 and verified['gif_frames']==101
        assert verified.get('render_version')==RENDER_VERSION
        metrics=json.loads((folder/'metrics.json').read_text())
        assert metrics['all_converged']
        for method in ['semi','velpro','direct']:
            entry=index['levels'][str(level)][method];assert entry['status']=='completed'
            run=Path(entry['run']);meta=json.loads((run/'run.json').read_text())
            audit=json.loads((Path(entry['root'])/'report'/run.name/'audit.json').read_text())
            assert meta['status']=='completed' and meta['accepted_frames']==200
            assert audit['all_native_steps_converged'] and not audit['saved_states_with_coarse_intersections']
            assert audit['fixed_vertices']==2 and audit['max_coarse_pin_error_m']<1e-12
            if method=='direct':
                assert audit['native_local_elastic_verified'] and not audit['psq_energy_replacement']
                assert not meta['coarse_trajectory_used'] and 'lower_run' not in meta
            else:
                assert audit['parent_trajectory_sha256_verified']
                assert audit['injected_predictor_max_error_m_s']<1e-10
                if level>1:assert Path(meta['lower_run'])==Path(index['levels'][str(level-1)][method]['run'])
            with (run/'positions.npy').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            bounds_min=np.full(2,np.inf);bounds_max=np.full(2,-np.inf)
            for state in np.load(run/'positions.npy',mmap_mode='r'):
                projected,_=project(state)
                bounds_min=np.minimum(bounds_min,projected.min(axis=0));bounds_max=np.maximum(bounds_max,projected.max(axis=0))
            assert np.all(bounds_min>=0) and np.all(bounds_max<np.array([480,450]))
            manifest.append(dict(level=level,method=method,run=str(run),accepted_steps=200,
                positions_sha256=digest,native_package=meta['native_package'],runtime_sha256=meta['runtime_sha256'],
                render_bounds_min_px=bounds_min.tolist(),render_bounds_max_px=bounds_max.tolist(),no_render_clipping=True))
        rows.append(dict(level=level,vertices=metrics['vertices'],faces=metrics['faces'],
            semi_mean_rms_mm=metrics['mean_area_weighted_rms_vs_direct_m'][0]*1000,
            velpro_mean_rms_mm=metrics['mean_area_weighted_rms_vs_direct_m'][1]*1000,
            maximum_stretch=max(metrics['max_stretch'])))
    with (report/'levels.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (report/'manifest.json').write_text(json.dumps(dict(verified_runs=27,verified_steps=5400,
        shared_L0=index['L0'],runs=manifest),indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,4.8),dpi=160,facecolor='#f3f6fa');ax.set_facecolor('#f3f6fa')
    levels=np.arange(1,10)
    ax.plot(levels,[r['semi_mean_rms_mm'] for r in rows],'o-',color='#c98736',label='Semi-diagonal',lw=2)
    ax.plot(levels,[r['velpro_mean_rms_mm'] for r in rows],'o-',color='#7772b6',label='VelPro',lw=2)
    ax.set_xticks(levels,[f'L{k}' for k in levels]);ax.set_ylabel('Mean position RMS vs native Direct (mm)')
    ax.set_ylim(bottom=0);ax.grid(alpha=.18);ax.legend(frameon=False)
    ax.set_title('Flag / comparison at every actual mesh level',loc='left',weight='bold',pad=18)
    fig.text(.10,.02,'Rest-area weighting; 200 matched time steps. Direct is a same-level reference, not exact physical ground truth.',fontsize=9,color='#647788')
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(report/'overview.png');plt.close(fig)
    tile_height=610  # Include the full free-edge label, before the RMS chart.
    mosaic=Image.new('RGB',(1440,3*tile_height),'#f3f6fa')
    for row,level in enumerate([1,5,9]):
        with Image.open(report/f'L{level}/frame_200.png') as im:mosaic.paste(im.crop((0,0,1440,tile_height)),(0,row*tile_height))
    mosaic.save(report/'L1_L5_L9_final.png')
    lines=['# 旗帜 L1–L9 三方法对比','',
        '27 组均完成 200 步并通过审计，共 5,400 步；共享的 L0 为另 200 步。每组保存 201 个状态，时长 8 秒。',
        '', '浏览动画：[逐层选择页面](index.html)。每层的三栏依次为 Semi-diagonal、VelPro、本层独立 Direct。',
        '', 'Direct 使用本层原生弹性和独立时间积分，不使用粗层轨迹或 PSQ 细层弹性能。',
        '', '| 层级 | 顶点 | 三角形 | Semi 平均位置差/mm | VelPro 平均位置差/mm |',
        '|---|---:|---:|---:|---:|']
    lines += [f"| L{r['level']} | {r['vertices']:,} | {r['faces']:,} | {r['semi_mean_rms_mm']:.2f} | {r['velpro_mean_rms_mm']:.2f} |" for r in rows]
    lines += ['', '位置差是相对同层 Direct 的静止面积加权 RMS，再对 200 个仿真时间点取平均。它包含整体位置和相位差，不应单独解读为褶皱精度；Direct 也不是精确物理真值。',
        '', '所有保存状态有限、两角点误差小于 1e-12 m、没有检测到自交；主伸长小于 1.1。渐进分支的父轨迹与逐步速度注入均已核验，Direct 自身速度历史与原生弹性能路径均已核验。',
        '', '各组并行运行的耗时受共享硬件负载影响，本次结果用于效果对比，不能直接用作严格的性能基准。',
        '', '可追溯文件：[全部运行与哈希](manifest.json)、[逐层指标](levels.csv)、[跨层曲线](overview.png)。']
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(Path(index['L0']['root'])/'report/flag_L0_parallel.mp4',report/'shared_L0.mp4')
    page=report/'index.html';html=page.read_text(encoding='utf-8')
    extra='<p id="final-results"><a href="RESULTS.md">完整结果与验证说明</a> · <a href="overview.png">跨层差异曲线</a> · <a href="L1_L5_L9_final.png">L1 / L5 / L9 同时刻截图</a></p><details><summary>查看共享 L0 驱动（911 顶点）</summary><p>L0 是两条渐进链共用的起点，不是三种方法的独立对照。</p><video controls loop playsinline src="shared_L0.mp4"></video></details>'
    if 'id="final-results"' not in html:page.write_text(html.replace('<label>选择层级',extra+'\n<label>选择层级'),encoding='utf-8')
    print(json.dumps(dict(runs=27,steps=5400,rows=rows),indent=2),flush=True)


if __name__=='__main__':main()
