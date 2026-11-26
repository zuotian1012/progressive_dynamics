"""PD 2024 Fig.18 kinetic energies and Fig.19 interlevel COM distances.

Saved 8-second flag suite only. Velocities are own-level backward differences,
not progressive predictor injections. Reference is each method's own L9.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from report_flag_paper_consistency import load_sources, digest, SUITE, METHODS, COLORS, TIMES

OUT = SUITE / 'report/pd2024_consistency'


def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def calculate(chains):
    data, per_frame, summaries, checks = {}, [], [], []
    for method, chain in chains.items():
        kinetic, centers, kinetic_com, kinetic_relative = [], [], [], []
        for level, d in enumerate(chain):
            x, mass = d['x'], d['mass']
            assert d['meta']['initial_velocity_m_s'] == 0
            v = np.zeros_like(x)
            v[1:] = np.diff(x, axis=0) / .04
            np.testing.assert_allclose(v[:,d['meta']['pin_indices']], 0, atol=1e-12)
            cm = np.einsum('i,tij->tj', mass, x) / mass.sum()
            cmv = np.einsum('i,tij->tj', mass, v) / mass.sum()
            K = .5 * np.einsum('i,tij,tij->t', mass, v, v)
            Kcm = .5 * mass.sum() * np.einsum('ti,ti->t', cmv, cmv)
            relative = v - cmv[:,None,:]
            Krel = .5 * np.einsum('i,tij,tij->t', mass, relative, relative)
            np.testing.assert_allclose(K, Kcm+Krel, atol=1e-14, rtol=1e-12)
            np.testing.assert_allclose(np.diff(cm,axis=0)/.04, cmv[1:], atol=1e-12, rtol=1e-11)
            velocity_check = None
            if method == 'direct' and level > 0:
                recorded = np.load(d['run']/'pre_step_velocities.npy', mmap_mode='r')
                velocity_check = float(np.max(np.abs(recorded-v[:-1])))
                assert velocity_check < 1e-10
            kinetic.append(K); centers.append(cm); kinetic_com.append(Kcm); kinetic_relative.append(Krel)
            checks.append(dict(method=method,level=level,mass_kg=float(mass.sum()),
                               decomposition_max_error_J=float(np.max(np.abs(K-Kcm-Krel))),
                               direct_recorded_velocity_max_error_m_s=velocity_check))
        K, cm, Kcm, Krel = map(np.array, [kinetic,centers,kinetic_com,kinetic_relative])
        error = np.linalg.norm(cm-cm[9],axis=2)*1000
        deltaK = np.abs(K-K[9])
        assert np.all(error[9] == 0) and np.all(deltaK[9] == 0)
        data[method] = dict(K=K, cm=cm, Kcm=Kcm, Krel=Krel, error=error)
        for level in range(10):
            peak = 1+int(np.argmax(K[level,1:]))
            row = dict(method=method,level=level,mean_kinetic_J=float(K[level,1:].mean()),
                       max_kinetic_J=float(K[level,peak]),peak_kinetic_time_s=float(TIMES[peak]),
                       mean_COM_error_to_L9_mm=float(error[level,1:].mean()),
                       max_COM_error_to_L9_mm=float(error[level,1:].max()),
                       initial_COM_error_to_L9_mm=float(error[level,0]),
                       mean_abs_kinetic_gap_to_L9_J=float(deltaK[level,1:].mean()),
                       rms_kinetic_gap_to_L9_J=float(np.sqrt(np.mean(deltaK[level,1:]**2))),
                       mean_COM_translation_kinetic_J=float(Kcm[level,1:].mean()),
                       mean_relative_motion_kinetic_J=float(Krel[level,1:].mean()),
                       peak_relative_motion_fraction=float(Krel[level,peak]/K[level,peak]))
            summaries.append(row)
            for frame,t in enumerate(TIMES):
                per_frame.append(dict(method=method,level=level,frame=frame,time_s=float(t),
                                      kinetic_J=float(K[level,frame]),COM_x_m=float(cm[level,frame,0]),
                                      COM_y_m=float(cm[level,frame,1]),COM_z_m=float(cm[level,frame,2]),
                                      COM_error_to_L9_mm=float(error[level,frame]),
                                      kinetic_abs_gap_to_L9_J=float(deltaK[level,frame]),
                                      COM_translation_kinetic_J=float(Kcm[level,frame]),
                                      relative_motion_kinetic_J=float(Krel[level,frame])))
        print(method, 'mean_K_J', np.round(K[:,1:].mean(1),5), flush=True)
        print(method, 'mean_COM_error_mm', np.round(error[:9,1:].mean(1),3), flush=True)
    return data, summaries, per_frame, checks


def plots(data):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = plt.get_cmap('turbo')(np.linspace(.05,.94,10))
    colors[0] = [0.15,.20,.26,1]
    bg='#f5f7fb'
    fig,axes=plt.subplots(2,2,figsize=(13,8.4),dpi=170,facecolor=bg)
    panels=[('semi',range(10),'Semi-diagonal · 全部十层'),('semi',range(7),'Semi-diagonal · L0–L6 放大'),
            ('velpro',range(10),'VelPro · 全部十层'),('direct',range(10),'Direct · 全部十层（L0 为共享参照）')]
    zoom_max=1.10*max(data['semi']['K'][:7].max(),data['velpro']['K'].max(),data['direct']['K'].max())
    for i,(ax,(method,levels,title)) in enumerate(zip(axes.flat,panels)):
        ax.set_facecolor(bg)
        for k in levels:
            ax.plot(TIMES,data[method]['K'][k],color=colors[k],lw=1.2 if k not in [0,9] else 1.8,
                    ls='--' if k==0 else '-',label=f'L{k}')
        ax.set_title(title,loc='left',weight='bold');ax.set_xlabel('时间（s）');ax.set_ylabel('动能（J）')
        ax.set_xlim(0,8);ax.set_ylim(0,data['semi']['K'].max()*1.08 if i==0 else zoom_max);ax.grid(alpha=.18)
        assert ax.get_ylim()[1] >= max(data[method]['K'][k].max() for k in levels)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=10,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle('按 PD Fig.18：各层动能随时间变化',fontsize=19,weight='bold',y=.995)
    fig.text(.06,.018,'注意纵轴：左上显示 Semi 的完整峰值；其余三图使用相同的放大尺度。速度由本层实际位置差分计算。',color='#526779',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.90));fig.savefig(OUT/'kinetic_energy.png');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.9),dpi=170,facecolor=bg)
    upper=max(d['error'].max() for d in data.values())*1.07
    for ax,method in zip(axes,METHODS):
        ax.set_facecolor(bg)
        for k in range(9):
            ax.plot(TIMES,data[method]['error'][k],color=colors[k],lw=1.1,label=f'L{k}–L9',ls='--' if k==0 else '-')
        ax.set_title(METHODS[method],loc='left',weight='bold');ax.set_xlabel('时间（s）')
        ax.set_ylabel('质心距离（mm）');ax.set_ylim(0,upper);ax.set_xlim(0,8);ax.grid(alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=9,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False,fontsize=10)
    fig.suptitle('按 PD Fig.19：各层相对本方法 L9 的质心轨迹误差',fontsize=19,weight='bold',y=.995)
    fig.text(.04,.02,'三个面板采用相同纵轴。单片旗帜只有一个物体，因此指标就是两层质心的欧氏距离；无对齐、无初态误差扣除。',fontsize=10,color='#526779')
    fig.tight_layout(rect=(0,.06,1,.84));fig.savefig(OUT/'center_of_mass_errors.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=170,facecolor=bg)
    for ax in axes:ax.set_facecolor(bg);ax.grid(alpha=.18)
    for method in METHODS:
        axes[0].plot(range(10),data[method]['K'][:,1:].mean(1),'o-',label=METHODS[method],color=COLORS[method])
        axes[1].plot(range(9),data[method]['error'][:9,1:].mean(1),'o-',label=METHODS[method],color=COLORS[method])
    axes[0].set_title('各层时间平均动能');axes[0].set_ylabel('J');axes[0].set_xticks(range(10),[f'L{k}' for k in range(10)])
    axes[1].set_title('各层对 L9 的时间平均质心误差');axes[1].set_ylabel('mm');axes[1].set_xticks(range(9),[f'L{k}' for k in range(9)])
    for ax in axes:ax.set_ylim(bottom=0);ax.legend(frameon=False)
    fig.suptitle('整体运动与局部运动需要分别检查',fontsize=17,weight='bold')
    fig.tight_layout();fig.savefig(OUT/'summary.png');plt.close(fig)


def html_report(rows):
    table=''
    for key,title,unit in [('mean_kinetic_J','各层平均动能','J'),('mean_COM_error_to_L9_mm','各层相对 L9 的平均质心误差','mm'),
                           ('mean_abs_kinetic_gap_to_L9_J','各层相对 L9 的平均动能绝对差（附加统计）','J')]:
        table+=f'<h2>{title}（{unit}）</h2><table><tr><th>层级</th><th>Semi</th><th>VelPro</th><th>Direct</th></tr>'
        for k in range(10):
            vals=[next(r[key] for r in rows if r['method']==m and r['level']==k) for m in METHODS]
            table+=f'<tr><td>L{k}</td>'+''.join(f'<td>{v:.4f}</td>' for v in vals)+'</tr>'
        table+='</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>旗帜 · PD Fig.18 与 Fig.19 指标</title>
<style>body{max-width:1250px;margin:32px auto;padding:0 24px;font:17px/1.7 system-ui;background:#f5f7fb;color:#23374b}img{width:100%;border-radius:10px}table{width:100%;border-collapse:collapse;background:white}th,td{padding:8px 16px;text-align:right;border-bottom:1px solid #dce4ed}th:first-child,td:first-child{text-align:left}code{background:#e5ebf3;padding:3px 6px}a{color:#3e5d9e}</style>
<h1>旗帜 · 按原始 PD Fig.18 / Fig.19 检查层间一致性</h1>
<p>同一组已完成的 L0–L9 旗帜轨迹：8 秒，201 个状态，dt = 0.04 s。不重新仿真。平均值取第 1–200 步，不包括初始静止状态。</p>
<p><b>主要结果：</b>Semi 高层动能显著增加，而质心偏离相对有限；VelPro 的各层动能更接近。相对各自 L9，VelPro 的质心误差随层级接近而逐步减小，Direct 则并不单调。单看质心会遗漏局部运动差异。</p>
<h2>Fig.18 方法：动能曲线</h2><p><code>v_l(t) = [x_l(t) − x_l(t−1)] / dt；K_l(t) = ½ Σ_i m_li |v_li(t)|²</code>。初始速度为零。使用本层自身的实际运动速度，不使用 Semi 或 VelPro 注入的预测速度。</p>
<img src="kinetic_energy.png" alt="各层动能曲线">
<h2>Fig.19 方法：质心轨迹误差</h2><p><code>c_l(t) = Σ_i m_li x_li(t) / Σ_i m_li；E_l,9(t) = |c_l(t) − c_9(t)|</code>。原论文对多个物体的质心距离求和，本实验只有一片旗帜，因此只有一个距离项。各方法分别以自身最细层 L9 为参照，L9 的误差为零。</p>
<img src="center_of_mass_errors.png" alt="各层相对 L9 的质心误差"><img src="summary.png" alt="动能和质心误差的时间均值">
TABLES
<h2>计算范围与解释</h2>
<ul><li>质量使用各层静止网格的三角形面积集总：密度 200 kg/m³，厚度 0.4 mm，各层总质量均为 0.12 kg。固定点保留在质量和质心计算中，其速度为零。</li>
<li>Direct 的 L1–L9 是各层独立 native 仿真；L0 沿用此次实验的共享粗层（含细层弹性代理）。所以 Direct 的 L0 不是另算的 native Direct L0，相关曲线用虚线标记，解读时应单独看待。</li>
<li>无刚体对齐、无相位匹配、无初始误差扣除。质心接近不能保证局部形状或褶皱接近；动能接近也不能单独证明轨迹相同。</li>
<li>动能曲线检查的是层间运动强度是否一致，不是能量守恒测试。旗帜持续受到风力和重力作用。</li>
<li>CSV 中额外给出整体质心平移动能及相对质心运动动能，用于定位高层动能增加；后者包括整体旋转和变形，不等同于弹性应变能或纯褶皱能量。这一分解不是 PD Fig.18/19 另行定义的指标。</li>
<li>没有通用的合格阈值；这些结果支持定位异常，不单独证明或否定实现正确性。</li></ul>
<p><a href="per_frame.csv">逐帧动能、质心坐标和误差 CSV</a> · <a href="summary.csv">逐层汇总 CSV</a> · <a href="verified.json">验证与源轨迹哈希</a> · <a href="../paper_consistency/index.html">PD++ 公式（13）的几何一致性报告</a></p></html>'''.replace('TABLES',table)
    (OUT/'index.html').write_text(page,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    chains,_=load_sources()
    data,rows,per_frame,checks=calculate(chains)
    write_csv('summary.csv',rows);write_csv('per_frame.csv',per_frame)
    summary={}
    for method in METHODS:
        d=data[method]
        summary[method]=dict(mean_K_L9_over_L0=float(d['K'][9,1:].mean()/d['K'][0,1:].mean()),
                             mean_COM_error_L0_to_L9_mm=float(d['error'][0,1:].mean()),
                             mean_COM_error_L8_to_L9_mm=float(d['error'][8,1:].mean()))
    manifest=dict(completed=True,method='PD2024 Section5.2, Fig18 kinetic energy; Fig19 summed object COM Euclidean distances',
                  number_of_objects=1,dt_s=.04,frames=201,duration_s=8,statistics_frames='1..200',
                  velocity='own-level backward position differences; initial velocity zero; NOT injected predictor velocity',
                  reference='own-method L9; no spatial or temporal alignment; no initial-error subtraction',
                  mass='rest triangle lumped mass, rho=200 kg/m3, thickness=0.0004m; includes fixed vertices',
                  direct_L0_note='Shared original proxy-based L0, not native Direct L0',
                  simulation_rerun=False,checks=checks,overview=summary,stats=rows,
                  source_runs={m:[dict(level=k,run=str(d['run']),positions_sha256=d['sha256'],mesh_sha256=d['mesh_sha256']) for k,d in enumerate(c)] for m,c in chains.items()},
                  script_sha256=digest(Path(__file__)))
    (OUT/'verified.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    plots(data);html_report(rows)
    print(json.dumps(summary,indent=2),flush=True)
    print('REPORT',OUT/'index.html',flush=True)


if __name__=='__main__':
    main()
