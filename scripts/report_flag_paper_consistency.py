"""Evaluate saved 8-second flag trajectories using PD++ Eq. (13).

Adjacent pairs use the published formula. Cross-level comparisons to L0 use
the pseudoinverse of the composed forward U, NOT a product of pseudoinverses.
No trajectory modification, alignment, pin reimposition, or initial subtraction.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.sparse import eye, load_npz
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from progressive.metrics import consistency_metrics

SUITE = ROOT / 'output/flag_to_L9_20261122'
OUT = SUITE / 'report/paper_consistency'
METHODS = {'semi': 'Semi-diagonal', 'velpro': 'VelPro', 'direct': 'Direct'}
COLORS = {'semi': '#d98a26', 'velpro': '#6256b3', 'direct': '#278c79'}
TIMES = np.arange(201) * .04


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_sources():
    index = json.loads((SUITE / 'index.json').read_text())
    assert json.loads((SUITE / 'summary.json').read_text())['complete']
    old = json.loads((SUITE / 'report/cross_level/verified.json').read_text())
    chains = {}
    for method in METHODS:
        chain = []
        for level in range(10):
            entry = index['L0'] if level == 0 else index['levels'][str(level)][method]
            root, run = Path(entry['root']), Path(entry['run'])
            meta = json.loads((run / 'run.json').read_text())
            scene = json.loads((root / 'scene.json').read_text())
            audit_path = root / 'report' / ('audit.json' if level == 0 else f'{run.name}/audit.json')
            audit = json.loads(audit_path.read_text())
            assert meta['status'] == 'completed' and meta['accepted_frames'] == 200
            assert meta['dt'] == .04 and meta['wind_scale'] == 3 and meta['pin_mode'] == 'corners'
            assert audit['all_native_steps_converged'] and audit['fixed_vertices'] == 2
            mesh_path = root / f'assets/mesh_{level}.npz'
            mesh = np.load(mesh_path)
            rest, faces = mesh['vertices'], mesh['faces']
            tri = rest[faces]
            area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
            mass = np.zeros(len(rest))
            np.add.at(mass, faces.ravel(), np.repeat(area * scene['density_kg_m3'] * scene['full_thickness_m'] / 3, 3))
            assert np.all(mass > 0)
            expected_mass = scene['width_m'] * scene['height_m'] * scene['density_kg_m3'] * scene['full_thickness_m']
            np.testing.assert_allclose(mass.sum(), expected_mass, rtol=1e-12)
            if 'total_mass_kg' in audit:
                np.testing.assert_allclose(mass.sum(), audit['total_mass_kg'], rtol=1e-12)
            x = np.load(run / 'positions.npy', mmap_mode='r')
            assert x.shape == (201, len(rest), 3) and np.isfinite(x).all()
            sha = digest(run / 'positions.npy')
            if method in old['source_runs']:
                assert sha == old['source_runs'][method][level]['positions_sha256']
            if level and method != 'direct':
                assert Path(meta['lower_run']).resolve() == chain[-1]['run'].resolve()
                assert meta['lower_positions_sha256'] == chain[-1]['sha256']
                assert meta['transfer_sha256'] == digest(root / f'assets/map_{level-1}_{level}.npz')
            if level and method == 'direct':
                assert not meta['coarse_trajectory_used'] and meta['elastic_model'] == 'native_current_level'
            chain.append(dict(root=root, run=run, x=x, rest=rest, mass=mass, sha256=sha,
                              mesh_sha256=digest(mesh_path), meta=meta))
        chains[method] = chain
    for level in range(10):
        for method in METHODS:
            assert chains[method][level]['mesh_sha256'] == chains['semi'][level]['mesh_sha256']
    print('Verified 28 unique saved trajectories; total mass =', chains['semi'][0]['mass'].sum(), flush=True)
    return chains, old


def evaluate(U, factor, low, high):
    mass = low['mass']
    raw, detail, fine_gap = [], [], []
    normal_residual = 0.
    for frame in range(201):
        fine = high['x'][frame]
        rhs = U.T @ fine
        restricted = factor.solve(rhs)
        delta = restricted - low['x'][frame]
        raw.append(float(np.sum(mass[:, None] * delta**2)))
        residual = fine - U @ restricted
        normal_residual = max(normal_residual, float(np.linalg.norm(U.T @ residual) / max(np.linalg.norm(rhs), 1e-30)))
        detail.append(float(np.sqrt(np.sum(high['mass'][:, None]*residual**2)/high['mass'].sum())*1000))
        gap = fine - U @ low['x'][frame]
        fine_gap.append(float(np.sqrt(np.sum(high['mass'][:, None]*gap**2)/high['mass'].sum())*1000))
    raw = np.array(raw)
    rms = np.sqrt(raw/mass.sum())*1000
    assert normal_residual < 1e-10
    return dict(raw=raw, rms=rms, detail=np.array(detail), fine_gap=np.array(fine_gap),
                normal_residual=normal_residual)


def summarize(result, method, kind, level, coarse_level, mass):
    r = result['rms']
    return dict(method=method, comparison=kind, fine_level=level, coarse_level=coarse_level,
                coarse_mass_kg=float(mass.sum()), mean_d_kg_m2=float(result['raw'][1:].mean()),
                mean_rms_mm=float(r[1:].mean()), time_rms_mm=float(np.sqrt(np.mean(r[1:]**2))),
                max_rms_mm=float(r[1:].max()), peak_time_s=float(TIMES[1+np.argmax(r[1:])]),
                final_rms_mm=float(r[-1]), initial_rms_mm=float(r[0]),
                mean_fine_space_gap_mm=float(result['fine_gap'][1:].mean()),
                mean_detail_residual_mm=float(result['detail'][1:].mean()),
                max_relative_normal_residual=result['normal_residual'])


def make_plots(results):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160, facecolor='#f5f7fb')
    for ax in axes.flat:
        ax.set_facecolor('#f5f7fb'); ax.grid(alpha=.18)
        ax.set_ylabel('归一化质量 RMS（mm）')
    for method in ['semi', 'velpro']:
        for col, kind in enumerate(['adjacent', 'to_L0']):
            curves = [results[(kind, k, method)]['rms'] for k in range(1, 10)]
            axes[0, col].plot(range(1, 10), [r[1:].mean() for r in curves], 'o-',
                              color=COLORS[method], label=METHODS[method], lw=2)
            axes[1, col].plot(TIMES, curves[-1], color=COLORS[method], label=METHODS[method], lw=1.8)
    axes[0, 0].set_title('相邻层误差 · 200 步平均', loc='left', weight='bold')
    axes[0, 1].set_title('各层对 L0 的误差 · 200 步平均', loc='left', weight='bold')
    axes[1, 0].set_title('L9 投影到 L8 · 随时间变化', loc='left', weight='bold')
    axes[1, 1].set_title('L9 投影到 L0 · 随时间变化', loc='left', weight='bold')
    for ax in axes[0]:
        ax.set_xticks(range(1, 10), [f'L{k}' for k in range(1, 10)]); ax.set_xlabel('细层'); ax.legend(frameon=False)
    for ax in axes[1]: ax.set_xlabel('物理时间（s）'); ax.set_xlim(0, 8)
    for ax in axes.flat:
        ax.set_ylim(bottom=0)
        assert ax.get_ylim()[1] >= max(np.max(line.get_ydata()) for line in ax.lines)
    fig.suptitle('旗帜层间一致性 · PD++ 公式（13）', fontsize=20, weight='bold', x=.07, ha='left')
    fig.text(.07, .025, '先将细层最小二乘投影回粗层，再比较位置。全体顶点参与；不做刚体对齐或初态差扣除。', color='#506477')
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(OUT/'paper_consistency.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    for col, kind in enumerate(['adjacent', 'to_L0']):
        for method in METHODS:
            means = [results[(kind, k, method)]['rms'][1:].mean() for k in range(1, 10)]
            axes[col].plot(range(1, 10), means, 'o-', color=COLORS[method], label=METHODS[method])
        axes[col].set_title('相邻层' if col == 0 else '各层投影到 L0')
        axes[col].set_xticks(range(1,10), [f'L{k}' for k in range(1,10)])
        axes[col].set_ylabel('时间平均 RMS（mm）'); axes[col].set_ylim(bottom=0); axes[col].grid(alpha=.2); axes[col].legend()
    fig.suptitle('包含 Direct 的参照比较（共享原有 L0）', fontsize=16)
    fig.tight_layout(); fig.savefig(OUT/'with_direct.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    colors = plt.get_cmap('viridis')(np.linspace(.05,.9,9))
    for row, method in enumerate(['semi', 'velpro']):
        for col, kind in enumerate(['adjacent', 'to_L0']):
            ax=axes[row,col]
            for k in range(1,10):
                ax.plot(TIMES,results[(kind,k,method)]['raw'],color=colors[k-1],label=f'L{k}',lw=1)
            ax.set_title(METHODS[method] + (' / 相邻层' if col==0 else ' / 投影到 L0'))
            ax.set_xlabel('物理时间（s）'); ax.set_ylabel('原始 d(t)（kg·m²）'); ax.set_xlim(0,8); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[0,1].legend(ncol=3,fontsize=8)
    fig.suptitle('公式（13）原始质量加权平方误差 · 全层逐帧曲线',fontsize=16)
    fig.tight_layout(); fig.savefig(OUT/'raw_eq13.png'); plt.close(fig)


def make_report(rows):
    tables = ''
    for kind, title in [('adjacent', '相邻层'), ('to_L0', '各层对 L0')]:
        tables += f'<h2>{title}：时间平均 RMS（mm）</h2><table><tr><th>细层 → 粗层</th><th>Semi</th><th>VelPro</th><th>Direct</th></tr>'
        for k in range(1,10):
            values = [next(r for r in rows if r['comparison']==kind and r['fine_level']==k and r['method']==m)['mean_rms_mm'] for m in METHODS]
            tables += f'<tr><td>L{k} → L{k-1 if kind=="adjacent" else 0}</td>' + ''.join(f'<td>{v:.3f}</td>' for v in values) + '</tr>'
        tables += '</table>'
    summary = { (r['comparison'], r['method']): r for r in rows if r['fine_level']==9 }
    findings = '<p><b>主要结果：</b>相邻高层的一致性以 VelPro 更好；但对共同 L0 的时间平均偏离以 Semi 更小，不能概括为某一种方法全面更一致。</p>'
    findings += '<ul>'
    for method in ['semi','velpro']:
        a,b=summary[('adjacent',method)],summary[('to_L0',method)]
        findings += f'<li>{METHODS[method]}：L9→L8 平均 {a["mean_rms_mm"]:.2f} mm，最大 {a["max_rms_mm"]:.2f} mm（{a["peak_time_s"]:.2f} s）；L9→L0 平均 {b["mean_rms_mm"]:.2f} mm。</li>'
    findings += '</ul><p>与旧指标相比，相邻高层数值变化很小：这组实验的大部分相邻层位置差在粗层空间中仍然可见，不能主要归因为投影后被滤除的细节。两种指标的权重和比较空间不同，数值不要求单调减小。</p>'
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>旗帜 · PD++ 公式（13）</title>
<style>body{max-width:1200px;margin:36px auto;padding:0 24px;font:17px/1.7 system-ui;background:#f5f7fb;color:#23374b}img{width:100%;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px 18px;text-align:right;border-bottom:1px solid #dde4ed}td:first-child,th:first-child{text-align:left}code{background:#e5ebf4;padding:3px 8px}a{color:#4160a6}</style>
<h1>旗帜层间一致性 · PD++ 公式（13）</h1>
<p>使用此前 L0–L9 对比的原始 8 秒轨迹，dt = 0.04 s，201 个状态。未重新仿真。均值使用第 1–200 步，初态保留在曲线和 CSV 中。</p>
FINDINGS
<p><code>R = (UᵀU)⁻¹Uᵀ; d(t) = (R x_f − x_c)ᵀ M_c (R x_f − x_c)</code><br>
毫米 RMS = 1000 √(d / Σm_c)。原始 d 的单位是 kg·m²；毫米 RMS 是便于阅读的归一化换算。表格是逐帧 RMS 的算术平均，不是均方误差开根；两种时间统计均保存在数据中。</p>
<p>U 是保存的线性内在延拓矩阵；投影使用未加权的最小二乘逆，随后才用粗层质量加权。质量来自静止三角形面积，密度 200 kg/m³，厚度 0.4 mm，总质量 0.12 kg。包括固定点；不做坐标对齐、不扣除初始误差、不把投影后固定点重新钉回去。</p>
<p>相邻层严格使用公式（13）。各层对 L0：先复合前向矩阵 U₀→k = Uₖ₋₁→ₖ … U₀→₁，再对复合矩阵求最小二乘逆；没有逐级相乘伪逆。这是将同一指标用于跨多层比较的明确实现。</p>
<img src="paper_consistency.png" alt="Semi 和 VelPro 的论文指标比较">
TABLES
<h2>Direct 参照</h2><p>L1–L9 Direct 为各层独立的 native 仿真；L0 沿用实验中共享的原始粗层驱动（含细层弹性代理），并非另做一条 native Direct L0。因此尤其 L1→L0 以及对 L0 的 Direct 误差须按此参照含义解读。</p><img src="with_direct.png" alt="含 Direct 的比较">
<h2>公式（13）的原始数值</h2><img src="raw_eq13.png" alt="原始质量加权平方误差">
<p>数值小表示粗尺度位置更接近，不单独证明物理准确性，也不评价时间连续性。论文没有统一的毫米合格阈值；应结合新增褶皱与动画形态判断。细空间差与投影残差为额外诊断，并非论文另行定义的丰富度指标。</p>
<p><a href="summary.csv">汇总 CSV</a> · <a href="per_frame.csv">逐帧原始 d 与归一化 RMS</a> · <a href="verified.json">计算说明、校验和源轨迹哈希</a> · <a href="../cross_level/index.html">此前十层动画与旧 RMS</a></p></html>'''.replace('TABLES', tables).replace('FINDINGS', findings)
    (OUT/'index.html').write_text(html,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    chains, old = load_sources()
    results, rows, per_frame, checks = {}, [], [], []
    cumulative = eye(len(chains['semi'][0]['rest']), format='csr')
    for level in range(1,10):
        high = chains['semi'][level]
        map_path = high['root']/f'assets/map_{level-1}_{level}.npz'
        U = load_npz(map_path).astype(np.float64).tocsr()
        cumulative = (U @ cumulative).tocsr()
        for kind, operator, coarse_level in [('adjacent',U,level-1),('to_L0',cumulative,0)]:
            low = chains['semi'][coarse_level]
            np.testing.assert_allclose(operator @ low['rest'], high['rest'], atol=2e-12, rtol=0)
            np.testing.assert_allclose(np.asarray(operator.sum(axis=1)).ravel(), 1, atol=2e-12, rtol=0)
            factor = splu((operator.T @ operator).tocsc())
            rng = np.random.default_rng(1729)
            probe = rng.normal(size=(operator.shape[1],3))
            recovered = factor.solve(operator.T @ (operator @ probe))
            inverse_error = float(np.max(np.abs(recovered-probe)))
            assert inverse_error < 1e-9
            checks.append(dict(comparison=kind, level=level, left_inverse_max_error=inverse_error,
                               map_sha256=digest(map_path), shape=list(operator.shape)))
            for method in METHODS:
                result = evaluate(operator, factor, chains[method][coarse_level], chains[method][level])
                results[(kind,level,method)] = result
                row = summarize(result,method,kind,level,coarse_level,low['mass']);rows.append(row)
                if kind=='adjacent' and method in ['semi','velpro']:
                    np.testing.assert_allclose(row['mean_fine_space_gap_mm'],old['stats'][method][level-1]['mean_adjacent_rms_mm'],atol=1e-9,rtol=1e-9)
                if level==1 and kind=='adjacent':
                    reference=consistency_metrics(chains[method][0]['x'][100],chains[method][1]['x'][100],SimpleNamespace(U=operator),low['mass'],high['mass'])
                    np.testing.assert_allclose(result['raw'][100],reference['coarse_mass_squared'],rtol=1e-12,atol=1e-20)
                for f,t in enumerate(TIMES):
                    per_frame.append(dict(method=method,comparison=kind,fine_level=level,coarse_level=coarse_level,
                                          frame=f,time_s=float(t),d_kg_m2=float(result['raw'][f]),rms_mm=float(result['rms'][f]),
                                          fine_space_gap_mm=float(result['fine_gap'][f]),detail_residual_mm=float(result['detail'][f])))
            print(kind,level, {m:round(results[(kind,level,m)]['rms'][1:].mean(),3) for m in METHODS},flush=True)
    for name, data in [('summary.csv',rows),('per_frame.csv',per_frame)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    manifest=dict(completed=True,formula='PD++ (2025), Section 7, Equation 13',
                  restriction='R = solve(U.T @ U, U.T); unweighted least squares',
                  cross_level_operator='pseudoinverse of composed forward U; NOT product of adjacent pseudoinverses',
                  mass='coarse rest-area lumped physical mass; rho=200 kg/m^3, thickness=0.0004 m; all vertices including pins',
                  raw_units='kg m^2',normalized_rms='1000 * sqrt(d / sum(coarse_mass)) [mm]',
                  time_mean='arithmetic mean of per-frame RMS over frames 1..200; time_rms is also saved',
                  alignment='none',initial_error_subtracted=False,projection_pins_reimposed=False,
                  simulation_rerun=False,frames=201,dt=.04,duration_s=8,
                  direct_L0_note='Shared original proxy-based L0, not independent native Direct L0',
                  old_fine_space_rms_reproduced=True,reference_function_checked=True,
                  max_relative_normal_residual=max(r['max_relative_normal_residual'] for r in rows),
                  operator_checks=checks,stats=rows,
                  source_runs={m:[dict(level=k,run=str(d['run']),positions_sha256=d['sha256'],mesh_sha256=d['mesh_sha256'],mass_kg=float(d['mass'].sum())) for k,d in enumerate(c)] for m,c in chains.items()},
                  script_sha256=digest(Path(__file__)))
    (OUT/'verified.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    make_plots(results);make_report(rows)
    print('REPORT',OUT/'index.html',flush=True)


if __name__=='__main__':
    main()
