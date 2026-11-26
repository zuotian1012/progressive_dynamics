"""Build a portable, local results viewer from completed experiment evidence."""
import argparse
import html
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sections = []
    for name, branch, count, duration in [
        ("旗帜：10 层、20 秒", "flag_budget1000", 10, "20 秒"),
        ("兔子披布：8 层、接触与滑落", "bunny_event_window", 8, "1 秒接触窗口"),
    ]:
        chain = read(root / branch / "chain.json")
        paths = [Path(item["run"]) for item in chain["chain"]]
        if "failed_level" in chain:
            paths.append(root / branch / f'L{chain["failed_level"]}')
        rows = []
        for path in paths:
            meta = read(path / "run.json")
            steps = min(meta["completed_steps"], chain["requested_steps"])
            records = read(path / "steps.json")[:steps]
            newton = sum(bool(item["native"]["converged"]) for item in records)
            preview = sum(bool(item["native"].get("preview_small_step") or
                               item["native"].get("preview_non_descent")) for item in records)
            status = "窗口完成" if steps == chain["requested_steps"] else meta["status"]
            status = {'running':'计算中','interrupted_process_exit':'进程退出，原因待查','failed':'未通过验收'}.get(status,status)
            audit = read(path / "audit.json") if (path / "audit.json").exists() else {}
            checked = "通过（抽样碰撞）" if audit.get("accepted_prefix_checks_passed") else "见原始记录"
            relative = path.relative_to(root).as_posix()
            evidence = f'<a href="{relative}/run.json">记录</a>'
            if (path / 'audit.json').exists():
                evidence += f' · <a href="{relative}/audit.json">检查</a>'
            rows.append(f'<tr><td>L{meta["level"]}</td><td>{meta["vertices"]:,}</td>'
                        f'<td>{steps * meta["dt"]:g} s</td><td>{newton} / {preview}</td>'
                        f'<td>{html.escape(status)}</td><td>{checked}</td>'
                        f'<td>{evidence}</td></tr>')
        comp = f"{branch}/comparison"
        clip='contact_closeup' if (root/comp/'contact_closeup.mp4').exists() else 'levels'
        poster=f'{clip}_poster.png' if (root/comp/f'{clip}_poster.png').exists() else f'{clip}_last.png'
        video = (f'<video controls preload="metadata" poster="{comp}/{poster}">'
                 f'<source src="{comp}/{clip}.mp4" type="video/mp4"></video>'
                 f'<p class="small">同步相机与物理时间；字幕注明截取窗口和播放速度。'
                 f'<a href="{comp}/levels.mp4">完整对比窗口视频</a> · <a href="{comp}/{clip}.gif">GIF</a></p>')
        if not (root / comp / "levels.mp4").exists():
            video = '<p class="note">最终对比视频尚未生成。</p>'
        if (root / comp / "paper_frames_166_168.png").exists():
            video += (f'<details><summary>对应论文 Fig. 5 的第 166–168 帧布局</summary>'
                      f'<img src="{comp}/paper_frames_166_168.png" alt="旗帜十层相邻三帧">'
                      '<p>本图展示本地 VelPro 轨迹，不替代论文的 Semi / Full / VelPro 方法对照。</p></details>')
        chain_status={'running':'计算中','completed_hierarchy':'全部层级窗口完成','blocked_by_failed_level':'在未完成层中断'}.get(chain['status'],chain['status'])
        sections.append(f'<section><h2>{name}</h2><p>目标：{count} 层；本次对比窗口：{duration}。'
                        f'当前状态：{html.escape(chain_status)}。以下为实际动力学层，不将细层代理当作细网格仿真。</p>'
                        + video + '<div class="scroll"><table><thead><tr><th>层</th><th>顶点</th>'
                        '<th>已接受时间</th><th>Newton / 粗层停止步</th><th>状态</th><th>输出检查</th>'
                        '<th>证据</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
                        f'<details><summary>各层速度与几何差异</summary><img src="{comp}/level_metrics.png" '
                        'alt="实际轨迹的速度与相邻层 PSQ 几何差异"><p>这是几何诊断，不是论文 Eq.12 '
                        '的时间连续性指标，也不能单独证明 VelPro 的稳定性优势。</p></details>'
                        f'<details><summary>收敛、粗层停止与迭代次数</summary><img src="{comp}/solver_diagnostics.png" '
                        'alt="各层求解器迭代次数与停止原因"><p>迭代次数用于定位求解难点；共享 GPU 的耗时不用于性能排名。</p></details></section>')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PD++ 论文规模实验</title>
<style>
:root{color-scheme:light;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#172d3b;background:#edf2f5}
body{margin:0}main{max-width:1180px;margin:auto;padding:40px 24px 70px}h1{font-size:36px;margin:8px 0 16px}
h2{font-size:24px}p,li{line-height:1.8}a{color:#126585}section{background:white;padding:26px;border-radius:14px;margin:26px 0}
.eyebrow{color:#426476;font-weight:650;letter-spacing:.06em}.note{background:#fff2de;border-left:4px solid #bf7528;padding:16px 20px}
.small{font-size:13px;color:#526777}video,img{display:block;width:100%;height:auto;border-radius:8px}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px 9px;border-bottom:1px solid #dbe4e9;white-space:nowrap}
th{background:#f4f7f9}summary{cursor:pointer;padding:18px 0;color:#126585}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:700px){main{padding:20px 12px}section{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:28px}}
</style></head><body><main><div class="eyebrow">PROGRESSIVE DYNAMICS · 2026-11-17</div>
<h1>从论文场景出发，检查真实层级运动</h1>
<p>两个场景分别覆盖受外力驱动的长时间运动，以及薄布与复杂障碍物的接触和滑落。
材料汇集实际计算轨迹、网格规模、停止原因和失败证据。</p>
<div class="note"><strong>目前是论文规模重建实验，尚不能称为完整论文复现。</strong>
网格层数与目标规模已对齐；作者完整材料、外力与网格资产尚未恢复。粗层停止与 Newton 收敛分开统计，最细层禁止粗层预览停止。
所有运行关闭原生半隐式和强制更新。</div>
''' + ''.join(sections) + '''
<section><h2>失败案例同样值得展示</h2>
<p><strong>旗帜 L0 在外力继续变化时冻结。</strong>500 步记录完成，不等于 20 秒持续摆动成功。
后约 16 秒无运动，说明粗层停止与搜索方向仍需诊断。</p>
<img src="gamma_full/flag/L0/report/freeze_diagnostic.png" alt="旗帜 L0 外力变化与运动冻结的诊断图">
<p>原 300 次 Newton 预算下，旗帜 L3 第 271 步、兔子 L3 第 174 步被拒绝。增加预算的重跑保留为独立分支；
旗帜重跑在旧失败时刻之前已有轨迹偏差，不能仅据越过原失败点证明预算变化是原因。</p>
<p>兔子场景中的滑落是论文预期行为。细层采用前 1 秒接触窗口；粗层 4 秒记录保留。
原 L4 的 4 秒请求主动停止，仅提取已验收的前 100 步，未把原长测改称完成。</p></section>
<section><h2>规模与来源</h2><div class="grid"><div><img src="report/flag_hierarchy.png" alt="旗帜十层网格规模">
<p>旗帜：551 → 35,000 顶点，10 层；厚度 0.4 mm，最大拉伸 1.1；目标时长 20 秒。</p></div>
<div><img src="report/bunny_hierarchy.png" alt="兔子布料八层网格规模">
<p>兔子布料：1,443 → 90,000 顶点，8 层；厚度 0.07 mm，最大拉伸 1.068，摩擦 0.3，dt=0.01 s。</p></div></div>
<p>旗帜 dt=0.04 s 为推定；尺寸、初始摆位、E=30,000 Pa、ν=0.3、ρ=200 kg/m³、旗帜体加速度与 PSQ γ=10⁻⁸ 为本地选择。
原生收敛度量及接触/应变刚度更新与论文仍有差异。抽样无自交不替代连续碰撞与厚度间距的完整独立证明。</p>
<p>尚未完成同配置 Direct / Semi / Full / VelPro 的论文规模对照。部分运行共享 GPU，耗时不作为公平性能比较。</p>
<p><a href="https://pcs-sim.github.io/pd%2B%2B/PD%2B%2B_main.pdf">PD++ 原论文</a> ·
<a href="https://pcs-sim.github.io/psq/psq-combined.pdf">PSQ 原论文</a> ·
<a href="说明.txt">完整说明</a> · <a href="catalog.json">实验目录与审计</a></p></section>
</main></body></html>'''
    (root / "index.html").write_text(page, encoding="utf-8")
    print(root / "index.html")


if __name__ == "__main__":
    main()
