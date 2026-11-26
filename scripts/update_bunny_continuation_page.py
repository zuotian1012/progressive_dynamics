"""Render the recorded, one-time L6/L7 acceptance snapshot in the existing demo."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'output/bunny_newton_converged_mu03_velpro_75_20261125/demo'
d = json.loads((SITE/'summary.json').read_text(encoding='utf-8'))
c = d['continuation']
c['runs']['7']['note'] = 'run.json 仍写 running，但验收时未发现对应进程；日志无明确异常结尾，停止原因未确定。64 个已接受步均记录为收敛，尚未完成整层审计。'
c['timing_definition'] = 'wall_seconds 为仿真记录的整段计时；completed_step_seconds 为已完成时间步 seconds 累计。均不包含后续审计。未完成层不推算最终耗时，并发 GPU 耗时不是独占基准。'
c['scope'] = 'L6/L7 状态与耗时验收；原 L0–L5 动画、指标和比较保留。'
def duration(sec):
    if sec is None:return '尚无最终记录'
    t=round(sec)
    return f'{t//3600} 小时 {(t%3600)//60:02d} 分 {t%60:02d} 秒'
rows=''
for level,r in c['runs'].items():
    status='完成，审计通过' if r['audit_passed'] else '未完成；未发现运行进程'
    rows+=f'<tr><td><b>L{level}</b></td><td>{status}</td><td>{r["completed_steps"]} / 75</td><td>{r["converged_steps"]} / {r["preview_steps"]}</td><td>{duration(r["wall_seconds"])}</td><td>{duration(r["completed_step_seconds"])}</td></tr>'
section=f'''<!-- continuation-check:start -->
<section id="continuation"><div class="section-head"><div class="eyebrow">L6 / L7 · SCHEDULED ACCEPTANCE</div><h2>L6 / L7：11:00 验收结果</h2><p>北京时间 2026-11-26 11:00 定时验收。以下为该次检查快照；上方动画与偏差曲线仍为 L0–L5。</p></div>
<div class="table-wrap"><table><thead><tr><th>层级</th><th>验收状态</th><th>完成步数</th><th>收敛 / 退出</th><th>仿真整段计时 wall_seconds</th><th>已完成步累计 seconds</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="method-note"><h3>L6 已完成；L7 尚不能验收为完成</h3><p>L6 的 75 步全部收敛、76 个保存状态通过检查，父层轨迹哈希、线搜索和补救模式审计通过；补救触发 0 步。</p>
<p><strong>L7 只完成 64 步。</strong>其 run.json 仍标为 running，但检查时未发现对应仿真进程，日志停在 07:53 左右、第 65 步 Newton 第 3 次迭代的线性系统求解阶段。没有明确异常结尾，停止原因未确定；未重启或恢复任何仿真。</p>
<p>L7 的 4 小时 04 分 04 秒仅是已完成 64 步累计耗时，不包含未完成的第 65 步，也不是最终耗时。两项时间均不包含后续审计；GPU 并发占用未受控，不能作为独占性能基准。</p><p class="caption">快照生成：{c['checked_at']}。详细秒数与核验信息已写入完整结果 JSON。</p></div></section>
<!-- continuation-check:end -->'''
p=SITE/'index.html';s=p.read_text(encoding='utf-8')
start=s.index('<!-- continuation-check:start -->');end=s.index('<!-- continuation-check:end -->')+len('<!-- continuation-check:end -->')
s=s[:start]+section+s[end:];assert '\ufffd' not in s
p.write_text(s,encoding='utf-8')
encoded=json.dumps(d,ensure_ascii=False,indent=2);assert '\ufffd' not in encoded
(SITE/'summary.json').write_text(encoded,encoding='utf-8')
(SITE/'data.js').write_text('window.BUNNY_DATA='+json.dumps(d,ensure_ascii=False,separators=(',',':'))+';',encoding='utf-8')
(SITE/'continuation_acceptance.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
print('Acceptance section and data updated; UTF-8 verified.')
