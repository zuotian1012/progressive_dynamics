"""Extend the existing local viewer to the audited L6 trajectory."""
from pathlib import Path
import re

SITE=Path(__file__).resolve().parents[1]/'output/bunny_newton_converged_mu03_velpro_75_20261125/demo'
p=SITE/'app.js';s=p.read_text(encoding='utf-8')
s=s.replace("levels=[0,1,2,3,4,5],colors=['','#1675a8','#d77527','#8256b0','#258568','#ba5275']", "levels=Object.keys(D.runs.improved).map(Number),sharedLevels=levels.filter(l=>D.runs.original[l]),colors=['','#1675a8','#d77527','#8256b0','#258568','#ba5275','#425bb5']")
s=s.replace("view:'all',strategy:'improved',leftStrategy:'original',rightStrategy:'improved',leftLevel:5,rightLevel:5,basis:'parent',chartLevel:'5'", "view:'pair',strategy:'improved',leftStrategy:'improved',rightStrategy:'improved',leftLevel:5,rightLevel:6,basis:'parent',chartLevel:'6'")
s=s.replace("S.view==='all'?levels.map", "S.view==='all'?Object.keys(D.runs[S.strategy]).map(Number).map")
s=s.replace("[1,2,3,4,5].map(l=>({name:", "levels.filter(l=>l>0).map(l=>({name:")
s=s.replace("['original','improved'].map(m=>({name:", "['original','improved'].filter(m=>D.metrics[m][S.chartLevel]).map(m=>({name:")
# Existing L0-L5 findings remain scoped to the matched experiment.
s=s.replace("mins=levels.map", "mins=sharedLevels.map").replace("oldSeconds=levels.reduce", "oldSeconds=sharedLevels.reduce")
s=s.replace("const a=D.runs.original[l],b=D.runs.improved[l];return", "const a=D.runs.original[l],b=D.runs.improved[l];return")
s=s.replace("${a.converged} / ${a.preview}", "${a?`${a.converged} / ${a.preview}`:'暂无数据'}")
s=s.replace("${(a.simulation_seconds/60).toFixed(2)} 分", "${a?(a.simulation_seconds/60).toFixed(2)+' 分':'暂无数据'}")
s=s.replace("[1,2,3,4,5].map(l=>{const a=D.metrics.original[l][S.basis].summary,b=", "levels.filter(l=>l>0).map(l=>{const a=D.metrics.original[l]?.[S.basis].summary,b=")
s=s.replace("mean=pct(b.mean_mm,a.mean_mm),final=pct(b.final_mm,a.final_mm);return", "mean=a?pct(b.mean_mm,a.mean_mm):0,final=a?pct(b.final_mm,a.final_mm):0;if(!a)return `<tr><td><b>L${l}</b></td><td>暂无数据</td><td>${fmt(b.mean_mm)}</td><td>—</td><td>暂无数据</td><td>${fmt(b.final_mm)}</td><td>—</td><td>${fmt(b.peak_mm)}</td></tr>`;return")
start=s.index("for(const id of ['leftLevel','rightLevel'])")
end=s.index("$('play').onclick",start)
s=s[:start]+'''function syncControls(){
  for(const side of ['left','right']){const mode=S[side+'Strategy'],available=Object.keys(D.runs[mode]).map(Number);if(!available.includes(S[side+'Level']))S[side+'Level']=available[available.length-1];$(side+'Level').innerHTML=available.map(l=>`<option value="${l}">L${l}</option>`).join('');$(side+'Level').value=S[side+'Level'];$(side+'Strategy').value=mode;}
  $('view').value=S.view;$('chartLevel').value=S.chartLevel;
}
for(const id of ['view','strategy','leftStrategy','rightStrategy','leftLevel','rightLevel'])$(id).onchange=e=>{pause();S[id]=id.endsWith('Level')?Number(e.target.value):e.target.value;syncControls();buildTiles();show(S.frame);};
syncControls();
'''+s[end:]
s=s.replace("extraAnalysis();", """extraAnalysis();
const l6=D.metrics.improved[6],l6run=D.runs.improved[6],card=document.createElement('article');
card.innerHTML=`<h3>L6：全部收敛后的层间偏差</h3><span class="large">${fmt(l6.parent.summary.mean_mm)} mm</span><p>L6 对 L5 的全程平均 RMS；末帧 ${fmt(l6.parent.summary.final_mm)} mm，峰值 ${fmt(l6.parent.summary.peak_mm)} mm。相对 L0 的末帧 RMS 为 ${fmt(l6.l0.summary.final_mm)} mm。75 步全部收敛，补救触发 0 步，仿真整段耗时 ${(l6run.full_run_wall_seconds/3600).toFixed(2)} 小时。原策略 L6 尚无完整结果，因此不计算该层两策略改善百分比。</p>`;
$('findings').prepend(card);
""")
p.write_text(s,encoding='utf-8')
p=SITE/'index.html';s=p.read_text(encoding='utf-8')
s=s.replace('L0–L5','L0–L6').replace('450 / 450','525 / 525').replace('<strong>456</strong>','<strong>532</strong>')
s=s.replace('同一时刻，看六层的形变','同步比较 L5 与 L6 的轨迹').replace('六层总览','全部已完成层总览').replace('改进版 L1–L5 总览','改进版 L1–L6 总览')
s=s.replace('<option value="5" selected>L5 · 两策略对照</option>','<option value="5">L5 · 两策略对照</option><option value="6" selected>L6 · 改进版</option>')
s=s.replace('所有 450 步','所有 525 步').replace('全部 456 个','全部 532 个')
s=s.replace('原策略取已完成的', 'L0–L5 提供两策略对照；L6 提供改进版及其层间偏差。原策略取已完成的')
s=s.replace('上方动画与偏差曲线仍为 L0–L6。','此处保留历史快照。上方已补入 L6 动画和指标；L7 正在另一个目录重跑，此快照不表示当前重跑进度。')
s=s.replace('停止原因未确定；未重启或恢复任何仿真。','该次检查时原因未确定；后续已定位到 GPU/驱动异常并开始独立重跑。')
s=s.replace('每层都补救，同时摩擦', '每层都补救，同时摩擦')
s=s.replace('所有画面使用相同、固定的相机和尺度。','默认展示改进版 L5 与 L6。改进版支持 L0–L6，原策略支持 L0–L5。所有画面使用相同、固定的相机和尺度。')
p.write_text(s,encoding='utf-8')
print('Viewer extended to L6; original strategy remains L0-L5.')
