"""Analyze observational PSQ directional probes and ABD gradient decomposition."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
meta=json.loads((root/'run.json').read_text());out=root/'diagnosis';out.mkdir(exist_ok=True)
probes={};restore=[];components={}
probe_re=re.compile(r'PSQ_CONSISTENCY frame=(\d+) newton=(\d+) frozen=(\d+) h=(\S+) predicted=(\S+) measured=(\S+) base=(\S+) trial=(\S+)')
restore_re=re.compile(r'PSQ_CONSISTENCY_RESTORE frame=(\d+) newton=(\d+) alpha=(\S+) before=(\S+) after=(\S+) error=(\S+)')
component_re=re.compile(r'PSQ_ABD_COMPONENT frame=(\d+) newton=(\d+) body=(\d+) stage=(\S+) g=\[([^\]]+)\]')
with root.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as stream:
    for line in stream:
        match=probe_re.search(line)
        if match:
            f,it,frozen=map(int,match.groups()[:3]);h,predicted,measured,base,trial=map(float,match.groups()[3:])
            probes.setdefault((f,it),[]).append(dict(frozen=bool(frozen),h=h,predicted=predicted,measured=measured,base=base,trial=trial));continue
        match=restore_re.search(line)
        if match:
            restore.append(dict(frame=int(match[1]),iteration=int(match[2]),alpha=float(match[3]),before=float(match[4]),after=float(match[5]),error=float(match[6])));continue
        match=component_re.search(line)
        if match:
            f,it,body=map(int,match.groups()[:3]);components.setdefault((f,it,body),{})[match[4]]=np.fromstring(match[5],sep=',')
rows=[]
for (frame,it),samples in sorted(probes.items()):
    full=sorted([s for s in samples if not s['frozen']],key=lambda s:s['h']);frozen=sorted([s for s in samples if s['frozen']],key=lambda s:s['h'])
    if len(full)!=5 or len(frozen)!=5:continue
    # Use h=1e-5 times the safe sweep, with adjacent h values retained to judge finite-difference stability.
    x,y=full[1],frozen[1];scale=max(abs(x['predicted']),1e-30)
    rows.append(dict(frame=frame,iteration=it,predicted=x['predicted'],actual_slope=x['measured'],frozen_slope=y['measured'],
          frozen_relative_error=abs(y['measured']-x['predicted'])/scale,
          sign_reversal=bool(x['predicted']<0 and all(s['measured']>0 for s in full[:3])),samples=samples))
body_to_name={v:k for k,v in meta['abd_body_mapping'].items()};gradients=[]
for (frame,it,body),stages in sorted(components.items()):
    if set(stages)!={'kinetic_shape','plus_drive','total'} or body==0:continue
    kinetic=stages['kinetic_shape'];drive=stages['plus_drive']-kinetic;contact=stages['total']-stages['plus_drive'];total=stages['total']
    row=dict(frame=frame,iteration=it,body=body,name=body_to_name[body],kinetic_shape=kinetic.tolist(),drive=drive.tolist(),contact=contact.tolist(),total=total.tolist())
    for suffix,sl in [('translation',slice(0,3)),('affine',slice(3,12))]:
        d,c,k,t=drive[sl],contact[sl],kinetic[sl],total[sl]
        denom=np.linalg.norm(d)+np.linalg.norm(c)+np.linalg.norm(k)
        row[suffix]=dict(drive_norm=float(np.linalg.norm(d)),contact_norm=float(np.linalg.norm(c)),kinetic_shape_norm=float(np.linalg.norm(k)),total_norm=float(np.linalg.norm(t)),
            residual_fraction=float(np.linalg.norm(t)/max(denom,1e-30)),drive_contact_cosine=float(d@c/max(np.linalg.norm(d)*np.linalg.norm(c),1e-30)))
    gradients.append(row)
last={}
for row in gradients:last[(row['frame'],row['body'])]=row
left=[row for (_,body),row in sorted(last.items()) if body==meta['abd_body_mapping']['left_link18']]
result=dict(status=meta['status'],completed_steps=meta['completed_steps'],probe_groups=len(rows),
            stable_sign_reversals_with_frozen_agreement=sum(r['sign_reversal'] and r['frozen_relative_error']<.01 for r in rows),
            maximum_restore_energy_error=max([abs(r['error']) for r in restore],default=0),restore_checks=len(restore),
            examples=[r for r in rows if (r['frame']==26 and r['iteration']==100) or (r['frame']==426 and r['iteration'] in [0,5,100,1023])],
            note='Actual directions of this instrumented rerun, not recovered checkpoints of the previous run. ABD vectors are objective gradients, not unscaled forces or individual contact-pair measurements.')
(out/'consistency_summary.json').write_text(json.dumps(result,indent=2));(out/'directional_probes.json').write_text(json.dumps(rows,indent=2))
(out/'restore_checks.json').write_text(json.dumps(restore,indent=2));(out/'abd_components.json').write_text(json.dumps(gradients,indent=2));(out/'left_finger_last_assembly.json').write_text(json.dumps(left,indent=2))
examples=[r for r in rows if r['frame']==26 and r['iteration']==100]
if examples:
    row=examples[0];fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for freeze,label in [(False,'Updated offset energy'),(True,'Frozen offset energy')]:
        s=sorted([v for v in row['samples'] if v['frozen']==freeze],key=lambda v:v['h']);ax.semilogx([v['h'] for v in s],[v['measured'] for v in s],'.-',label=label)
    ax.axhline(row['predicted'],color='black',ls='--',label='Assembled gradient prediction');ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel='Trial step fraction',ylabel='Directional energy slope',title='Actual Newton direction | frame 26, iteration 100');ax.legend();fig.savefig(out/'directional_slope.png',dpi=150);plt.close(fig)
if left:
    fig,ax=plt.subplots(2,1,figsize=(9,6),layout='constrained')
    for key in ['drive_norm','contact_norm','kinetic_shape_norm','total_norm']:
        ax[0].semilogy([r['frame'] for r in left],[max(r['translation'][key],1e-20) for r in left],label=key)
    ax[1].plot([r['frame'] for r in left],[r['translation']['drive_contact_cosine'] for r in left]);ax[0].legend(fontsize=8)
    ax[0].set(ylabel='Translation objective gradient norm');ax[1].set(xlabel='Physical step',ylabel='Drive/contact cosine',ylim=(-1.05,1.05));fig.savefig(out/'left_finger_balance.png',dpi=150);plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
