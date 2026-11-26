"""Show the original rejected search and independently audited repair events."""
from pathlib import Path
import argparse,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
    trials=[];repairs=[]
    with (ROOT/'output/flag_fine_20261118/L3.log').open(encoding='utf-8-sig',errors='replace') as f:
        for line in f:
            if 'FLAG_SEARCH_TRIAL frame=442 newton=22 ' in line:
                d=dict(re.findall(r'(\w+)=([^\s]+)',line));trials.append({k:float(d[k]) for k in ['search','alpha','E0','E']})
    with (ROOT/'output/flag_strict_20261118/L3.log').open(encoding='utf-8-sig',errors='replace') as f:
        for line in f:
            if 'FLAG_PSQ_RESCUE_ACCEPT ' in line:
                d=dict(re.findall(r'(\w+)=([^\s]+)',line));repairs.append({k:float(d[k]) for k in ['frame','newton','E0','E1','alpha']})
    assert len(trials)==8 and all(t['E']>t['E0'] for t in trials)
    assert repairs and all(t['E1']<t['E0'] for t in repairs)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    axs[0].plot([t['search']+1 for t in trials],[(t['E']-t['E0'])/t['E0'] for t in trials],'o-',color='#b05546')
    axs[0].set_yscale('log');axs[0].set_xlabel('Line-search trial (limit = 8)');axs[0].set_ylabel('Relative energy increase');axs[0].set_title('Original L3: step 442, Newton 22\nAll 8 trials increase energy')
    axs[0].set_xticks(range(1,9));axs[0].grid(alpha=.2)
    axs[1].bar(range(len(repairs)),[100*(r['E0']-r['E1'])/r['E0'] for r in repairs],color='#197d9c')
    axs[1].set_xticks(range(len(repairs)),[str(int(r['frame'])) for r in repairs]);axs[1].set_xlabel('Time-step index in the rebuilt L3 run');axs[1].set_ylabel('Full-energy decrease on retry (%)');axs[1].set_title('Rebuilt L3: every exact-gradient retry\nRestored state + accepted energy decrease');axs[1].grid(axis='y',alpha=.2)
    fig.suptitle('Failure and repair evidence | different trajectories, not a same-state ablation',fontsize=13)
    fig.savefig(a.output/'failure_and_repair.png',dpi=150);plt.close(fig)
    (a.output/'failure_and_repair.json').write_text(json.dumps(dict(original_trials=trials,rebuilt_rescues=repairs),indent=2))

if __name__=='__main__':main()
