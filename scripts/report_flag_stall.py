"""Expose loss of dynamic response instead of presenting completed frames as success."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();m=json.loads((a.run/'run.json').read_text())
    x=np.load(a.run/'positions.npy',mmap_mode='r')[:m['completed_steps']+1];t=np.arange(1,len(x))*m['dt']
    speed=np.array([np.linalg.norm(x[i]-x[i-1],axis=1).max()/m['dt'] for i in range(1,len(x))])
    zero=speed<1e-10;trailing=0
    for z in zero[::-1]:
        if not z:break
        trailing+=1
    onset=(len(x)-1-trailing)*m['dt'] if trailing else None
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    axes[0].plot(t,8+5*np.sin(2*np.pi*1.3*t),color='#168eb0');axes[0].set_ylabel('Prescribed acceleration y (m/s²)')
    axes[1].plot(t,speed,color='#af4a31');axes[1].set_ylabel('Maximum vertex speed (m/s)');axes[1].set_xlabel('Physical time (s)')
    if trailing:
        for ax in axes:ax.axvspan(onset,t[-1],color='#c96445',alpha=.15)
        axes[1].text(onset+.25,max(speed)*.72,f'No motion for final {trailing*m["dt"]:.2f} s\nwhile forcing continues to vary',color='#88391f')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('FLAG L0 | forcing and dynamic response',fontsize=16)
    out=a.run/'report';out.mkdir(exist_ok=True);fig.savefig(out/'freeze_diagnostic.png',dpi=160);plt.close(fig)
    (out/'freeze_diagnostic.json').write_text(json.dumps(dict(trailing_zero_motion_steps=trailing,onset_seconds=onset,
        duration_seconds=trailing*m['dt'],definition='max vertex displacement < 1e-10 * dt',
        interpretation='Zero saved displacement while forcing varies; stopping reasons must be read separately from steps.json, not inferred from zero motion'),indent=2))

if __name__=='__main__':main()
