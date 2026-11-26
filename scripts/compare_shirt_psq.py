"""Compare completed independent PSQ resolutions at identical physical times."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from shirt_demo.report import load_run,draw,plt,Image


def compare(paths,out):
    data=[load_run(p) for p in paths]
    audits=[json.loads((p/'report/audit.json').read_text()) for p in paths]
    first=data[0]['meta']
    for d in data:
        assert d['meta']['completed']
        for key in ('config','material_parameters','robot_q_sha256','steps','dt'):
            assert d['meta'][key]==first[key],f'Mismatched {key}'
    out.mkdir(parents=True,exist_ok=False)
    n=first['completed_steps'];images=[]
    for k in np.unique(np.linspace(0,n,150,dtype=int)):
        fig=plt.figure(figsize=(12,6),dpi=100)
        for j,(d,a) in enumerate(zip(data,audits)):
            m=d['meta'];flag='INITIAL' if k==0 else ('FORCED' if a['records'][str(k)]['forced'] else 'CONVERGED')
            draw(fig.add_subplot(1,len(data),j+1,projection='3d'),d,k,
                 f'L{m["level"]}: {m["vertices"]:,} vertices / {m["faces"]:,} faces | {flag}',True)
        fig.suptitle(f'Independent resolutions | same 55k elastic proxy | t={k*first["dt"]:.2f}s\nExploratory forced-update runs; not validated folding',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.92]);fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
        if k==n:fig.savefig(out/'final.png',dpi=150)
        plt.close(fig)
    duration=max(40,round(n*first['dt']*1000/len(images)/10)*10)
    images[0].save(out/'comparison.gif',save_all=True,append_images=images[1:],duration=duration,loop=0)
    rows=[]
    for d,a,p in zip(data,audits,paths):
        rows.append(dict(level=d['meta']['level'],vertices=d['meta']['vertices'],faces=d['meta']['faces'],source=str(p),
                         **{key:a[key] for key in ('physical_seconds','physics_seconds','forced_count','strictly_converged_count',
                           'maximum_principal_stretch','maximum_body_origin_tracking_error_m','final_body_origin_tracking_errors_m')}))
    (out/'comparison.json').write_text(json.dumps(dict(matched_configuration=True,
        scope='Independent BDF1 resolution comparison; L1 does not use L0 velocity or positions; one run each, not a speedup benchmark',runs=rows),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path,nargs=2);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.runs,a.output)
