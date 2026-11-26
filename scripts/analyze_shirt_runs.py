"""Compare compute costs, mesh correspondence and measured grip-region motion."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz


def analyze(path):
    meta=json.loads((path/'run.json').read_text())
    if not meta['completed']:
        raise ValueError(f'Incomplete run: {path}')
    x=np.load(path/'positions.npy',mmap_mode='r')
    transforms=np.load(path/'robot.npy',mmap_mode='r')
    times=np.load(path/'timing.npz')['step_seconds']
    params=meta['parameters']
    def index(frame):
        return params['settle_steps']+(frame+1)*params['substeps']
    grasp={}
    for name,link in [('left',1),('right',5)]:
        k=index(218)
        t=transforms[k,link]
        tcp=t[:3,:3]@np.array([.155,.001786,.014])+t[:3,3]
        distance=np.linalg.norm(x[k]-tcp,axis=1)
        ids=np.argsort(distance)[:8]
        frames=np.arange(k,index(362)+1)
        points=x[frames][:,ids].mean(1)
        wrist=transforms[frames,link]
        local=np.einsum('nij,nj->ni',np.linalg.inv(wrist[:,:3,:3]),points-wrist[:,:3,3])
        drift=np.linalg.norm(local-local[0],axis=1)
        grasp[name]={'tracked_vertex_ids':ids.tolist(),
                     'nearest_tcp_distance_at_close_m':float(distance.min()),
                     'region_max_tcp_distance_at_close_m':float(distance[ids].max()),
                     'peak_mean_region_lift_m':float((points[:,2]-points[0,2]).max()),
                     'max_wrist_relative_drift_m':float(drift.max()),
                     'note':'Nearest 8 material vertices at closure; geometric tracking diagnostic, not contact-force proof.'}
    baseline=x[params['settle_steps']]
    final=x[-1]
    span=np.ptp(final[:,:2],axis=0)
    return {'path':str(path),'method':meta['method'],'level':meta['level'],
            'vertices':meta['vertices'],'faces':meta['faces'],'completed_steps':meta['completed_steps'],
            'simulated_seconds':meta['dt']*meta['completed_steps'],
            'physics_seconds':float(times.sum()),'physics_ms_median':float(np.median(times)*1000),
            'physics_ms_p95':float(np.percentile(times,95)*1000),'physics_steps_per_second':float(1/times.mean()),
            'simulated_seconds_per_wall_second':float(meta['dt']/times.mean()),
            'bbox_footprint_ratio_vs_settled':float(np.prod(span)/np.prod(np.ptp(baseline[:,:2],axis=0))),
            'final_height_above_table_m':np.percentile(final[:,2]-.8,[0,50,90,100]).tolist(),
            'grip_region_tracking':grasp,'finite':bool(np.isfinite(x).all())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('output/shirt_soft_fold'))
    parser.add_argument('--assets',type=Path,default=Path('output/shirt_assets'))
    args=parser.parse_args()
    paths=[args.root/n for n in ('coarse','progressive_fine','direct_fine')]
    rows=[analyze(p) for p in paths]
    coarse,progressive,direct=rows
    p=load_npz(args.assets/'map_0_1.npz')
    x=np.load(paths[0]/'positions.npy',mmap_mode='r')
    xp=np.load(paths[1]/'positions.npy',mmap_mode='r')
    xd=np.load(paths[2]/'positions.npy',mmap_mode='r')
    # Common material samples on the identical fine mesh, evaluated at fixed steps.
    ids=np.unique(np.linspace(0,len(x)-1,101,dtype=int))
    errors_p=np.array([np.sqrt(np.mean(np.sum((xp[k]-p@x[k])**2,axis=1))) for k in ids])
    errors_d=np.array([np.sqrt(np.mean(np.sum((xd[k]-p@x[k])**2,axis=1))) for k in ids])
    record={'scope':'First source segment only; native Windows contact/friction; one timing sample per method.',
            'timing_scope':'Predictor, native step, retrieve and cloth state transfer. Excludes initialization, FK, file writes and rendering.',
            'paper_scope':'VelPro-inspired velocity predictor with per-level elastic energies; no PSQ or shared fine proxy energy.',
            'runs':rows,'preview_speedup_vs_direct_fine':direct['physics_seconds']/coarse['physics_seconds'],
            'progressive_incremental_speedup_vs_direct_fine':direct['physics_seconds']/progressive['physics_seconds'],
            'progressive_total_seconds':coarse['physics_seconds']+progressive['physics_seconds'],
            'progressive_total_speedup_vs_direct_fine':direct['physics_seconds']/(coarse['physics_seconds']+progressive['physics_seconds']),
            'progressive_vs_lifted_coarse_rms_m':{'mean':float(errors_p.mean()),'max':float(errors_p.max())},
            'direct_fine_vs_lifted_coarse_rms_m':{'mean':float(errors_d.mean()),'max':float(errors_d.max())},
            'metric_note':'Uniform-vertex RMS on 101 common time samples; not paper mass-norm continuity or proof of superior physical fidelity.'}
    (args.root/'comparison_metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
