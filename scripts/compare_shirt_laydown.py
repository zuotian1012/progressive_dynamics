"""Compare two completed physical first-fold runs at corresponding action phases."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import ConvexHull
from shirt_demo.report import load_run,draw,plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    paths=[a.previous,a.candidate];data=[load_run(path/'run') for path in paths];plans=[json.loads((path/'plan.json').read_text()) for path in paths]
    for d in data:assert d['meta']['completed']
    cross_level=data[0]['meta']['level']!=data[1]['meta']['level']
    xmin=np.min([d['x'].min(axis=(0,1)) for d in data],axis=0);xmax=np.max([d['x'].max(axis=(0,1)) for d in data],axis=0)
    bounds=[(min(.15,float(xmin[0])-.03),max(1.18,float(xmax[0])+.03)),
        (min(-.55,float(xmin[1])-.03),max(.55,float(xmax[1])+.03)),
        (min(.78,float(xmin[2])-.03),max(1.26,float(xmax[2])+.03))]
    for d in data:d['plot_bounds']=bounds
    for key in ['material_parameters','config','runtime_hashes','robot_drive_strength']:assert data[0]['meta'][key]==data[1]['meta'][key]
    if cross_level:
        assert [d['meta']['level'] for d in data]==[0,1]
        assert Path(data[1]['meta']['lower_run']).resolve()==(a.previous/'run').resolve()
        for key in ['trajectory_sha256','robot_q_sha256','completed_steps','dt']:assert data[0]['meta'][key]==data[1]['meta'][key]
        assert plans[0]['stages_physical_steps']==plans[1]['stages_physical_steps']
        audit=json.loads((a.candidate/'run/report/audit.json').read_text())
        assert audit['rebuilt_hierarchy_verified'] and audit['velocity_transfer_audit']['all_injected_velocities_verified']
        labels=['L0 driver','L1 progressive dynamics']
    else:
        for key in ['mesh_sha256','psq_bundle_sha256']:assert data[0]['meta'][key]==data[1]['meta'][key]
        np.testing.assert_array_equal(data[0]['f'],data[1]['f']);np.testing.assert_array_equal(data[0]['x'][0],data[1]['x'][0])
        labels=['Previous vertical placement','Forward laydown + wrist rotation']
    rows=[]
    for label,path,d,plan in zip(labels,paths,data,plans):
        f=d['f'];tri=d['x'][0][f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        samples=[]
        for name in ['placement_end','opened','observe_end']:
            k=plan['stages_physical_steps'][name];x=d['x'][k];height=x[:,2]-.8
            samples.append(dict(phase=name,step=k,vertex_height_median_mm=float(np.median(height)*1000),
                vertex_height_p90_mm=float(np.percentile(height,90)*1000),maximum_height_mm=float(height.max()*1000),
                rest_area_weighted_mean_height_mm=float(np.average(height[f].mean(1),weights=area)*1000),
                footprint_convex_hull_ratio=float(ConvexHull(x[:,:2]).volume/ConvexHull(d['x'][60,:,:2]).volume)))
        rows.append(dict(label=label,experiment=str(path.resolve()),samples=samples))
    fig=plt.figure(figsize=(14,9))
    for row,(label,d,plan) in enumerate(zip(labels,data,plans)):
        for col,(name,title) in enumerate([('placement_end','Placed'),('opened','Opened'),('observe_end','Final')]):
            k=plan['stages_physical_steps'][name];draw(fig.add_subplot(2,3,row*3+col+1,projection='3d'),d,k,f'{label}\n{title}',False)
    fig.tight_layout();fig.savefig(a.output/'placement_comparison.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(11,5));norm=plt.Normalize(float(data[0]['x'][0,:,0].min()),float(data[0]['x'][0,:,0].max()))
    for i,(label,d,plan) in enumerate(zip(labels,data,plans)):
        k=plan['stages_physical_steps']['observe_end'];ax=fig.add_subplot(1,2,i+1,projection='3d');draw(ax,d,k,label,False)
        ax.collections[0].set_facecolor(plt.cm.viridis(norm(d['x'][0,d['f'],0].mean(1))));ax.view_init(elev=90,azim=-90);ax.set_proj_type('ortho');ax.set_zticks([]);ax.set_zlabel('')
    fig.suptitle('Final states, fixed original material colors');fig.tight_layout();fig.savefig(a.output/'final_material_comparison.png',dpi=140);plt.close(fig)
    if cross_level:
        fig=plt.figure(figsize=(14,9))
        for row,(label,d,plan) in enumerate(zip(labels,data,plans)):
            for col,(title,k) in enumerate([('Settled',60),('Lifted',plan['stages_physical_steps']['lift_hold_end']),('Before fold',plan['stages_physical_steps']['transfer_hold_end'])]):
                draw(fig.add_subplot(2,3,row*3+col+1,projection='3d'),d,k,f'{label}\n{title}',True)
        fig.tight_layout();fig.savefig(a.output/'pickup_comparison.png',dpi=140);plt.close(fig)
    result=dict(runs=rows,unchanged_material_contact_solver_runtime_verified=True,cross_level=cross_level,
        limitations=[('Different mesh resolutions with verified L0-to-L1 velocity transfer; vertex percentiles depend on sampling.' if cross_level else 'Several trajectory parameters change together; not a single-variable causal ablation.'),
        'Repeat-run prefix states need not be identical despite identical targets.',
        'Lower height or smaller footprint alone does not certify a correctly layered fold.',
        'Cloth hanging off a finite table can reduce height statistics; inspect table extent before interpreting flatness.'])
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
