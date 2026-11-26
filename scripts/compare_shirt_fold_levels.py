"""Compare actual L0/L1/L2 states on a common physical time grid."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import ConvexHull
from shirt_demo.report import load_run, draw, plt, Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for level in range(3):
        p.add_argument(f'--l{level}', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--skip-animation', action='store_true')
    a = p.parse_args()
    paths = [a.l0, a.l1, a.l2]
    data = [load_run(path) for path in paths]
    meta = [d['meta'] for d in data]
    assert all(m['status'] != 'running' for m in meta)
    assert [m['level'] for m in meta] == [0, 1, 2]
    assert meta[2]['finest_requires_convergence']
    plan = json.loads((a.l0.parent/'plan.json').read_text())
    stages = plan['stages_physical_steps']
    end = min(m['completed_steps'] for m in meta)
    assert end > 0
    a.output.mkdir(parents=True, exist_ok=True)
    for level, m in enumerate(meta):
        for key in ['trajectory_sha256', 'robot_q_sha256', 'dt', 'material_parameters', 'runtime_hashes', 'robot_drive_strength']:
            assert m[key] == meta[0][key], key
        expected = copy.deepcopy(meta[0]['config'])
        if m['parameters'].get('line_search_max_iterations') is not None:
            assert level == 2
            expected['line_search']['max_iter'] = m['parameters']['line_search_max_iterations']
        assert expected == m['config']
        if level:
            assert Path(m['lower_run']).resolve() == paths[level-1].resolve()
            audit = json.loads((paths[level]/'report/audit.json').read_text())
            assert audit['velocity_transfer_audit']['all_injected_velocities_verified']
    records = [json.loads((path/'steps.json').read_text()) for path in paths]
    assert all(r['native']['converged'] and not r['native']['preview_non_descent'] for r in records[2])
    full = all(m['completed'] and m['completed_steps'] == stages['observe_end'] for m in meta)
    chosen = [('Initial', 0), ('Settled', 60), ('Lifted', stages['lift_hold_end']),
              ('Before fold', stages['transfer_hold_end']), ('Fold arc end', stages['fold_arc_end']),
              ('Placed', stages['placement_end']), ('Opened', stages['opened']),
              ('Final' if full else 'Last common state', end)]
    chosen = [(name, k) for name, k in chosen if k <= end]
    xmin = np.min([d['x'][:end+1].min(axis=(0,1)) for d in data], axis=0)
    xmax = np.max([d['x'][:end+1].max(axis=(0,1)) for d in data], axis=0)
    bounds = [(min(.15, float(xmin[0])-.03), max(1.18, float(xmax[0])+.03)),
              (min(-.55, float(xmin[1])-.03), max(.55, float(xmax[1])+.03)),
              (min(.78, float(xmin[2])-.03), max(1.26, float(xmax[2])+.03))]
    summary = []
    for level, (d, m, rr) in enumerate(zip(data, meta, records)):
        d['plot_bounds'] = bounds
        f = d['f']; x = d['x']; tri = x[0][f]
        areas = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)*.5
        samples = []
        for name, k in chosen:
            q = x[k]; height = q[:,2]-.8; below = height < -.001
            samples.append(dict(stage=name, step=k, time_s=k*m['dt'],
                height_percentiles_mm=(np.percentile(height,[0,50,90,100])*1000).tolist(),
                rest_area_weighted_mean_height_mm=float(np.average(height[f].mean(1),weights=areas)*1000),
                below_table_vertex_count=int(below.sum()), below_table_vertex_fraction=float(below.mean()),
                below_table_vertices_beyond_right_edge=int(np.count_nonzero(below & (q[:,0]>1.15))),
                footprint_convex_hull_ratio=float(ConvexHull(q[:,:2]).volume/ConvexHull(x[min(60,end),:,:2]).volume)))
        fold_path = paths[level]/'report/custom_fold_audit.json'
        fold = json.loads(fold_path.read_text()) if fold_path.exists() else None
        summary.append(dict(level=level, vertices=m['vertices'], faces=m['faces'], status=m['status'],
            saved_steps=m['completed_steps'], physics_seconds=m['physics_seconds'],
            native_steps_in_common_prefix=sum(r['native']['converged'] for r in rr[:end]),
            preview_steps_in_common_prefix=sum(r['native']['preview_non_descent'] for r in rr[:end]),
            newton_limits_in_common_prefix=sum(r['native']['hit_newton_limit'] for r in rr[:end]),
            total_native_steps=m['native_converged_steps'],total_preview_steps=m['preview_stopped_steps'],
            total_newton_limits=m['newton_limit_steps'],solver_overrides=m['solver_overrides'],
            hands=fold['hands'] if fold else None, samples=samples))
    result = dict(full_first_fold_completed=full, common_steps=end, requested_steps=stages['observe_end'],
        common_physical_seconds=end*meta[0]['dt'], levels=summary,
        paths=[str(p.resolve()) for p in paths],
        failed_step=json.loads((a.l2/'failed_step.json').read_text()) if (a.l2/'failed_step.json').exists() else None,
        limitations=['Only actual saved positions are displayed; no prolongated cloth is presented as L2 simulation.',
            'L0/L1 preview exits are not Newton convergence. L2 accepts only converged steps.',
            'Vertex fractions depend on mesh sampling and are not cloth area fractions.',
            'Negative heights outside the finite table indicate overhang, not improved fold flatness.',
            'Completion does not certify a correctly aligned fold or continuous collision-free motion.'])
    (a.output/'comparison.json').write_text(json.dumps(result,indent=2))
    fig = plt.figure(figsize=(15,12))
    for row,(view,elev,azim) in enumerate([('Overview',48,-125),('Overhead',90,-90),('Table edge',22,55)]):
        for level,d in enumerate(data):
            ax=fig.add_subplot(3,3,row*3+level+1,projection='3d')
            draw(ax,d,end,f'L{level} | {view} | {end*meta[0]["dt"]:.2f}s',level<2)
            ax.view_init(elev=elev,azim=azim)
            if view=='Overhead':ax.set_proj_type('ortho');ax.set_zticks([]);ax.set_zlabel('')
            if view=='Table edge':ax.plot([1.15,1.15],[-.55,.55],[.8,.8],color='#d65a32',linewidth=2)
    fig.suptitle('Full first-fold trajectory: simulated motion, not a fold-success verdict' if full else f'Accepted common prefix only: {end}/{stages["observe_end"]} steps')
    fig.tight_layout(h_pad=4,w_pad=2);fig.savefig(a.output/'three_levels_final.png',dpi=140,bbox_inches='tight');plt.close(fig)
    keyframes=[('Settled',60),('Lifted',stages['lift_hold_end']),('Fold arc end',stages['fold_arc_end'])]
    keyframes=[(name,k) for name,k in keyframes if k<=end]
    if not full:keyframes.insert(0,('Initial',0))
    keyframes.append(('Final' if full else 'Last saved',end))
    columns=len(keyframes)
    fig=plt.figure(figsize=(4.25*columns,11))
    for level,d in enumerate(data):
        for col,(name,k) in enumerate(keyframes):
            draw(fig.add_subplot(3,columns,level*columns+col+1,projection='3d'),d,k,f'L{level} | {name} | {k*meta[0]["dt"]:.2f}s',level<2)
    fig.tight_layout();fig.savefig(a.output/'three_levels_stages.png',dpi=130,bbox_inches='tight');plt.close(fig)
    if not a.skip_animation:
        images=[]
        frames=np.unique(np.linspace(0,end,min(120,max(20,int(np.ceil(end*meta[0]['dt']*6)))))).astype(int)
        for k in frames:
            fig=plt.figure(figsize=(15,5),dpi=90)
            for level,d in enumerate(data):
                ax=fig.add_subplot(1,3,level+1,projection='3d')
                draw(ax,d,k,f'L{level} | {meta[level]["faces"]:,} faces',level<2)
                ax.view_init(elev=25,azim=55)
                ax.plot([1.15,1.15],[-.55,.55],[.8,.8],color='#d65a32',linewidth=1)
            fig.suptitle(f'{k*meta[0]["dt"]:.2f}s | '+('Full first-fold trajectory' if full else f'Accepted prefix: {end}/{stages["observe_end"]} steps'))
            fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        images[0].save(a.output/'three_levels.gif',save_all=True,append_images=images[1:],
            duration=max(20,round(end*meta[0]['dt']*100/len(images))*10),loop=0)
        if not full:
            # Show the full requested timeline, leaving unsolved L2 frames blank.
            # Never freeze or prolongate the last L2 state to imply continuation.
            requested=stages['observe_end'];images=[]
            lo=np.min([d['x'][:m['completed_steps']+1].min(axis=(0,1)) for d,m in zip(data,meta)],axis=0)
            hi=np.max([d['x'][:m['completed_steps']+1].max(axis=(0,1)) for d,m in zip(data,meta)],axis=0)
            common_bounds=[(min(.15,float(lo[0])-.03),max(1.18,float(hi[0])+.03)),
                           (min(-.55,float(lo[1])-.03),max(.55,float(hi[1])+.03)),
                           (min(.78,float(lo[2])-.03),max(1.26,float(hi[2])+.03))]
            for d in data:d['plot_bounds']=common_bounds
            for k in np.unique(np.linspace(0,requested,120).astype(int)):
                fig=plt.figure(figsize=(15,5),dpi=90)
                for level,d in enumerate(data):
                    ax=fig.add_subplot(1,3,level+1,projection='3d')
                    if k<=meta[level]['completed_steps']:
                        draw(ax,d,k,f'L{level} | {meta[level]["faces"]:,} faces',level<2)
                        ax.view_init(elev=25,azim=55)
                        ax.plot([1.15,1.15],[-.55,.55],[.8,.8],color='#d65a32',linewidth=1)
                    else:
                        ax.set_axis_off()
                        ax.text2D(.5,.62,f'L{level} STOPPED',transform=ax.transAxes,ha='center',fontsize=17,color='#ac392f')
                        ax.text2D(.5,.43,f'Last converged state: {meta[level]["completed_steps"]*meta[level]["dt"]:.2f} s\nNo simulated state at this time\nFirst fold NOT completed',transform=ax.transAxes,ha='center',fontsize=12)
                fig.suptitle(f'Full requested timeline | {k*meta[0]["dt"]:.2f} s | Missing results are left blank')
                fig.tight_layout();fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
            images[0].save(a.output/'three_levels_full_timeline.gif',save_all=True,append_images=images[1:],
                duration=max(20,round(requested*meta[0]['dt']*100/len(images))*10),loop=0)
    print(json.dumps(dict(full_first_fold_completed=full,common_steps=end,output=str(a.output.resolve())),indent=2))


if __name__ == '__main__':
    main()
