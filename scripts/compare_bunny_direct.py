"""Compare saved native L0 runs with PSQ on their common accepted prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--strain', type=Path, required=True)
    p.add_argument('--psq', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    roots = [a.native, a.strain, a.psq]
    metas = [json.loads((r/'run.json').read_text()) for r in roots]
    states = [np.load(r/'accepted_positions.npy') for r in roots]
    rows = [json.loads((r/'steps.json').read_text()) for r in roots]
    for m, x, records in zip(metas, states, rows):
        assert m['config'] == metas[0]['config']
        assert m['material'] == metas[0]['material']
        assert m['assets'] == metas[0]['assets']
        assert len(x) == m['completed_steps']+1 == len(records)+1
        assert np.isfinite(x).all()
        assert all(r['native']['converged'] or r['native'].get('preview_non_descent') for r in records)
        assert not m.get('force_updates')
    assert all(r['native']['converged'] for records in rows[:2] for r in records)
    n = min(m['completed_steps'] for m in metas)
    cloth = np.load('output/bunny_paper_assets/mesh_0.npz')
    bunny = np.load('output/bunny_paper_assets/bunny.npz')
    names = ['Native coarse | no strain barrier', 'Native coarse | strain barrier', 'PSQ coarse | strain barrier']
    summary = dict(common_steps=n, dt=metas[0]['dt'], matched_config_material_assets=True,
        interpretation='Native elasticity changes the discrete elastic objective. Native-only also removes the custom strain barrier. No speedup or equivalent-physics claim.',
        independent_collision_audit=False, runs=[])
    for name, root, m, x, records in zip(names, roots, metas, states, rows):
        audit=json.loads((root/'audit.json').read_text())
        summary['runs'].append(dict(name=name, path=str(root.resolve()), completed=m['completed'],
            accepted_steps=m['completed_steps'], requested_steps=m['requested_steps'],
            strict_steps=m['strictly_converged_steps'], preview_steps=m['preview_stopped_steps'],
            failed_step=m.get('failed_step'), error=m.get('error'),
            maximum_principal_stretch=audit['max_principal_stretch'],
            maximum_fixed_bunny_error=max(r['fixed_error'] for r in records),
            max_newton_iterations=max(r['native']['newton_iterations'] for r in records),
            common_prefix_seconds=sum(r['seconds'] for r in records[:n]),
            trajectory_sha256=hashlib.sha256((root/'accepted_positions.npy').read_bytes()).hexdigest()))
    summary['common_final_rms_native_vs_strain'] = float(np.sqrt(np.mean(np.sum((states[0][n]-states[1][n])**2,axis=1))))
    summary['common_final_rms_strain_vs_psq'] = float(np.sqrt(np.mean(np.sum((states[1][n]-states[2][n])**2,axis=1))))
    (a.output/'comparison.json').write_text(json.dumps(summary,indent=2))
    pts=np.concatenate([x[:n+1].reshape(-1,3) for x in states]+[bunny['vertices']])
    center=(pts.min(0)+pts.max(0))/2;radius=np.ptp(pts,axis=0).max()*.55
    images=[]
    for i in np.unique(np.linspace(0,n,min(n+1,26)).astype(int)):
        fig=plt.figure(figsize=(13,4.6))
        for k,(name,x,records) in enumerate(zip(names,states,rows)):
            ax=fig.add_subplot(1,3,k+1,projection='3d')
            triangles=[];colors=[]
            for v,f,c in [(x[i],cloth['faces'],[.29,.62,.85]),(bunny['vertices'],bunny['faces'],[.6,.62,.65])]:
                t=v[f];norm=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-20)
                triangles.append(t);colors.append((.4+.6*np.abs(norm@np.array([-.3,-.4,.866])))[:,None]*c)
            ax.add_collection3d(Poly3DCollection(np.concatenate(triangles),facecolors=np.concatenate(colors),edgecolors='none'))
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
            ax.view_init(18,-65);ax.set_box_aspect((1,1,1));ax.set_axis_off()
            status='Initial' if i==0 else ('Newton converged' if records[i-1]['native']['converged'] else 'Preview stop; NOT converged')
            ax.set_title(name+'\n'+status,fontsize=10)
        fig.suptitle(f'Bunny L0: actual accepted trajectories | t={i*.01:.2f}s\nCommon prefix only; different elastic models | slowed playback',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.84));fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
    images[-1].save(a.output/'comparison.png')
    images[0].save(a.output/'comparison.gif',save_all=True,append_images=images[1:],duration=100,loop=0)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
