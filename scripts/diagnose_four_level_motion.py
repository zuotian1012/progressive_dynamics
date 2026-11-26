"""Measure motion and velocity-predictor contributions without rerunning dynamics."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from progressive.diagnostics import nearest_material_point
from scripts.render_four_level_fold import SOURCES
import matplotlib.pyplot as plt

OUT = Path('output/shirt_four_level_full_fold_comparison_20261119/motion_diagnosis')
STAGES = [0, 60, 356, 596, 1076, 1436, 1616, 2216, 2576, 2696, 2936, 3116, 3476]


def average_track(array, weights):
    result = np.empty((len(array), 3))
    for k in range(0, len(array), 64):
        result[k:k+64] = np.einsum('tnc,n->tc', array[k:k+64], weights, optimize=False)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    output = []
    fine_x = np.load(SOURCES[-1] / 'positions.npy', mmap_mode='r')[0]
    fine_f = np.load(SOURCES[-1] / 'faces.npy')
    targets = []
    for query in [[.90,.10,.925],[.98,.10,.925],[.85,-.15,.925],[.85,.20,.925]]:
        face,bary,_ = nearest_material_point(fine_x, fine_f, np.array(query))
        targets.append(bary @ fine_x[fine_f[face]])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    for level, source in enumerate(SOURCES):
        meta = json.loads((source / 'run.json').read_text())
        x = np.load(source / 'positions.npy', mmap_mode='r')
        faces = np.load(source / 'faces.npy')
        tri = x[0][faces]
        areas = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1) / 2
        weights = np.zeros(x.shape[1])
        np.add.at(weights, faces.ravel(), np.repeat(areas / 3, 3))
        weights /= weights.sum()
        center = average_track(x, weights)
        # Match identical L3 rest-surface locations on each initial mesh.
        # Cross-level material correspondence is approximate; record distances.
        probes, probe_specs = [], []
        for target in targets:
            face, bary, distance = nearest_material_point(x[0], faces, target)
            point = np.einsum('j,tjc->tc', bary, x[:,faces[face]])
            probes.append(point)
            probe_specs.append(dict(target=target.tolist(), initial_distance=distance, face=face, barycentric=bary.tolist()))
        probe = np.mean(probes, axis=0)
        records = json.loads((source / 'steps.json').read_text())
        predictor = None
        if level:
            velocity = np.load(source / 'injected_velocities.npy', mmap_mode='r')
            mean_velocity = average_track(velocity, weights)
            predictor = np.vstack((np.zeros(3), np.cumsum(mean_velocity * meta['dt'], axis=0)))
        samples = []
        for k in STAGES:
            row = dict(step=k, center_displacement_mm=((center[k]-center[0])*1000).tolist(),
                       far_patch_displacement_mm=((probe[k]-probe[0])*1000).tolist())
            if predictor is not None:
                row['cumulative_injected_displacement_mm'] = (predictor[k]*1000).tolist()
                row['cumulative_local_correction_mm'] = ((center[k]-center[0]-predictor[k])*1000).tolist()
            samples.append(row)
        preview = [i+1 for i,r in enumerate(records) if r['native'].get('preview_non_descent')]
        result = dict(level=level, source=str(source), rest_area_m2=float(areas.sum()),
                      rest_center=center[0].tolist(), samples=samples, probes=probe_specs,
                      preview_steps=len(preview), preview_by_interval=[sum(a<k<=b for k in preview) for a,b in zip(STAGES[:-1],STAGES[1:])],
                      material=meta.get('material_parameters'), parameters=meta.get('parameters'),
                      max_gripper_origin_error_mm=max(max(r['body_origin_tracking_errors_m']) for r in records)*1000)
        output.append(result)
        t = np.arange(len(x)) * meta['dt']
        axes[0].plot(t,(center[:,0]-center[0,0])*1000,label=f'L{level}',color=f'C{level}')
        axes[1].plot(t,(probe[:,0]-probe[0,0])*1000,label=f'L{level}',color=f'C{level}')
        if predictor is not None:
            axes[2].plot(t,(center[:,0]-center[0,0]-predictor[:,0])*1000,label=f'L{level}',color=f'C{level}')
        np.savez_compressed(OUT/f'L{level}_tracks.npz',center=center,far_patch=probe,predictor=predictor if predictor is not None else np.empty((0,3)))
    for ax, title in zip(axes, ['Rest-area weighted center: x motion', 'Matched far surface probes: x motion', 'Accumulated local correction to injected motion']):
        ax.set(title=title,xlabel='Physical time (s)',ylabel='Displacement (mm)')
        for step in [1076,1616,2216,2936]:ax.axvline(step/120,color='gray',lw=.6,alpha=.5)
        ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'motion_comparison.png',dpi=150)
    (OUT/'measurements.json').write_text(json.dumps(dict(levels=output,
        limitation='Center motion includes folding; probes also include deformation and approximate nearest-rest-surface matching across levels. Local correction is a kinematic decomposition, not a measured friction force or a causal ablation.'),indent=2))
    for row in output:
        print('LEVEL',row['level'],'area',row['rest_area_m2'],'preview',row['preview_steps'],'probe max initial distance',max(p['initial_distance'] for p in row['probes']),flush=True)
        for s in row['samples']:
            print(s['step'],'COMx',round(s['center_displacement_mm'][0],2),'farx',round(s['far_patch_displacement_mm'][0],2),
                  'localx',round(s.get('cumulative_local_correction_mm',[0])[0],2),flush=True)


if __name__ == '__main__':main()
