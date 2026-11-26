"""Audit all accepted strip frames and draw synchronized material-space comparisons."""
import argparse
import csv
import json
from pathlib import Path

import ipctk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.sparse import load_npz

from run_strip_swing_demo import DT, STEPS, PACKAGES, run_path, write_json
from progressive.prolongation import frames

COLORS = ('#343a46', '#cb642b', '#137787')
STYLES = ('-', '--', ':')
LABELS = {'semi':'PD Semi', 'velpro':'PD++ VelPro', 'historical':'Historical', 'corrected':'Corrected'}


def main(root):
    assets = root/'assets'
    fine = np.load(assets/'mesh_2.npz')
    rest, faces = fine['vertices'], fine['faces']
    center = np.flatnonzero(rest[:,1] == 0)
    tip = center[-1]
    mass = np.zeros(len(rest))
    np.add.at(mass, faces.ravel(), np.repeat(frames(rest, faces)[3]/3, 3))
    mass /= mass.sum()
    maps = [load_npz(assets/'map_0_1.npz'), load_npz(assets/'map_1_2.npz')]
    to_fine = [maps[1]@maps[0], maps[1], None]
    t = DT*np.arange(STEPS+1)
    curves, audits, summaries, rows = {}, {}, {}, []
    for build in PACKAGES:
        for case in ('fall','swing'):
            for method in ('semi','velpro'):
                lifted = []
                for level in range(3):
                    out = run_path(root, build, case, method, level)
                    key = out.relative_to(root).as_posix()
                    x = np.load(out/'positions.npy')
                    if key not in audits:
                        meta = json.loads((out/'run.json').read_text())
                        steps = json.loads((out/'steps.json').read_text())
                        if meta['status']!='completed' or len(x)!=STEPS+1 or len(steps)!=STEPS:
                            raise ValueError(f'Incomplete trajectory: {out}')
                        mesh = np.load(assets/f'mesh_{level}.npz')
                        v, f = mesh['vertices'], mesh['faces'].astype(np.int32)
                        edges = np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1), axis=0)
                        cm = ipctk.CollisionMesh(v, edges, f)
                        systems = json.loads((out/'backend/systems.json').read_text())['sim_systems']
                        names = [system['name'] for system in systems if system['valid']]
                        if any('SoftPositionConstraint' in s or 'FrictionalContact' in s for s in names):
                            raise ValueError(f'Unexpected constraint or friction system: {out}')
                        if meta['contact_pair_enabled'] or meta['consistency_weight'] or meta['friction']:
                            raise ValueError(f'Unexpected contact or consistency settings: {out}')
                        audit = dict(self_intersection_frames=[i for i,y in enumerate(x) if ipctk.has_intersections(cm,y)],
                                     max_stretch=max(s['stretch'] for s in steps), max_pin_error_m=max(s['pin_error_m'] for s in steps),
                                     preview_frames=sum(bool(s['native']['preview_non_descent']) for s in steps))
                        if case=='fall':
                            analytic = np.broadcast_to(v, x.shape).copy()
                            n = np.arange(len(x))
                            analytic[:,:,2] -= (.5*9.81*DT**2*n*(n+1))[:,None]
                            audit['max_error_discrete_gravity_m'] = float(np.linalg.norm(x-analytic, axis=2).max())
                        audits[key] = audit
                    u = to_fine[level]
                    lifted.append(np.asarray([u@y for y in x]) if u is not None else x)
                if case=='fall':
                    continue
                curves[build,method] = [x[:,center,:] for x in lifted]
                for level in range(3):
                    x = lifted[level]
                    err = x-lifted[0]
                    rms = np.sqrt(np.einsum('i,tij,tij->t',mass,err,err))*1000
                    tip_error = np.linalg.norm(err[:,tip,:],axis=1)*1000
                    center_error = np.linalg.norm(err[:,center,:],axis=2).max(axis=1)*1000
                    z = x[:,tip,2]
                    mins = np.flatnonzero((z[1:-1]<z[:-2]) & (z[1:-1]<=z[2:]))+1
                    crossings = np.flatnonzero(rms>1.)
                    summaries[f'{build}/{method}/L{level}'] = dict(
                        peak_rms_mm=float(rms.max()), rms_peak_frame=int(rms.argmax()), final_rms_mm=float(rms[-1]),
                        first_rms_above_1mm=int(crossings[0]) if len(crossings) else None,
                        max_tip_distance_mm=float(tip_error.max()), max_centerline_distance_mm=float(center_error.max()),
                        first_tip_min_frame=int(mins[0]) if len(mins) else None,
                        first_tip_min_height_mm=float(z[mins[0]]*1000) if len(mins) else None)
                    for i in range(len(t)):
                        rows.append((build,method,level,i,t[i],z[i]*1000,rms[i],tip_error[i],center_error[i]))
    write_json(root/'audit.json', audits)
    write_json(root/'metrics.json', summaries)
    with (root/'per_frame.csv').open('w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(('build','method','level','frame','time_s','tip_height_mm','rms_vs_L0_mm','tip_vs_L0_mm','max_centerline_vs_L0_mm'))
        w.writerows(rows)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    fig, axes = plt.subplots(2,3,figsize=(10,5.8), sharex=True,sharey=True,layout='constrained')
    for row, method in enumerate(('semi','velpro')):
        for col, i in enumerate((15,30,60)):
            ax = axes[row,col]
            for k, x in enumerate(curves['corrected',method]):
                ax.plot(100*x[i,:,0],100*x[i,:,2],STYLES[k],color=COLORS[k],lw=2.6-.35*k,label=f'L{k}')
            ax.scatter(0,0,c='black',s=17)
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(f'{LABELS[method]}   |   {t[i]:.2f} s')
            if not col: ax.set_ylabel('Height z (cm)')
            if row: ax.set_xlabel('Horizontal x (cm)')
    axes[0,0].legend(ncol=3,loc='lower center',fontsize=8,handlelength=1.5)
    fig.suptitle('Fixed-edge strip  |  Material centerlines, corrected solver',fontsize=15)
    fig.savefig(root/'centerlines.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(9.6,6.3),sharex=True,sharey=True,layout='constrained')
    for row, build in enumerate(('historical','corrected')):
        for col, method in enumerate(('semi','velpro')):
            ax = axes[row,col]
            for k,x in enumerate(curves[build,method]):
                ax.plot(t,100*x[:,-1,2],STYLES[k],color=COLORS[k],lw=2.7-.4*k,label=f'L{k}')
            ax.set_title(f'{LABELS[build]}  |  {LABELS[method]}')
            if not col: ax.set_ylabel('Tip height z (cm)')
            if row: ax.set_xlabel('Time (s)')
    axes[0,0].legend()
    fig.suptitle('Height of the same material point at the free tip',fontsize=15)
    fig.savefig(root/'tip_height.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(7.7,3.7),sharey=True,layout='constrained')
    lines = []
    for ax, method in zip(axes, ('semi','velpro')):
        ax.set(xlim=(-43,43),ylim=(-44,4),xlabel='Horizontal x (cm)',title=LABELS[method],aspect='equal')
        ax.scatter(0,0,c='black',s=17)
        for k,x in enumerate(curves['corrected',method]):
            line, = ax.plot([],[],STYLES[k],color=COLORS[k],lw=3-.5*k,label=f'L{k}')
            lines.append((line,x))
    axes[0].set_ylabel('Height z (cm)')
    axes[0].legend(loc='lower left',fontsize=8)
    title = fig.suptitle('')
    def draw(i):
        for line,x in lines: line.set_data(100*x[i,:,0],100*x[i,:,2])
        title.set_text(f'Corrected solver   |   t = {t[i]:.2f} s')
    FuncAnimation(fig,draw,frames=len(t)).save(root/'strip_swing.gif',writer=PillowWriter(fps=20),dpi=105)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output.resolve())
