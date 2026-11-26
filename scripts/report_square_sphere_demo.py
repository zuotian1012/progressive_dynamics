"""Audit physical trajectories and render a synchronized four-level comparison."""
import argparse
import csv
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import ipctk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from numba import njit
from scipy.sparse import load_npz

from run_square_sphere_demo import DT, STEPS, ROOT, run_path, write_json


def main(root):
    global STEPS
    scene = json.loads((root/'scene.json').read_text())
    STEPS = scene['steps']
    sphere = dict(np.load(root/'assets/sphere.npz'))
    out = root/'report'
    out.mkdir(exist_ok=True)
    data, audits, rows = {}, {}, []
    shared_config = json.loads((root/'shared_L0/run.json').read_text())['config']
    for method in ('semi', 'velpro'):
        for level in range(4):
            path = run_path(root, method, level)
            meta = json.loads((path/'run.json').read_text())
            steps = json.loads((path/'steps.json').read_text())
            x = np.load(path/'positions.npy')
            if meta['status'] != 'completed' or len(x) != STEPS+1 or len(steps) != STEPS:
                raise ValueError(f'Incomplete trajectory: {path}')
            if meta['config'] != shared_config or meta['friction'] != scene['friction'] or meta['requested_contact_stiffness'] != scene['requested_contact_stiffness']:
                raise ValueError(f'Inconsistent physical settings: {path}')
            mesh = dict(np.load(root/'assets'/f'mesh_{level}.npz'))
            rest, f, pins = mesh['vertices'], mesh['faces'], mesh['pin_indices']
            edges = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
            if path.name not in audits:
                combined_rest = np.vstack((rest, sphere['vertices']))
                combined_faces = np.vstack((f, sphere['faces']+len(rest))).astype(np.int32)
                combined_edges = np.unique(np.sort(combined_faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
                cm = ipctk.CollisionMesh(combined_rest, combined_edges, combined_faces)
                intersections = [i for i,y in enumerate(x) if ipctk.has_intersections(cm,np.vstack((y,sphere['vertices'])))]
                pin_error = 0.
                if intersections:
                    raise ValueError(f'Intersecting cloth/sphere geometry: {path}: {intersections}')
                log = (path/'native.log').read_text(errors='replace')
                if 'clamped' in log:
                    raise ValueError(f'Contact stiffness was changed by backend: {path}')
                systems = json.loads((path/'backend/systems.json').read_text())['sim_systems']
                names = {s['name'].split('::')[-1] for s in systems if s['valid']}
                if not {'IPCSimplexNormalContact', 'IPCSimplexFrictionalContact'} <= names or 'SoftPositionConstraint' in names:
                    raise ValueError(f'Unexpected native contact or constraint systems: {path}')
                count = scene['analytic_freefall_steps']
                analytic = np.broadcast_to(rest,(count+1,*rest.shape)).copy()
                n = np.arange(count+1)
                analytic[:,:,2] -= (.5*9.81*DT**2*n*(n+1))[:,None]
                analytic_error = float(np.max(np.abs(x[:count+1]-analytic)))
                if analytic_error > 1e-12:
                    raise ValueError(f'Incorrect free-fall prefix: {path}')
                if not np.isfinite(x).all() or pin_error > 1e-10:
                    raise ValueError(f'Invalid saved geometry: {path}')
                audits[path.name] = dict(accepted_frames=len(x)-1, intersection_frames=intersections, analytic_freefall_steps=scene['analytic_freefall_steps'],
                    analytic_prefix_error_m=analytic_error, native_normal_contact=True, native_friction=True,
                    max_sphere_error_m=max(s['sphere_error_m'] for s in steps),
                    max_pin_error_m=pin_error, max_stretch=max(s['stretch'] for s in steps),
                    newton_converged_frames=sum(bool(s['native'] and s['native']['converged']) for s in steps),
                    coarse_preview_frames=sum(bool(s['native'] and s['native']['preview_non_descent']) for s in steps),
                    elapsed_s=meta['elapsed_s'])
                print(f'Audited {path.name}: {len(intersections)} intersecting cloth/sphere frames', flush=True)
            tip = np.flatnonzero(np.isclose(mesh['uv'][:, 0], .5) & np.isclose(mesh['uv'][:, 1], 0))[0]
            angle = x[:,:,2].mean(axis=1)
            tri = rest[f]
            mass = np.zeros(len(rest))
            area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
            np.add.at(mass, f.ravel(), np.repeat(area/3, 3))
            if level:
                U = load_npz(root/'assets'/f'map_{level-1}_{level}.npz')
                parent = data[method, level-1]['x']
                diff = x-np.asarray([U @ y for y in parent])
                rms = np.sqrt(np.einsum('i,tij,tij->t', mass/mass.sum(), diff, diff))*1000
            else:
                rms = np.zeros(len(x))
            data[method, level] = dict(x=x, mesh=mesh, angle=angle, rms=rms)
            for i in range(len(x)):
                rows.append((method, level, i, i*DT, angle[i], x[i, tip, 1], x[i, tip, 2], rms[i]))
    write_json(out/'audit.json', audits)
    with (out/'per_frame.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('method', 'level', 'frame', 'time_s', 'vertex_mean_height_m', 'bottom_center_y_m', 'bottom_center_z_m', 'rms_vs_prolongated_parent_mm'))
        writer.writerows(rows)
    differences = {}
    for level in range(4):
        distance = np.linalg.norm(data['semi', level]['x']-data['velpro', level]['x'], axis=2)
        differences[f'L{level}'] = dict(max_vertex_distance_mm=float(distance.max()*1000),
            final_vertex_rms_mm=float(np.sqrt(np.mean(distance[-1]**2))*1000))
    write_json(out/'method_differences.json', differences)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = ['#586578', '#ca7045', '#15888e', '#695daf']
    for row, method in enumerate(('semi', 'velpro')):
        for level in range(4):
            d = data[method, level]
            axes[row, 0].plot(np.arange(STEPS+1)*DT, d['angle'], color=colors[level], label=f'L{level}')
            if level:
                axes[row, 1].plot(np.arange(STEPS+1)*DT, d['rms'], color=colors[level], label=f'L{level}')
        axes[row, 0].set_title(('PD / Semi-diagonal' if method == 'semi' else 'PD++ / VelPro')+' : descent')
        axes[row, 1].set_title('Coarse/fine position difference\n(not reference-solution error)', fontsize=11)
        axes[row, 0].set_ylabel('Mean vertex height (m)')
        axes[row, 1].set_ylabel('Mass-weighted RMS (mm)')
        for ax in axes[row]:
            ax.set_xlabel('Physical time (s)')
            ax.legend(ncol=4)
            ax.grid(alpha=.2)
    fig.savefig(out/'metrics.png', dpi=150)
    plt.close(fig)

    render(data, sphere, out, audits, fine_only=False)
    render(data, sphere, out, audits, fine_only=True)
    print(out, flush=True)


@njit(cache=True)
def rasterize(triangles, depths, colors, sphere_count, size):
    """Orthographic per-pixel depth test, retaining the original triangles."""
    img = np.empty((size,size,3), dtype=np.uint8)
    img[:,:,0] = 244; img[:,:,1] = 246; img[:,:,2] = 249
    zbuffer = np.full((size,size), -np.inf)
    for face in range(len(triangles)):
        p = triangles[face].copy()
        for j in range(3):
            p[j,0] = (p[j,0]+.79)/1.58*size
            p[j,1] = (.79-p[j,1])/1.49*size
        xmin=max(0,int(np.floor(np.min(p[:,0])))); xmax=min(size-1,int(np.ceil(np.max(p[:,0]))))
        ymin=max(0,int(np.floor(np.min(p[:,1])))); ymax=min(size-1,int(np.ceil(np.max(p[:,1]))))
        den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den)<1e-15:
            continue
        for row in range(ymin,ymax+1):
            for col in range(xmin,xmax+1):
                px,py=col+.5,row+.5
                a=((p[1,1]-p[2,1])*(px-p[2,0])+(p[2,0]-p[1,0])*(py-p[2,1]))/den
                b=((p[2,1]-p[0,1])*(px-p[2,0])+(p[0,0]-p[2,0])*(py-p[2,1]))/den
                c=1-a-b
                if a<0 or b<0 or c<0:
                    continue
                z=a*depths[face,0]+b*depths[face,1]+c*depths[face,2]
                if z<=zbuffer[row,col]:
                    continue
                zbuffer[row,col]=z
                shade=.83 if face>=sphere_count and min(a,b,c)<.008 else 1.
                for channel in range(3):
                    img[row,col,channel]=min(255.,max(0.,255*colors[face,channel]*shade))
    return img


def render(data, sphere, out, audits, fine_only, methods=('semi','velpro')):
    direction = np.array([1.4, -3., 1.9]); direction /= np.linalg.norm(direction)
    right = np.cross([0,0,1], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    basis = np.column_stack((right, up))
    center = np.array([0.,0.,.65])
    project = lambda p: (p-center) @ basis
    light = np.array([-.3,-1.,1.5]); light /= np.linalg.norm(light)
    fig, axes = plt.subplots(1 if fine_only else 2, len(methods) if fine_only else 4,
        figsize=(6*len(methods),7) if fine_only else (16,10), dpi=100, facecolor='#f4f6f9', squeeze=False)
    fig.subplots_adjust(left=.025,right=.985,top=.79,bottom=.11,hspace=.28,wspace=.04)
    title = 'CLOTH DROP ON SPHERE' + (' / L3' if fine_only else ' / ALL LEVELS')
    fig.text(.04,.945,title,fontsize=21,weight='bold',color='#213147')
    fig.text(.04,.892,'1 m square | fixed sphere R = 0.30 m | friction = 0.3 | gravity only',fontsize=12,color='#526078')
    label = fig.text(.96,.945,'',ha='right',fontsize=15,color='#213147')
    caption = 'Semi / VelPro share L0; Direct evolves independently on L3' if 'direct' in methods else 'Shared L0'
    fig.text(.04,.056,caption+' | identical material and contact settings | actual meshes',fontsize=10,color='#526078')
    fig.text(.04,.025,'0–0.22 s: exact discrete free fall; afterwards: native contact solve. No geometric smoothing.',fontsize=9,color='#526078')
    items = [(m,3) for m in methods] if fine_only else [(m,k) for m in methods for k in range(4)]
    polys = []
    sphere_tri = sphere['vertices'][sphere['faces']]
    for ax,(method,level) in zip(axes.flat,items):
        d=data[method,level]; mesh=d['mesh']; faces=mesh['faces']
        ax.set_facecolor('#f4f6f9'); ax.set(xlim=(-.79,.79),ylim=(-.70,.79),aspect='equal'); ax.axis('off')
        name={'semi':'Semi-diagonal','velpro':'VelPro','direct':'Direct / independent'}[method]
        ax.set_title(f'{name}  /  L{level}\n{len(mesh["vertices"]):,} vertices',fontsize=12,color='#34465c',pad=5)
        poly=ax.imshow(np.full((8,8,3),[244,246,249],dtype=np.uint8),extent=(-.79,.79,-.70,.79),origin='upper')
        uv=mesh['uv'][faces].mean(axis=1)
        checker=(np.floor(uv[:,0]*4)+np.floor(uv[:,1]*4)).astype(int)%2
        cloth_base=np.where(checker[:,None]==0,[.12,.51,.62],[.32,.72,.77])
        base=np.vstack((np.tile([.86,.83,.77],(len(sphere_tri),1)),cloth_base))
        polys.append((poly,d,base))
    tag='direct_comparison' if 'direct' in methods else ('finest' if fine_only else 'all_levels')
    ids=np.unique(np.r_[np.arange(0,STEPS+1,4),STEPS]).astype(int)
    gifs=[]
    with imageio.get_writer(out/f'{tag}.mp4',fps=25,codec='libx264',macro_block_size=1,quality=8) as writer:
        for frame in ids:
            for poly,d,base in polys:
                tri=np.concatenate((sphere_tri,d['x'][frame][d['mesh']['faces']]))
                normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
                # Sphere uses outward lighting; cloth is rendered two-sided.
                lighting=np.abs(normals@light)
                lighting[:len(sphere_tri)]=np.maximum(0,normals[:len(sphere_tri)]@light)
                colors=base*(.58+.42*lighting[:,None])
                poly.set_data(rasterize(project(tri),tri@direction,colors,len(sphere_tri),800 if fine_only else 600))
            label.set_text(f't = {frame*DT:.2f} s')
            fig.canvas.draw()
            image=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
            writer.append_data(image)
            gifs.append(image if fine_only else image[::2,::2])
            if frame==ids[min(len(ids)-1,round(.8/(4*DT)))]:
                fig.savefig(out/f'{tag}_poster.png',dpi=130)
        fig.savefig(out/f'{tag}_final.png',dpi=130)
    imageio.mimsave(out/f'{tag}_half_speed.gif',gifs,duration=80,loop=0)
    plt.close(fig)
    write_json(out/f'{tag}_render.json',dict(frames=ids.tolist(),video_fps=25,gif_frame_ms=80,
        camera_direction=direction.tolist(),geometry='actual accepted positions; no smoothing',
        coarse_preview_frames=sum(a['coarse_preview_frames'] for a in audits.values())))
    print(f'Rendered {tag}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=ROOT/'output/square_sphere_demo')
    main(parser.parse_args().root.resolve())
