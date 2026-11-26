"""Compare audited adjacent-level ablations sharing one parent trajectory (SI units)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.sparse import load_npz
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from progressive.prolongation import ShellProlongation, frames
from report_bunny_ground_fine import rasterize


def read_run(path):
    meta = json.loads((path/'run.json').read_text())
    if not meta['completed'] or not json.loads((path/'continuation_audit.json').read_text())['passed']:
        raise ValueError(f'Expected an audited, complete run: {path}')
    states = np.load(path/'positions.npy', mmap_mode='r')
    if len(states) != meta['completed_steps'] + 1:
        raise ValueError(f'Incomplete trajectory: {path}')
    return meta, states


def near_bunny(points, triangles, threshold):
    centers = triangles.mean(axis=1)
    radius = np.linalg.norm(triangles-centers[:, None], axis=2).max()
    candidates = cKDTree(centers).query_ball_point(points, radius + threshold)
    counts = np.fromiter(map(len, candidates), int)
    selected = np.zeros(len(points), dtype=bool)
    if counts.sum():
        ids = np.repeat(np.arange(len(points)), counts)
        faces = np.concatenate([group for group in candidates if len(group)])
        nearest = closest_point(triangles[faces], points[ids])
        selected[ids[np.linalg.norm(nearest-points[ids], axis=1) <= threshold]] = True
    return selected


def mass_fraction(rest, faces):
    area = frames(rest, faces)[3]
    mass = np.bincount(faces.ravel(), weights=np.repeat(area/3, 3), minlength=len(rest))
    return mass/mass.sum()


def movie(out, panels, bunny, dt, measure_silhouettes=False, columns=None):
    azimuth, elevation = np.deg2rad([-55, 28])
    camera = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    basis = np.stack([right, np.cross(camera, right), camera], axis=1)
    width, height = 540, 360
    projected_bunny = bunny.reshape(-1,3) @ basis[:,:2]
    bunny_bounds = np.array([projected_bunny.min(0),projected_bunny.max(0)])
    colors, material_uv = [], []
    for _, states, faces in panels:
        material_uv.append(np.ascontiguousarray(states[0][faces][:,:,:2]))
        colors.append(np.vstack([np.tile([.11,.45,.62], (len(faces),1)), np.tile([.79,.76,.70], (len(bunny),1))]))
    if columns is None:
        columns = min(len(panels), 4 if len(panels)>4 else 2)
    rows = (len(panels)+columns-1)//columns
    canvas_height = rows*(height+35)
    canvas_height += canvas_height % 2  # yuv420p requires even dimensions.
    last = len(panels[0][1])-1
    overlaps = []
    with imageio.get_writer(out/'comparison.mp4', fps=25, codec='libx264', macro_block_size=1) as writer:
        for frame in np.unique(np.r_[np.arange(0,last+1,4),last]):
            canvas = np.full((canvas_height, columns*width, 3), 246, np.uint8)
            # Fit one common camera to this time instant so sliding on the floor
            # later in the run does not shrink the contact phase to a thumbnail.
            bounds = [bunny_bounds]
            for _, states, _ in panels:
                projected = states[frame] @ basis[:,:2]
                bounds.append([projected.min(0),projected.max(0)])
            low, high = np.min(bounds,axis=0)[0], np.max(bounds,axis=0)[1]
            center = (low+high)/2
            span = np.array([width,height])*np.max((high-low)/[width,height])*1.07
            for index, (label, states, faces) in enumerate(panels):
                tri = np.vstack([states[frame][faces], bunny]); normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                normal /= np.maximum(1e-30, np.linalg.norm(normal,axis=1)[:,None])
                shade = .55+.45*np.abs(normal @ np.array([.3,-.7,.64]))
                rgba = rasterize(tri@basis, colors[index]*shade[:,None], center-span/2, center+span/2, width, height,
                                 material_uv[index],shade[:len(faces),None]*np.array([.14,.63,.74]))[::-1]
                if measure_silhouettes:
                    # In the depth-tested render both cloth colors have B > R.
                    # The bunny has R > B, and empty pixels have R = B = 0.
                    visible_cloth = rgba[:, :, 2] > rgba[:, :, 0]
                    if index == 0:
                        reference_cloth = visible_cloth
                    union = np.count_nonzero(reference_cloth | visible_cloth)
                    intersection = np.count_nonzero(reference_cloth & visible_cloth)
                    overlaps.append((frame, frame*dt, index, intersection, union,
                                     intersection / union if union else 1.0))
                top, left = (index//columns)*(height+35), (index%columns)*width
                canvas[top+35:top+35+height,left:left+width] = np.where(rgba[:,:,3:]!=0, rgba[:,:,:3], 246)
                cv2.putText(canvas, f'{label}   |   {frame*dt:.2f} s', (left+15,top+26), cv2.FONT_HERSHEY_SIMPLEX,.67,(23,47,58),1,cv2.LINE_AA)
            writer.append_data(canvas)
            if frame==last:imageio.imwrite(out/'final.png',canvas)
    if measure_silhouettes:
        with (out/'silhouettes.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('frame','time_s','level','intersection_pixels','union_pixels','iou_from_L0'))
            writer.writerows(overlaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lower',type=Path);parser.add_argument('fine',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--movie',action='store_true')
    parser.add_argument('--compare-newton-tolerance',action='store_true',help='Compare recorded Newton tolerance controls sharing the same parent')
    parser.add_argument('--ancestor',type=Path,help='Include the verified parent of the reference and measure drift from its successively prolonged trajectory')
    args = parser.parse_args();args.output.mkdir(exist_ok=True, parents=True)
    meta, lower = read_run(args.lower); assets=Path(meta['assets'])
    if meta['scene']!='bunny':raise ValueError('Expected bunny parent')
    parent_level=meta['level'];child_level=parent_level+1
    with (args.lower/'positions.npy').open('rb') as source: lower_hash=hashlib.file_digest(source,'sha256').hexdigest()
    coarse=np.load(assets/f'mesh_{parent_level}.npz');fine=np.load(assets/f'mesh_{child_level}.npz');bunny=np.load(assets/'bunny.npz')
    pair=f'{parent_level}_{child_level}'
    transfer=ShellProlongation(coarse['vertices'],fine['vertices'],coarse['faces'],load_npz(assets/f'map_{pair}.npz'),np.load(assets/f'anchors_{pair}.npy'))
    targets=np.array([transfer.position(state) for state in lower]); bunny_tri=bunny['vertices'][bunny['faces']]
    panels=[];summary=[];timeseries=[]
    if args.ancestor:
        ancestor_meta,ancestor=read_run(args.ancestor)
        ancestor_level=ancestor_meta['level']
        with (args.ancestor/'positions.npy').open('rb') as source:
            if parent_level!=ancestor_level+1 or meta['lower_sha256']!=hashlib.file_digest(source,'sha256').hexdigest():
                raise ValueError('Ancestor is not the reference trajectory source')
        ancestor_mesh=np.load(assets/f'mesh_{ancestor_level}.npz');pair=f'{ancestor_level}_{parent_level}'
        first=ShellProlongation(ancestor_mesh['vertices'],coarse['vertices'],ancestor_mesh['faces'],load_npz(assets/f'map_{pair}.npz'),np.load(assets/f'anchors_{pair}.npy'))
        ancestor_parent=first.position(ancestor[len(lower)-1]);ancestor_child=transfer.position(ancestor_parent)
        panels.append((f'L{ancestor_level}  reference',ancestor[:len(lower)],ancestor_mesh['faces']))
    for path in [args.lower,*args.fine]:
        run, states=read_run(path);level=run['level'];weight=run.get('consistency_weight',0.)
        child=path!=args.lower
        expected_cfg = json.loads(json.dumps(meta['config']))
        if child and run.get('newton_iteration_budgets'):
            budgets = run['newton_iteration_budgets']
            if budgets['previous_level'] != expected_cfg['newton']['max_iter']:
                raise ValueError(f'Inconsistent Newton budget provenance: {path}')
            expected_cfg['newton']['max_iter'] = budgets['current_level']
        tol = run['config']['newton']['velocity_tol']
        if child and args.compare_newton_tolerance:
            if run.get('bunny_newton_velocity_tol',tol)!=tol or not 0<tol<=expected_cfg['newton']['velocity_tol']:
                raise ValueError(f'Invalid Newton tolerance control: {path}')
            expected_cfg['newton']['velocity_tol']=tol
        if child and (level!=child_level or run['lower_sha256']!=lower_hash or len(states)!=len(lower) or
            run['config']!=expected_cfg or any(run[key]!=meta[key] for key in ['dt','assets','ground','contact_friction','material'])):
            raise ValueError(f'Not a paired L{child_level} comparison: {path}')
        faces=(fine if child else coarse)['faces'];mass=mass_fraction(states[0],faces)
        residual=np.sqrt(np.einsum('tvi,tvi,v->t',states-targets,states-targets,mass)) if child else np.zeros(len(states))
        floor=(states[:,:,2]<.01)@mass;near=near_bunny(states[-1],bunny_tri,.005)&(states[-1,:,2]>.01)
        label=f'L{level}  {run["method"]}  w={weight:g}' if child or parent_level else 'L0  reference'
        if run.get('velocity_map')=='psq-jacobian':
            label+='  DP'
        if child and args.compare_newton_tolerance:
            label+=f'  tol={tol:g}'
        panels.append((label,states,faces));summary.append(dict(run=str(path.resolve()),label=label,method=run['method'],
            peak_rms_m=float(residual.max()),peak_rms_frame=int(residual.argmax()),
            final_rms_m=float(residual[-1]),mean_frame_rms_m=float(residual.mean()),final_floor_mass_pct=100*floor[-1],
            final_near_bunny_off_floor_mass_pct=100*mass[near].sum(),final_max_z_m=float(states[-1,:,2].max())))
        if args.ancestor:
            delta=states[-1]-(ancestor_child if child else ancestor_parent)
            summary[-1]['final_rms_from_successive_ancestor_transfer_m']=float(np.sqrt(np.einsum('vi,vi,v->',delta,delta,mass)))
        timeseries.extend((path.name,level,weight,frame*meta['dt'],err,100*floor_mass) for frame,(err,floor_mass) in enumerate(zip(residual,floor)))
    with (args.output/'metrics.csv').open('w',newline='') as dest:
        writer=csv.writer(dest);writer.writerow(['run','level','w','time_s','mass_weighted_rms_m','mass_within_1cm_floor_pct']);writer.writerows(timeseries)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    if args.movie:movie(args.output,panels,bunny_tri,meta['dt'])


if __name__=='__main__':main()
