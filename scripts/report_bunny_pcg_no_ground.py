"""Audit every saved state and render the actual L0 friction pair without a floor."""
from pathlib import Path
import hashlib
import json
import sys
import cv2
import imageio.v2 as imageio
import ipctk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report_bunny_ground_fine import rasterize
from bunny_runtime import runtime_provenance, check_run_sources

OUT = ROOT/'output/bunny_pcg_no_ground_20261124'

def main():
    report = OUT/'report'
    report.mkdir(exist_ok=True)
    data = []
    summaries = []
    for name, mu in [('mu0', 0.), ('mu03', .3)]:
        run = OUT/name
        m = json.loads((run/'run.json').read_text())
        assert m['completed'] and m['completed_steps'] == m['steps'] == 100
        assert m['dt'] == .01 and m['level'] == 0 and not m['ground']['enabled']
        assert m['bunny_linear_solver'] == 'native_pcg'
        assert m['config']['linear_system']['solver'] == 'fused_pcg'
        assert m['contact_friction']['cloth_bunny'] == m['contact_friction']['cloth_self'] == mu
        assert m['config']['contact']['friction']['enable'] == int(mu > 0)
        provenance = runtime_provenance(m['native_package'])
        assert provenance['dll_sha256'] == m['library_sha256']['uipc_backend_cuda.dll']
        check_run_sources(run, m['source_files_sha256'])
        assert json.loads((run/'audit.json').read_text())['accepted_prefix_checks_passed']
        assert json.loads((run/'report/non_descent_audit.json').read_text())['all_accepted_preview_evidence_verified']
        log = run.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
        assert 'BUNNY_DIRECT_SOLVE' not in log
        assert 'LinearFusedPCG:' in log
        import re
        halfplanes = re.findall(r'VertexHalfPlaneTrajectoryFilter PHs: (\d+)', log)
        assert halfplanes and all(int(v) == 0 for v in halfplanes)
        mesh = np.load(Path(m['assets'])/'mesh_0.npz')
        bunny = np.load(Path(m['assets'])/'bunny.npz')
        rest = np.vstack([mesh['vertices'], bunny['vertices']])
        faces = np.vstack([mesh['faces'], bunny['faces']+len(mesh['vertices'])]).astype(np.int32)
        edges = np.unique(np.sort(faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1), axis=0).astype(np.int32)
        collision = ipctk.CollisionMesh(rest, edges, faces)
        x = np.load(run/'positions.npy')
        assert x.shape == (101, 1443, 3) and np.isfinite(x).all()
        intersections = [i for i, state in enumerate(x)
                         if ipctk.has_intersections(collision, np.vstack([state, bunny['vertices']]))]
        assert not intersections, (name, intersections)
        rows = json.loads((run/'steps.json').read_text())
        assert len(rows) == 100 and max(r['obstacle_error'] for r in rows) == 0
        result = dict(friction=mu, completed_steps=100, physical_seconds=1.,
                      newton_converged=m['native_steps'], preview_stops=m['preview_steps'],
                      maximum_stretch=max(r['maximum_stretch'] for r in rows),
                      minimum_height_m=float(x[:,:,2].min()), final_bounds=[x[-1].min(0).tolist(),x[-1].max(0).tolist()],
                      saved_states_checked=101, intersecting_states=intersections,
                      no_ground=True, pcg_confirmed=True, fixed_bunny_error_m=0.,
                      wall_seconds=m['wall_seconds'],
                      positions_sha256=hashlib.sha256((run/'positions.npy').read_bytes()).hexdigest(),
                      limitation='Saved-state intersection checks do not independently certify continuous collision clearance.')
        (run/'no_ground_audit.json').write_text(json.dumps(result, indent=2))
        summaries.append(result)
        data.append((m,x,mesh['faces'],bunny['vertices'][bunny['faces']]))
        print('Audited', name, result, flush=True)
    a,b = (d[0] for d in data)
    cfg0=json.loads(json.dumps(a['config']));cfg1=json.loads(json.dumps(b['config']))
    cfg0['contact']['friction']['enable']=cfg1['contact']['friction']['enable']=0
    assert cfg0 == cfg1 and a['material']==b['material'] and a['bundle_sha256']==b['bundle_sha256']
    np.testing.assert_array_equal(data[0][1][0],data[1][1][0])
    az, el = np.deg2rad([-55, 28])
    camera = np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    right = np.array([-np.sin(az),np.cos(az),0.])
    basis = np.stack([right,np.cross(camera,right),camera],axis=1)
    width,height=640,570
    gif=[]
    snapshots=[]
    with imageio.get_writer(report/'comparison.mp4', fps=25, codec='libx264', macro_block_size=1) as writer:
        for frame in range(101):
            canvas=np.full((650,1280,3),246,np.uint8)
            bounds=np.vstack([arr.reshape(-1,3)@basis[:,:2] for _,x,_,bt in data for arr in [x[frame],bt]])
            low,high=bounds.min(0),bounds.max(0)
            center=(low+high)/2
            span=np.array([width,height])*max((high-low)/[width,height])*1.09
            for k,(m,x,faces,bt) in enumerate(data):
                tri=np.vstack([x[frame][faces],bt])
                normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
                shade=.55+.45*np.abs(normals@np.array([.3,-.7,.64]))
                colors=np.vstack([np.tile([.11,.45,.62],(len(faces),1)),np.tile([.79,.76,.70],(len(bt),1))])
                rgba=rasterize(tri@basis, colors*shade[:,None], center-span/2,center+span/2,width,height,
                               np.ascontiguousarray(x[0][faces][:,:,:2]),shade[:len(faces),None]*np.array([.14,.63,.74]))[::-1]
                canvas[45:45+height,k*width:(k+1)*width]=np.where(rgba[:,:,3:]!=0,rgba[:,:,:3],246)
                label=f'L0 | friction = {m["contact_friction"]["cloth_bunny"]:g} | t = {frame*.01:.2f} s'
                cv2.putText(canvas,label,(k*width+20,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(23,47,58),1,cv2.LINE_AA)
            cv2.putText(canvas,'GPU PCG | No ground | Actual L0 mesh | Playback: 4x slow',(24,636),cv2.FONT_HERSHEY_SIMPLEX,.6,(60,70,80),1,cv2.LINE_AA)
            writer.append_data(canvas)
            if frame%2==0:gif.append(cv2.resize(canvas,(960,488),interpolation=cv2.INTER_AREA))
            if frame in [0,20,40,60,80,100]:
                imageio.imwrite(report/f'frame_{frame:03d}.png',canvas)
                snapshots.append(cv2.resize(canvas,(960,488),interpolation=cv2.INTER_AREA))
            if frame==100:imageio.imwrite(report/'final.png',canvas)
    imageio.mimsave(report/'comparison.gif',gif,duration=80,loop=0)
    imageio.imwrite(report/'contact_sheet.png',np.vstack(snapshots))
    cap=cv2.VideoCapture(str(report/'comparison.mp4'))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==101
    decoded=0
    while True:
        ok,_=cap.read()
        if not ok:break
        decoded+=1
    cap.release()
    assert decoded==101
    summary=dict(runs=summaries,only_physical_difference='cloth-bunny and cloth-self friction coefficient',
                 video_frames=101,video_fps=25,playback='4x slow',camera='fixed angle; shared center and scale fitted each frame; both trajectories fully visible')
    (report/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Rendered',report,flush=True)

if __name__=='__main__':main()
