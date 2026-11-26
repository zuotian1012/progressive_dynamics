"""Validate rendered media and compare the saved L0/L1 ground trajectories."""
from pathlib import Path
import argparse, json
import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.sparse import load_npz

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    m=json.loads((a.run/'run.json').read_text());lower=Path(m['lower_run']);out=a.run/'report'
    assert json.loads((a.run/'continuation_audit.json').read_text())['passed']
    x=np.load(a.run/'positions.npy');y=np.load(lower/'positions.npy')
    U=load_npz(Path(m['assets'])/f'map_{m["level"]-1}_{m["level"]}.npz')
    residual=np.array([np.linalg.norm(x[k]-(x[0]+U@(y[k]-y[0])),axis=1) for k in range(len(x))])
    comparison=dict(level=m['level'],lower_level=m['level']-1,
        final_displacement_from_linear_lower_transfer_rms_m=float(np.sqrt(np.mean(residual[-1]**2))),
        final_displacement_from_linear_lower_transfer_max_m=float(residual[-1].max()),
        note='Geometric difference from linear transfer of lower displacements; not accuracy error. Video views use the same angle but independently fitted extents.')
    for key,path in [('lower',lower),('current',a.run)]:
        meta=json.loads((path/'run.json').read_text());audit=json.loads((path/'audit.json').read_text());ground=json.loads((path/'ground_audit.json').read_text())
        comparison[key]=dict(vertices=meta['vertices'],faces=meta['faces'],wall_seconds=meta['wall_seconds'],native_steps=meta['native_steps'],preview_steps=meta['preview_steps'],maximum_stretch=audit['maximum_stretch'],minimum_height_m=ground['minimum_vertex_height_m'],zero_motion_steps=audit['zero_motion_steps'])
    (out/'level_comparison.json').write_text(json.dumps(comparison,indent=2))
    caps=[cv2.VideoCapture(str(path/'report'/f'bunny_L{k}_ground_slow.mp4')) for path,k in [(lower,m['level']-1),(a.run,m['level'])]]
    target=out/f'bunny_L{m["level"]-1}_L{m["level"]}_ground_comparison.mp4'
    with imageio.get_writer(target,fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
        for k in range(401):
            frames=[]
            for cap in caps:
                ok,frame=cap.read();assert ok
                frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(960,600),interpolation=cv2.INTER_AREA))
            frame=np.concatenate(frames,axis=1);writer.append_data(frame)
            if k in [0,40,400]:imageio.imwrite(out/f'comparison_{k:03}.png',frame)
    for cap in caps:cap.release()
    validation=[]
    for name,expected in [(f'bunny_L{m["level"]}_ground_1x.mp4',101),(f'bunny_L{m["level"]}_ground_slow.mp4',401),(target.name,401)]:
        cap=cv2.VideoCapture(str(out/name));fps=cap.get(cv2.CAP_PROP_FPS);count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==expected and fps==25
        validation.append(dict(file=name,decoded_frames=count,fps=fps,duration_seconds=count/fps,passed=True))
    (out/'media_validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(comparison,indent=2));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
