"""Matched-time still or short video from accepted checkpoints, labeled as a preview."""
import argparse
import json
import time
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from pathlib import Path
from report_flag_hierarchy import SUITE,METHODS,COLORS,project,raster,font

p=argparse.ArgumentParser();p.add_argument('--level',type=int,required=True);p.add_argument('--frame',type=int,required=True)
p.add_argument('--video',action='store_true',help='Also export the actual trajectory from state 0 through --frame.')
args=p.parse_args();assert 1<=args.frame<=200
entries=json.loads((SUITE/'index.json').read_text())['levels'][str(args.level)]
states=[];full_completed=True
for method in METHODS:
    run=Path(entries[method]['run'])
    full_completed&=json.loads((run/'run.json').read_text())['status']=='completed'
    records=json.loads((run/'steps.json').read_text())
    assert len(records)>=args.frame
    assert all(record['frame']==i+1 and record['native']['converged'] for i,record in enumerate(records[:args.frame]))
    # Copy and close immediately: do not memory-map a live checkpoint file.
    for attempt in range(5):
        try:
            state=np.load(run/'positions.npy')[:args.frame+1].copy()
            assert len(state)==args.frame+1
            break
        except (OSError,ValueError,EOFError):
            if attempt==4:raise
            time.sleep(.2)
    assert np.isfinite(state).all();states.append(state)
root=Path(entries['semi']['root']);mesh=np.load(root/f'assets/mesh_{args.level}.npz')
rest,faces=mesh['vertices'],mesh['faces']
panel=Image.new('RGB',(480,450),'#f3f6fa');pd=ImageDraw.Draw(panel)
uv,_=project(np.array([[0,0,-.65],[0,0,1.42]]));pd.line([tuple(v) for v in uv],fill='#4a6072',width=4)
uv,_=project(np.array([[0,0,.25],[1.5,0,.25],[1.5,0,1.25],[0,0,1.25],[0,0,.25]]));pd.line([tuple(v) for v in uv],fill='#ccd6e0',width=1)
uv,_=project(np.array([[.2,0,1.4],[1.,0,1.4],[.9,0,1.44],[1.,0,1.4],[.9,0,1.36]]));pd.line([tuple(v) for v in uv],fill='#bd852d',width=3)
pins,_=project(np.array([[0,0,.25],[0,0,1.25]]))
def render(frame):
    image=Image.new('RGB',(1440,640),'#f3f6fa');draw=ImageDraw.Draw(image)
    draw.text((35,18),f'FLAG / L{args.level} / PREVIEW AT {frame*.04:.2f} s',font=font(30,True),fill='#163347')
    detail=f'Preview 0-{args.frame*.04:.2f} s of 8.00 s' if args.video else f'Accepted frame {frame} of 200'
    status='Preview of completed simulations' if full_completed else 'Full simulations not yet complete'
    draw.text((35,61),f'{len(rest):,} actual vertices / {len(faces):,} triangles | {detail} | {status}',font=font(18),fill='#416476')
    for column,(trajectory,name) in enumerate(zip(states,['Semi-diagonal','VelPro','Direct (native local)'])):
        state=trajectory[frame]
        draw.text((30+480*column,103),name,font=font(24,True),fill=tuple((COLORS[column]*190).astype(int)))
        tri=state[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
        shade=(.68+.32*np.abs(normal@np.array([.3,-.7,.64])))[:,None]*COLORS[column]
        uv,depth=project(state);assert (uv>=0).all() and (uv<np.array([480,450])).all()
        tile=Image.fromarray(raster(np.array(panel),uv,depth,faces,shade,False));td=ImageDraw.Draw(tile)
        for a,b in pins:td.ellipse((a-4,b-4,a+4,b+4),fill='#cf5158')
        image.paste(tile,(column*480,140))
    draw.text((35,605),'Same physical time and camera | Two fixed corners | +x body wind | Actual positions, displacement scale 1:1',font=font(18),fill='#647788')
    return image

folder=SUITE/'report'/f'L{args.level}';folder.mkdir(exist_ok=True)
output=folder/f'preview_frame_{args.frame:03}.png';render(args.frame).save(output)
print(output)
if args.video:
    prefix=folder/f'preview_through_{args.frame:03}'
    frames=[]
    with imageio.get_writer(prefix.with_suffix('.mp4'),fps=25,codec='libx264',macro_block_size=1,quality=8) as writer:
        for frame in range(args.frame+1):
            image=render(frame);writer.append_data(np.asarray(image))
            frames.append(image.resize((1080,480)))
    frames[0].save(prefix.with_suffix('.gif'),save_all=True,append_images=frames[1:],duration=40,loop=0,optimize=False,disposal=2)
    reader=imageio.get_reader(prefix.with_suffix('.mp4'))
    assert reader.count_frames()==args.frame+1
    reader.get_data(args.frame);reader.close()
    with Image.open(prefix.with_suffix('.gif')) as check:assert check.n_frames==args.frame+1
    prefix.with_suffix('.json').write_text(json.dumps(dict(partial_preview=True,accepted_steps=args.frame,
        physical_duration_s=args.frame*.04,mp4_frames=args.frame+1,gif_frames=args.frame+1,
        all_prefix_steps_converged=True,all_frames_real=True),indent=2),encoding='utf-8')
    print(prefix.with_suffix('.mp4'))
