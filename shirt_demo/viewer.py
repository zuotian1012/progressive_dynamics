"""Interactive synchronized playback of measured IPC runs (rendering is untimed)."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import polyscope as ps
from polyscope import imgui
import trimesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="+",type=Path)
    parser.add_argument("--snapshot",type=Path,help="Render the final frame to an image and exit")
    parser.add_argument("--movie",type=Path,help="Export synchronized GIF playback and exit")
    args=parser.parse_args()
    args.runs=[p.resolve() for p in args.runs]
    if args.snapshot:
        args.snapshot=args.snapshot.resolve()
    if args.movie:
        args.movie=args.movie.resolve()
    runs=[]
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_background_color((.95,.96,.98))
    end=None
    for i,path in enumerate(args.runs):
        meta=json.loads((path/'run.json').read_text())
        x=np.load(path/'positions.npy',mmap_mode='r')
        transforms=np.load(path/'robot.npy',mmap_mode='r')
        end=meta['completed_steps'] if end is None else min(end,meta['completed_steps'])
        offset=np.array([0,1.25*(i-(len(args.runs)-1)/2),0])
        name=f"{meta['method']} L{meta['level']} ({meta['faces']} faces)"
        mesh=ps.register_surface_mesh(name,x[0]+offset,np.load(path/'faces.npy'),color=(.12,.43,.68),smooth_shade=True)
        robot=[]
        for j in range(8):
            with np.load(path/f'collider_{j}.npz') as g:
                v,f=g['vertices'],g['faces']
            t=transforms[0,j]
            drawable=ps.register_surface_mesh(f'{i}/jaw{j}',v@t[:3,:3].T+t[:3,3]+offset,f,color=(.5,.55,.6),smooth_shade=False)
            robot.append((v,drawable))
        table=trimesh.creation.box(extents=(1,1.05,.03))
        ps.register_surface_mesh(f'{i}/table',table.vertices+[.65,0,.785]+offset,table.faces,color=(.8,.83,.87))
        runs.append((meta,x,transforms,mesh,robot,offset))
    ps.set_view_projection_mode("orthographic")
    ps.look_at((2.15,0,2.2),(.65,0,.85))
    state={'step':0,'playing':False,'speed':1.,'last':time.perf_counter(),'wire':False}

    def update_geometry():
        k=state['step']
        for meta,x,transforms,mesh,robot,offset in runs:
            mesh.update_vertex_positions(x[k]+offset)
            for j,(v,drawable) in enumerate(robot):
                t=transforms[k,j]
                drawable.update_vertex_positions(v@t[:3,:3].T+t[:3,3]+offset)

    def callback():
        now=time.perf_counter()
        imgui.TextUnformatted('Scene527 | physical gripper contact | synchronized playback')
        if imgui.Button('Pause' if state['playing'] else 'Play'):
            state['playing']=not state['playing']
        imgui.SameLine()
        if imgui.Button('Restart'):
            state['step']=0
            update_geometry()
        changed,step=imgui.SliderInt('Physics step',state['step'],0,end)
        if changed:
            state['step']=step
            update_geometry()
        changed,speed=imgui.SliderFloat('Playback speed',state['speed'],.1,3.)
        if changed:
            state['speed']=speed
        changed,wire=imgui.Checkbox('Show cloth mesh',state['wire'])
        if changed:
            state['wire']=wire
            for _,_,_,mesh,_,_ in runs:
                mesh.set_edge_width(.6 if wire else 0.)
        if state['playing']:
            delta=int((now-state['last'])*state['speed']/runs[0][0]['dt'])
            if delta:
                state['step']=min(end,state['step']+delta)
                state['last']=now
                update_geometry()
            if state['step']==end:
                state['playing']=False
        else:
            state['last']=now
        imgui.TextUnformatted(f"Simulation time: {state['step']*runs[0][0]['dt']:.3f} s")
        for meta,*_ in runs:
            imgui.TextUnformatted(f"{meta['method']} L{meta['level']}: {meta['physics_seconds']:.1f} s solve; {meta['physics_steps_per_second']:.2f} steps/s")
        imgui.TextUnformatted('Playback speed is independent of recorded simulation throughput.')
    ps.set_user_callback(callback)
    if args.movie:
        from PIL import Image, ImageDraw, ImageFont
        ps.set_window_size(1500,650)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
        small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
        titles=['粗网格预览','粗 → 细：渐进细化','直接细网格仿真']
        for _,_,_,mesh,_,_ in runs:
            mesh.set_edge_width(.35)
        frames=[]
        for k in np.unique(np.linspace(0,end,130,dtype=int)):
            state['step']=int(k)
            update_geometry()
            raw=Image.fromarray(ps.screenshot_to_buffer(transparent_bg=False,include_UI=False)).convert('RGB')
            # Fixed viewport crops enlarge the three equally spaced orthographic scenes.
            # The same crop is used at every time; no per-frame zoom or geometric changes.
            image=Image.new('RGB',raw.size,'white')
            if len(runs)==3:
                rw,rh=raw.size
                for i,center in enumerate((.274,.5,.726)):
                    tile=raw.crop((round((center-.11)*rw),round(.23*rh),
                                   round((center+.11)*rw),round(.738*rh)))
                    tile=tile.resize((rw//3,500),Image.Resampling.LANCZOS)
                    image.paste(tile,(i*rw//3,100))
            else:
                image=raw
            draw=ImageDraw.Draw(image)
            w,h=image.size
            draw.rectangle((0,0,w,105),fill='white')
            for i,(meta,*_) in enumerate(runs):
                center=(i+.5)*w/len(runs)
                label=titles[i] if i<len(titles) else meta['method']
                draw.text((center,22),label,font=font,fill='#17324d',anchor='mm')
                cost=f"{meta['faces']:,} 面  |  求解 {meta['physics_seconds']:.1f} 秒"
                if meta['method']=='velpro_inspired' and i>0:
                    cost+=f"（含粗层 {meta['physics_seconds']+runs[i-1][0]['physics_seconds']:.1f} 秒）"
                draw.text((center,58),cost,font=small,fill='#42576b',anchor='mm')
            draw.rectangle((0,h-45,w,h),fill='white')
            draw.text((25,h-26),f"仿真时间 {k*runs[0][0]['dt']:.2f} 秒  ·  实体夹爪 + 接触摩擦  ·  三组同步回放",font=small,fill='#17324d',anchor='lm')
            frames.append(image)
        frames[0].save(args.movie,save_all=True,append_images=frames[1:],duration=100,loop=0)
        frames[len(frames)*2//3].save(args.movie.with_suffix('.png'))
        return
    if args.snapshot:
        state['step']=end
        update_geometry()
        ps.set_window_size(1500,650)
        ps.screenshot(str(args.snapshot),transparent_bg=False,include_UI=False)
        return
    ps.show()


if __name__=='__main__':
    main()
