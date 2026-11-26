"""Render matched-time, matched-scale flat vs perturbed L0 runs."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--wind',action='store_true');group.add_argument('--pins',action='store_true')
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--variant-root',type=Path)
    parser.add_argument('--metric',choices=['out_of_plane','free_edge_height'],default='out_of_plane')
    args=parser.parse_args();wind_comparison=args.wind;pin_comparison=args.pins
    roots=([ROOT/'output/flag_L0_seed1mm_20261122',ROOT/'output/flag_L0_seed1mm_wind150_20261122'] if wind_comparison
        else [ROOT/'output/flag_L0_parallel_wind_20261121',ROOT/'output/flag_L0_seed1mm_20261122'])
    if pin_comparison:roots=[ROOT/'output/flag_L0_seed1mm_wind150_20261122',ROOT/'output/flag_L0_two_pins_wind150_20261122']
    if bool(args.baseline_root)!=bool(args.variant_root):parser.error('Provide both --baseline-root and --variant-root')
    if args.baseline_root:roots=[args.baseline_root.resolve(),args.variant_root.resolve()]
    out=roots[1]/'report';out.mkdir(exist_ok=True)
    metas=[json.loads((r/'L0/run.json').read_text()) for r in roots]
    audits=[json.loads((r/'report/audit.json').read_text()) for r in roots]
    assert metas[0]['config']==metas[1]['config'] and metas[0]['runtime_sha256']==metas[1]['runtime_sha256']
    hashes={}
    for name in ['assets/mesh_0.npz','operator/operator.npz','bundle.bin']:
        h=[hashlib.sha256((r/name).read_bytes()).hexdigest() for r in roots]
        assert h[0]==h[1];hashes[name]=h[0]
    force=[json.loads((r/'L0/applied_wind.json').read_text()) for r in roots]
    if wind_comparison:
        ratio=metas[1]['wind_scale']/metas[0].get('wind_scale',1.)
        assert ratio>1 and metas[0]['pins']==metas[1]['pins']
        for a,b in zip(*force):
            assert a['frame']==b['frame'] and a['t_s']==b['t_s']
            np.testing.assert_allclose(b['acceleration_x_m_s2'],ratio*a['acceleration_x_m_s2'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(b['total_force_N'],ratio*np.array(a['total_force_N']),rtol=0,atol=1e-12)
    elif pin_comparison:
        assert [m['pins'] for m in metas]==[140,2]
        for a,b in zip(*force):
            assert a['frame']==b['frame'] and a['t_s']==b['t_s']
            assert a['acceleration_x_m_s2']==b['acceleration_x_m_s2']
    else:assert force[0]==force[1]
    xs=[np.load(r/'L0/positions.npy') for r in roots]
    assert all(x.shape==(201,911,3) for x in xs)
    if wind_comparison or pin_comparison:np.testing.assert_array_equal(xs[0][0],xs[1][0])
    d=np.load(roots[0]/'assets/mesh_0.npz');rest,f=d['vertices'],d['faces']
    left=np.isclose(rest[:,0],0,atol=1e-12)
    ends=left&(np.isclose(rest[:,2],.25,atol=1e-12)|np.isclose(rest[:,2],1.25,atol=1e-12))
    if pin_comparison:
        tri=rest[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        mass=np.zeros(len(rest));np.add.at(mass,f.ravel(),np.repeat(area*.0004*200/3,3))
        for pin,records in zip([left,ends],force):
            for record in records:
                np.testing.assert_allclose(record['total_force_N'],[mass[~pin].sum()*record['acceleration_x_m_s2'],0,0],rtol=0,atol=1e-12)
    t=np.arange(201)*.04;ys=[np.abs(x[:,:,1]).max(axis=1) for x in xs]
    right=np.isclose(rest[:,0],1.5,atol=1e-12)
    free_edges=[x[:,right].mean(axis=1) for x in xs]
    chart_values=[edge[:,2] for edge in free_edges] if args.metric=='free_edge_height' else ys
    metrics=dict(same_asset_sha256=hashes,same_native_runtime=True,same_solver_config=True,same_applied_force=not(wind_comparison or pin_comparison),
        pinned_vertex_counts=[m['pins'] for m in metas],
        max_left_edge_interior_displacement_m=[float(np.linalg.norm(x[:,left & ~ends]-rest[left & ~ends],axis=2).max()) for x in xs],
        wind_scales=[m.get('wind_scale',1.) for m in metas],same_initial_positions=bool(np.array_equal(xs[0][0],xs[1][0])),
        final_free_edge_mean_position_m=[x[-1,right].mean(axis=0).tolist() for x in xs],
        mean_free_edge_height_last_2s_m=[float(x[-51:,right,2].mean()) for x in xs],
        mean_free_edge_x_last_2s_m=[float(x[-51:,right,0].mean()) for x in xs],
        initial_amplitude_m=[a['initial_max_abs_y_m'] if 'initial_max_abs_y_m' in a else 0. for a in audits],
        peak_abs_y_m=[float(y.max()) for y in ys],final_abs_y_m=[float(y[-1]) for y in ys],
        peak_time_s=[float(t[y.argmax()]) for y in ys],
        final_position_difference_rms_m=float(np.sqrt(np.mean(np.sum((xs[0][-1]-xs[1][-1])**2,axis=1)))),
        interpretation='Controlled L0 comparison; not a robustness or aerodynamic-flutter validation.')
    (out/'comparison.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#f3f6fa')
    fig.text(.035,.947,'FLAG L0 / '+('TWO ATTACHMENT POINTS' if pin_comparison else 'STRONGER WIND' if wind_comparison else 'INITIAL PERTURBATION'),fontsize=21,weight='bold',color='#163347')
    subtitle=(f'Same 1 mm seed, mesh and material | +x force scaled by {ratio:g} | displacement scale 1:1' if wind_comparison
        else 'Same mesh, material, +x wind and time step | displacement scale 1:1')
    if pin_comparison:subtitle='Same mesh, 1 mm seed and 1.5x wind | Red marks = fixed vertices | displacement scale 1:1'
    fig.text(.035,.9,subtitle,fontsize=12,color='#416476')
    timer=fig.text(.81,.947,'',fontsize=15,color='#167e99',weight='bold')
    polys=[];labels=[];colors=[np.array([.19,.65,.76]),np.array([.91,.59,.28])]
    titles=([f'{label} / {3*m.get("wind_scale",1.):g}-{13*m.get("wind_scale",1.):g} m/s²'
        for label,m in zip(['PREVIOUS','STRONGER'],metas)] if wind_comparison else ['FLAT START / 0 mm','PERTURBED START / 1 mm'])
    if pin_comparison:titles=['WHOLE EDGE / 140 fixed vertices','TWO ENDPOINTS / 2 fixed vertices']
    for i,title in enumerate(titles):
        a=fig.add_axes([.01+i*.5,.21,.48,.64],projection='3d',facecolor='#f3f6fa')
        a.set_proj_type('ortho');a.view_init(18,-66)
        a.set_xlim(-.12,1.72);a.set_ylim(-.85,.85);a.set_zlim(-.75,1.5)
        a.set_box_aspect((1.84,1.7,2.25),zoom=1.1);a.set_axis_off()
        a.plot([0,0],[0,0],[-.65,1.42],color='#4a6072',lw=3)
        if pin_comparison:
            if i==0:a.plot([0,0],[0,0],[.25,1.25],color='#d04e54',lw=4)
            else:a.scatter(rest[ends,0],rest[ends,1],rest[ends,2],color='#d04e54',s=45,depthshade=False)
        elif metas[i]['pins']==2:
            a.scatter(rest[ends,0],rest[ends,1],rest[ends,2],color='#d04e54',s=45,depthshade=False)
        a.plot([0,1.5,1.5,0,0],[0]*5,[.25,.25,1.25,1.25,.25],color='#9aaabb',ls='--',lw=.8,alpha=.65)
        a.quiver(.2,0,1.4,1,0,0,length=.8,arrow_length_ratio=.16,color='#c08330',linewidth=2)
        poly=Poly3DCollection(xs[i][0][f],linewidths=.18,edgecolors=(.04,.2,.29,.35));a.add_collection3d(poly);polys.append(poly)
        fig.text(.08+i*.5,.85,title,fontsize=12,weight='bold',color='#416476')
        labels.append(fig.text(.08+i*.5,.235,'',fontsize=11,color='#416476'))
    chart=fig.add_axes([.1,.085,.82,.105],facecolor='#f3f6fa');lines=[]
    for i,y in enumerate(chart_values):
        chart.plot(t,y,color=colors[i],alpha=.25,lw=1)
        line,=chart.plot([],[],color=colors[i],lw=2);lines.append(line)
    chart.set_xlim(0,8);chart.set_ylim(min(0,min(y.min() for y in chart_values)),max(y.max() for y in chart_values)*1.1)
    chart.set_ylabel('Free-edge z (m)' if args.metric=='free_edge_height' else 'max |y| (m)');chart.set_xlabel('Time (s)');chart.grid(alpha=.15)
    chart.spines[['top','right']].set_visible(False)
    fig.text(.035,.015,'911 dynamic vertices | 200 converged steps each | Initial velocity = 0 | Prescribed body load, not aerodynamics',fontsize=10,color='#647788')
    gifs=[]
    with imageio.get_writer(out/'comparison.mp4',fps=25,codec='libx264',macro_block_size=1,quality=8) as writer:
        for k in range(201):
            for i,poly in enumerate(polys):
                tri=xs[i][k][f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
                poly.set_verts(tri);poly.set_facecolor(colors[i]*(.72+.28*np.abs(n@np.array([.3,-.7,.64])))[:,None])
                lines[i].set_data(t[:k+1],chart_values[i][:k+1])
                labels[i].set_text(f'Free-edge mean: x = {free_edges[i][k,0]:.3f} m | z = {free_edges[i][k,2]:.3f} m'
                    if args.metric=='free_edge_height' else f'Max out-of-plane displacement: {ys[i][k]:.3f} m')
            timer.set_text(f'{t[k]:.2f} / 8.00 s');fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
            writer.append_data(rgb)
            if k%2==0:gifs.append(Image.fromarray(rgb).resize((960,540)))
            if k in [0,40,200]:Image.fromarray(rgb).save(out/f'comparison_{k:03}.png')
    gifs[0].save(out/'comparison.gif',save_all=True,append_images=gifs[1:],duration=80,loop=0)
    plt.close(fig)
    reader=imageio.get_reader(out/'comparison.mp4');assert reader.count_frames()==201;reader.get_data(200);reader.close()
    with Image.open(out/'comparison.gif') as im:assert im.n_frames==101
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
