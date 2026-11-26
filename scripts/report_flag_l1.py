"""Compare the accepted L0 trajectory with Semi-diagonal and VelPro L1."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--direct',action='store_true');parser.add_argument('--local-direct',action='store_true');args=parser.parse_args();include_direct=args.direct or args.local_direct
    base=ROOT/'output/flag_L0_two_pins_wind300_20261122'
    root=ROOT/'output/flag_L1_two_pins_wind300_20261122';report=root/'report';out=report/'with_direct' if include_direct else report;out.mkdir(parents=True,exist_ok=True)
    if args.local_direct:out=report/'with_local_direct';out.mkdir(exist_ok=True)
    paths=[base/'L0',root/'L1_semi',root/'L1_velpro']
    if include_direct:paths.append(root/('L1_direct_local' if args.local_direct else 'L1_direct'))
    meta=[json.loads((p/'run.json').read_text()) for p in paths]
    audit=[json.loads((base/'report/audit.json').read_text())]+[json.loads((report/p.name/'audit.json').read_text()) for p in paths[1:]]
    assert all(m['status']=='completed' and m['accepted_frames']==200 and m['pins']==2 and m['wind_scale']==3 for m in meta)
    assert all(m['config']==meta[0]['config'] and m['runtime_sha256']==meta[0]['runtime_sha256'] for m in meta)
    for name in ['mesh_0.npz','mesh_1.npz','mesh_9.npz','map_0_1.npz']:
        assert (base/'assets'/name).read_bytes()==(root/'assets'/name).read_bytes()
    x=[np.load(p/'positions.npy') for p in paths]
    meshes=[np.load(root/f'assets/mesh_{i}.npz') for i in [0]+[1]*(len(paths)-1)]
    for xx,d in zip(x,meshes):assert xx.shape==(201,len(d['vertices']),3)
    wind=[json.loads((p/'applied_wind.json').read_text()) for p in paths]
    np.testing.assert_array_equal([r['acceleration_x_m_s2'] for r in wind[0]],[r['acceleration_x_m_s2'] for r in wind[1]])
    assert wind[1]==wind[2]
    if include_direct:
        assert wind[1]==wind[3] and audit[3]['independent_time_integration']
        np.testing.assert_array_equal(x[1][0],x[3][0]);np.testing.assert_array_equal(x[2][0],x[3][0])
        if args.local_direct:assert audit[3]['native_local_elastic_verified'] and audit[3]['elastic_proxy_vertices']==0
    U=load_npz(root/'assets/map_0_1.npz');lift=np.array([U@v for v in x[0]])
    rms=[np.sqrt(np.mean(np.sum((xx-lift)**2,axis=2),axis=1)) for xx in x[1:]]
    t=np.arange(201)*.04;names=['L0 / Reference','L1 / Semi-diagonal','L1 / VelPro']
    if include_direct:names.append('L1 / Direct')
    display_rms=([np.sqrt(np.mean(np.sum((xx-x[3])**2,axis=2),axis=1)) for xx in x[1:3]] if include_direct else rms)
    edges=[xx[:,np.isclose(d['vertices'][:,0],1.5,atol=1e-12)].mean(axis=1) for xx,d in zip(x,meshes)]
    metrics=dict(names=names,vertices=[len(d['vertices']) for d in meshes],faces=[len(d['faces']) for d in meshes],
        max_rms_difference_from_L0_prolongation_m=[float(a.max()) for a in rms],
        mean_rms_difference_from_L0_prolongation_m=[float(a[1:].mean()) for a in rms],
        final_rms_difference_from_L0_prolongation_m=[float(a[-1]) for a in rms],
        mean_free_edge_height_last_2s_m=[float(a[-51:,2].mean()) for a in edges],
        peak_out_of_plane_m=[float(np.abs(xx[:,:,1]).max()) for xx in x],
        max_principal_stretch=[a['max_principal_stretch'] for a in audit],
        parent_sha256=hashlib.sha256((base/'L0/positions.npy').read_bytes()).hexdigest(),
        note='Differences from L0 prolongation are not errors against a fine-level ground truth; all renders show actual dynamic meshes.')
    if include_direct:
        metrics.update(direct_reference='Independent L1 time integration with the same shared PSQ elastic proxy; not a converged fine-mesh ground truth',
            comparison_methods=['Semi-diagonal','VelPro'],
            mean_rms_difference_from_direct_m=[float(v[1:].mean()) for v in display_rms],
            max_rms_difference_from_direct_m=[float(v.max()) for v in display_rms],
            final_rms_difference_from_direct_m=[float(v[-1]) for v in display_rms])
        if args.local_direct:metrics['direct_reference']='Native L1 elasticity and independent L1 dynamics; no PD coupling or PSQ fine proxy.'
    (out/'comparison.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    methods=['semi','velpro']+(['direct'] if include_direct else [])
    header=['time_s']+[m+'_rms_vs_lift_m' for m in methods]+[m+'_free_z_m' for m in ['L0']+methods]
    columns=[t,*rms,*[e[:,2] for e in edges]]
    if include_direct:header+=['semi_rms_vs_direct_m','velpro_rms_vs_direct_m'];columns+=display_rms
    np.savetxt(out/'comparison_metrics.csv',np.column_stack(columns),delimiter=',',header=','.join(header),comments='')
    colors=[np.array([.19,.65,.76]),np.array([.91,.59,.28]),np.array([.48,.45,.76])]
    if include_direct:colors.append(np.array([.30,.67,.46]))
    cols=len(paths)
    fig=plt.figure(figsize=(20 if include_direct else 16,8),dpi=100,facecolor='#f3f6fa')
    fig.text(.035,.95,'FLAG / L0 TO L1',fontsize=24,weight='bold',color='#163347')
    fig.text(.035,.903,'Two fixed endpoints | Same +x wind, material and dt = 0.04 s | Actual dynamic meshes, scale 1:1',fontsize=13,color='#416476')
    timer=fig.text(.83,.948,'',fontsize=17,color='#167e99',weight='bold')
    polys=[];labels=[]
    for i,(name,d) in enumerate(zip(names,meshes)):
        a=fig.add_axes([.003+i/cols,.25,1/cols-.003,.60],projection='3d',facecolor='#f3f6fa')
        a.set_proj_type('ortho');a.view_init(18,-66)
        a.set_xlim(-.12,1.72);a.set_ylim(-.85,.85);a.set_zlim(-.75,1.5)
        a.set_box_aspect((1.84,1.7,2.25),zoom=1.15);a.set_axis_off()
        r=d['vertices'];pins=np.isclose(r[:,0],0,atol=1e-12)&(np.isclose(r[:,2],.25,atol=1e-12)|np.isclose(r[:,2],1.25,atol=1e-12))
        a.plot([0,0],[0,0],[-.65,1.42],color='#4a6072',lw=3)
        a.scatter(r[pins,0],r[pins,1],r[pins,2],color='#d04e54',s=35,depthshade=False)
        a.plot([0,1.5,1.5,0,0],[0]*5,[.25,.25,1.25,1.25,.25],color='#9aaabb',ls='--',lw=.7,alpha=.65)
        a.quiver(.2,0,1.4,1,0,0,length=.8,arrow_length_ratio=.16,color='#c08330',linewidth=2)
        poly=Poly3DCollection(x[i][0][d['faces']],linewidths=.16,edgecolors=(.04,.2,.29,.33));a.add_collection3d(poly);polys.append(poly)
        fig.text(.025+i/cols,.85,name,fontsize=14,weight='bold',color='#416476')
        fig.text(.025+i/cols,.818,f'{len(r):,} vertices / {len(d["faces"]):,} triangles',fontsize=11,color='#647788')
        labels.append(fig.text(.025+i/cols,.255,'',fontsize=11,color='#416476'))
    c1=fig.add_axes([.08,.09,.36,.12],facecolor='#f3f6fa');c2=fig.add_axes([.60,.09,.36,.12],facecolor='#f3f6fa')
    lines1=[];lines2=[]
    for i,v in enumerate(display_rms):
        c1.plot(t,v*1000,color=colors[i+1],alpha=.2,lw=1);line,=c1.plot([],[],color=colors[i+1],lw=2);lines1.append(line)
    for i,e in enumerate(edges):
        c2.plot(t,e[:,2],color=colors[i],alpha=.2,lw=1);line,=c2.plot([],[],color=colors[i],lw=2);lines2.append(line)
    c1.set_ylim(0,max(v.max() for v in display_rms)*1100);c1.set_ylabel('RMS vs Direct L1 (mm)' if include_direct else 'RMS vs lifted L0 (mm)',fontsize=10)
    c2.set_ylim(min(0,min(e[:,2].min() for e in edges)),.85);c2.set_ylabel('Free-edge height (m)',fontsize=10)
    for a in [c1,c2]:a.set_xlim(0,8);a.set_xlabel('Time (s)');a.grid(alpha=.15);a.spines[['top','right']].set_visible(False)
    footer=('Direct: native L1 elasticity, no PD / PSQ | Other runs: 35,000-vertex PSQ elastic proxy | 200 converged steps each' if args.local_direct
        else '200 converged steps per run | Shared 35,000-vertex elastic proxy | Prescribed body force, not aerodynamic flow')
    fig.text(.035,.018,footer,fontsize=11,color='#647788')
    gifs=[]
    with imageio.get_writer(out/'comparison.mp4',fps=25,codec='libx264',macro_block_size=1,quality=8) as writer:
        for k in range(201):
            for i,(poly,d) in enumerate(zip(polys,meshes)):
                tri=x[i][k][d['faces']];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
                poly.set_verts(tri);poly.set_facecolor(colors[i]*(.72+.28*np.abs(n@np.array([.3,-.7,.64])))[:,None])
                labels[i].set_text(f'Free-edge mean height: {edges[i][k,2]:.3f} m');lines2[i].set_data(t[:k+1],edges[i][:k+1,2])
            for i in range(2):lines1[i].set_data(t[:k+1],display_rms[i][:k+1]*1000)
            timer.set_text(f'{t[k]:.2f} / 8.00 s');fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();writer.append_data(rgb)
            if k%2==0:gifs.append(Image.fromarray(rgb).resize((1500,600) if include_direct else (1200,600)))
            if k in [0,40,100,200]:Image.fromarray(rgb).save(out/f'comparison_{k:03}.png')
    gifs[0].save(out/'comparison.gif',save_all=True,append_images=gifs[1:],duration=80,loop=0)
    plt.close(fig)
    reader=imageio.get_reader(out/'comparison.mp4');assert reader.count_frames()==201;reader.get_data(200);reader.close()
    with Image.open(out/'comparison.gif') as im:assert im.n_frames==101
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
