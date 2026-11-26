"""Audit and render the actual L0 trajectory; no interpolated or staged motion."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
from PIL import Image
from scipy.sparse import csr_matrix
import ipctk

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from progressive.prolongation import ShellProlongation,frames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT/'output/flag_L0_parallel_wind_20261121')
    parser.add_argument('--run-name',default='L0');parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    run=root/args.run_name;out=root/'report' if args.run_name=='L0' else root/'report'/args.run_name;out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((run/'run.json').read_text());steps=json.loads((run/'steps.json').read_text())
    scene=json.loads((root/'scene.json').read_text())
    local=meta.get('elastic_model')=='native_current_level'
    x=np.load(run/'positions.npy');d=np.load(root/('operator_local/operator.npz' if local else 'operator/operator.npz'))
    rest,f=d['coarse_rest'],d['coarse_faces'];fixed=np.isclose(rest[:,0],0,atol=1e-12)
    left=fixed.copy();corners=meta.get('pin_mode','edge')=='corners'
    if corners:fixed &= np.isclose(rest[:,2],.25,atol=1e-12)|np.isclose(rest[:,2],1.25,atol=1e-12)
    assert fixed.sum()==meta['pins']==(2 if corners else 140)
    if 'pin_indices' in meta:np.testing.assert_array_equal(np.flatnonzero(fixed),meta['pin_indices'])
    assert meta['status']=='completed' and len(steps)==200 and len(x)==201 and np.isfinite(x).all()
    assert all(s['native']['converged'] and not s['native']['preview_non_descent'] and not s['native']['preview_small_step'] for s in steps)
    assert all(s['wind']['max_off_axis_force_N']==0 for s in steps)
    wind_scale=meta.get('wind_scale',1.)
    np.testing.assert_allclose([s['wind']['acceleration_x_m_s2'] for s in steps],wind_scale*(8+5*np.sin(2*np.pi*1.3*np.arange(1,201)*.04)),rtol=0,atol=1e-12)
    U=csr_matrix((d['U_data'],d['U_indices'],d['U_indptr']),shape=tuple(d['U_shape']))
    p=ShellProlongation(rest,d['fine_rest'],f,U,d['anchors'])
    finefixed=np.isclose(d['fine_rest'][:,0],0,atol=1e-12)
    if corners:finefixed &= np.isclose(d['fine_rest'][:,2],.25,atol=1e-12)|np.isclose(d['fine_rest'][:,2],1.25,atol=1e-12)
    assert finefixed.sum()==fixed.sum()
    pole=(float(np.abs(x[:,fixed]-rest[fixed]).max()) if local else max(float(np.abs(p.position(v)[finefixed]-d['fine_rest'][finefixed]).max()) for v in x))
    edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(rest,edges,f.astype(np.int32))
    intersections=[k for k,v in enumerate(x) if ipctk.has_intersections(cm,v)]
    assert not intersections and pole<1e-12 and np.abs(x[:,fixed]-rest[fixed]).max()<1e-12
    systems=json.loads((run/'backend/systems.json').read_text())['sim_systems']
    names=[s['name'].split('::')[-1] for s in systems if s['valid']]
    assert 'IPCSimplexNormalContact' in names and 'IPCSimplexFrictionalContact' in names
    assert 'SoftPositionConstraint' not in names
    t=np.arange(len(x))*.04; wind=wind_scale*(8+5*np.sin(2*np.pi*1.3*t))
    y=np.abs(x[:,:,1]).max(axis=1);right=np.isclose(rest[:,0],1.5,atol=1e-12)
    seed=meta['initial_out_of_plane_seed_m']
    expected=rest.copy();expected[:,1]+=seed*(rest[:,0]/1.5)**2*np.sin(2*np.pi*(rest[:,2]-.25))
    np.testing.assert_allclose(x[0],expected,rtol=0,atol=1e-12)
    tri=x[0][f];F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)@frames(rest,f)[2]
    initial_stretch=float(np.linalg.svd(F,compute_uv=False).max())
    mass=np.zeros(len(rest));area=frames(rest,f)[3]
    np.add.at(mass,f.ravel(),np.repeat(area*.0004*200/3,3))
    # Reconstruct each vertex force before reducing, as in the animator. Summing
    # a strided (N,3) array and multiplying a pairwise mass sum have different
    # roundoff at fine resolutions; retain the strict check on like reductions.
    recorded_force=np.array([s['wind']['total_force_N'] for s in steps])
    reconstructed_force=[]
    for step in steps:
        force=np.zeros_like(rest)
        force[:,0]=mass*step['wind']['acceleration_x_m_s2']
        force[fixed]=0.
        reconstructed_force.append(force.sum(axis=0))
    np.testing.assert_allclose(recorded_force,reconstructed_force,rtol=0,atol=1e-12)
    analytic_force=np.column_stack([wind[1:]*mass[~fixed].sum(),np.zeros((200,2))])
    # Standard gamma_n bound for summation, plus multiplication and the
    # independently checked acceleration's absolute tolerance.
    eps=np.finfo(mass.dtype).eps;n=len(mass)+2;gamma_n=n*eps/(1-n*eps)
    aggregate_bound=2*gamma_n*np.abs(analytic_force)+mass.sum()*1e-12
    assert np.all(np.abs(recorded_force-analytic_force)<=aggregate_bound)
    tip=x[:,right].mean(axis=1);stretch=np.array([initial_stretch]+[s['stretch'] for s in steps])
    audit=dict(completed_frames=200,duration_s=8.,all_native_steps_converged=True,
        level=meta['level'],method=meta.get('method','direct'),
        total_mass_kg=float(mass.sum()),body_force_mass_weighting_verified=True,no_soft_position_constraint=True,
        force_reconstruction_max_error_N=float(np.abs(recorded_force-reconstructed_force).max()),
        force_aggregate_roundoff_max_N=float(np.abs(recorded_force-analytic_force).max()),
        wind_scale=wind_scale,
        pin_mode=meta.get('pin_mode','edge'),fixed_vertices=int(fixed.sum()),
        max_left_edge_interior_displacement_m=float(np.linalg.norm(x[:,left & ~fixed]-rest[left & ~fixed],axis=2).max()) if corners else 0.,
        dynamic_vertices=len(rest),dynamic_faces=len(f),elastic_proxy_vertices=0 if local else len(d['fine_rest']),
        elastic_model=meta.get('elastic_model','PSQ_original_fine_proxy'),
        max_coarse_pin_error_m=float(np.abs(x[:,fixed]-rest[fixed]).max()),max_psq_fixed_vertex_error_m=pole,
        saved_states_with_coarse_intersections=intersections,max_principal_stretch=float(stretch.max()),
        max_out_of_plane_m=float(y.max()),first_step_out_of_plane_m=float(y[1]),
        imposed_out_of_plane_force_N=0.,initial_out_of_plane_seed_m=seed,
        initial_max_abs_y_m=float(y[0]),initial_state_formula_verified=True,
        note='Prescribed initial displacement only; material rest is flat. This body-force model does not simulate aerodynamic flutter.',
        runtime_contact_kappa_note='Requested 1e9; native mesh-dependent clamp and adaptation apply. See native initialization log.',
        active_native_systems=names)
    if meta['level'] and meta['method']!='direct':
        import hashlib
        from scipy.sparse import load_npz
        lower=Path(meta['lower_run']);assert hashlib.sha256((lower/'positions.npy').read_bytes()).hexdigest()==meta['lower_positions_sha256']
        level=meta['level']
        low=np.load(lower/'positions.npy');U01=load_npz(root/f'assets/map_{level-1}_{level}.npz')
        lowrest=np.load(root/f'assets/mesh_{level-1}.npz')['vertices']
        np.testing.assert_allclose(U01@lowrest,rest,rtol=0,atol=1e-12)
        injected=np.load(run/'injected_velocities.npy');assert injected.shape==(200,len(rest),3)
        error=0.
        for k in range(200):
            displacement=np.zeros_like(low[0]) if k==0 else low[k]-low[k-1]
            target=(U01@low[k] if meta['method']=='semi' else x[k])+U01@displacement
            expected=(target-x[k])/.04;expected[fixed]=0.
            error=max(error,float(np.abs(expected-injected[k]).max()))
        assert error<1e-10
        audit.update(injected_predictor_max_error_m_s=error,parent_trajectory_sha256_verified=True)
    elif meta['level']:
        assert not meta['coarse_trajectory_used'] and not meta['velocity_injection_each_step'] and 'lower_run' not in meta
        assert not (run/'injected_velocities.npy').exists()
        velocities=np.load(run/'pre_step_velocities.npy')
        expected=np.concatenate([np.zeros_like(x[:1]),np.diff(x[:-1],axis=0)/.04])
        np.testing.assert_allclose(velocities,expected,rtol=0,atol=1e-10)
        audit.update(independent_time_integration=True,own_velocity_history_max_error_m_s=float(np.abs(velocities-expected).max()))
        if local:
            bundle=json.loads((root/'local_bundle.json').read_text());assert bundle['identity']
            assert meta['native_psq_mode']=='reference' and bundle['counts']['proxy_vertices']==len(rest)
            logpath=root/f'{args.run_name}.log'
            if not logpath.exists():logpath=ROOT/'output/flag_L1_direct_local_simulation.log'
            log=logpath.read_text()
            assert 'replace=false strain_limit=1.1' in log and f'coarse={len(rest)} proxy={len(rest)}' in log
            audit.update(native_local_elastic_verified=True,psq_energy_replacement=False)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    np.savetxt(out/'metrics.csv',np.column_stack([t,wind,y,tip,stretch]),delimiter=',',
        header='time_s,wind_acceleration_x_m_s2,max_abs_y_m,free_edge_mean_x_m,free_edge_mean_y_m,free_edge_mean_z_m,max_principal_stretch',comments='')
    if args.audit_only:
        print(json.dumps({k:v for k,v in audit.items() if k!='active_native_systems'},indent=2));return
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#f3f6fa')
    fig.text(.035,.947,f'FLAG / L{meta["level"]}',fontsize=23,weight='bold',color='#163347')
    fig.text(.035,.903,f'Wind along +x: {wind_scale:g}x baseline | Initial seed: {seed*1000:g} mm',fontsize=13,color='#416476')
    timer=fig.text(.76,.943,'',fontsize=18,weight='bold',color='#167e99')
    ax=fig.add_axes([.01,.12,.65,.75],projection='3d',facecolor='#f3f6fa')
    ax.set_proj_type('ortho');ax.view_init(elev=18,azim=-66)
    ax.set_xlim(-.12,1.72);ax.set_ylim(-.85,.85);ax.set_zlim(-.75,1.5)
    ax.set_box_aspect((1.84,1.7,2.25),zoom=1.1);ax.set_axis_off()
    ax.plot([0,0],[0,0],[-.65,1.42],color='#4a6072',lw=4)
    ax.plot([0,1.5,1.5,0,0],[0]*5,[.25,.25,1.25,1.25,.25],color='#9aaabb',ls='--',lw=.8,alpha=.65)
    ax.quiver(.2,0,1.4,1,0,0,length=.8,arrow_length_ratio=.16,color='#e39330',linewidth=3)
    ax.text(.45,0,1.52,'wind +x',color='#a96717',fontsize=11)
    colors=np.tile([.19,.65,.76],(len(f),1))
    poly=Poly3DCollection(x[0][f],linewidths=.22,edgecolors=(.04,.2,.29,.38))
    ax.add_collection3d(poly)
    if corners:ax.scatter(rest[fixed,0],rest[fixed,1],rest[fixed,2],color='#d04e54',s=35,depthshade=False)
    else:ax.plot([0,0],[0,0],[.25,1.25],color='#d04e54',lw=3)
    front=fig.add_axes([.70,.51,.26,.31],facecolor='#f3f6fa',projection='3d')
    front.set_proj_type('ortho');front.view_init(0,-90)
    front.set_xlim(-.12,1.72);front.set_ylim(-.85,.85);front.set_zlim(-.75,1.5)
    front.set_box_aspect((1.84,1.7,2.25),zoom=1.4);front.set_axis_off()
    front.plot([0,0],[0,0],[-.65,1.42],color='#4a6072',lw=3)
    poly2=Poly3DCollection(x[0][f],linewidths=.12,edgecolors=(.04,.2,.29,.3));front.add_collection3d(poly2)
    if corners:front.scatter(rest[fixed,0],rest[fixed,1],rest[fixed,2],color='#d04e54',s=25,depthshade=False)
    fig.text(.71,.85,'FRONT VIEW  /  x-z',fontsize=10,weight='bold',color='#416476')
    chart=fig.add_axes([.71,.295,.255,.125],facecolor='#f3f6fa')
    chart.plot(t,wind,color='#d69a45',alpha=.4,lw=1);line,=chart.plot([],[],color='#c67c18',lw=2)
    chart.set_xlim(0,8);chart.set_ylim(0,15*wind_scale);chart.set_ylabel('a_x (m/s²)',fontsize=9);chart.set_xticks([0,2,4,6,8]);chart.grid(alpha=.15)
    chart2=fig.add_axes([.71,.105,.255,.125],facecolor='#f3f6fa')
    chart2.plot(t,y,color='#43a4b8',alpha=.35,lw=1);line2,=chart2.plot([],[],color='#167e99',lw=2)
    chart2.set_xlim(0,8);chart2.set_ylim(0,max(y)*1.12);chart2.set_ylabel('max |y| (m)',fontsize=9);chart2.set_xlabel('Time (s)',fontsize=9);chart2.grid(alpha=.15)
    label=fig.text(.035,.125,'',fontsize=11,color='#274b60')
    fig.text(.035,.077,f'{len(rest):,} L{meta["level"]} vertices  /  {len(f):,} triangles  /  {fixed.sum()} fixed vertices (red)',fontsize=10,color='#416476')
    fig.text(.035,.035,'Prescribed body load + gravity | dt = 0.04 s | Actual mesh, displacement scale 1:1',fontsize=10,color='#647788')
    gifs=[]
    def draw(k):
        tri=x[k][f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
        shade=.72+.28*np.abs(n@np.array([.3,-.7,.64]))
        for item in [poly,poly2]:item.set_verts(tri);item.set_facecolor(colors*shade[:,None])
        timer.set_text(f'{t[k]:05.2f} / 8.00 s')
        label.set_text(f'Wind: {wind[k]:.2f} m/s²     Max stretch: {stretch[k]:.4f}')
        line.set_data(t[:k+1],wind[:k+1]);line2.set_data(t[:k+1],y[:k+1]);fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
    with imageio.get_writer(out/'flag_L0_parallel.mp4',fps=25,codec='libx264',macro_block_size=1,quality=8) as writer:
        for k in range(len(x)):
            rgb=draw(k);writer.append_data(rgb)
            if k%2==0:gifs.append(Image.fromarray(rgb).resize((960,540)))
            if k in [0,40,100,200]:Image.fromarray(rgb).save(out/f'frame_{k:03}.png')
    gifs[0].save(out/'flag_L0_parallel.gif',save_all=True,append_images=gifs[1:],duration=80,loop=0)
    plt.close(fig)
    reader=imageio.get_reader(out/'flag_L0_parallel.mp4');assert reader.count_frames()==201;reader.get_data(200);reader.close()
    with Image.open(out/'flag_L0_parallel.gif') as gif:assert gif.n_frames==101
    print(json.dumps({k:v for k,v in audit.items() if k!='active_native_systems'},indent=2),flush=True)


if __name__=='__main__':main()
