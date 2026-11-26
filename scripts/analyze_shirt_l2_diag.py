"""Audit restored diag baseline, native preconditioner blocks, and bounded probes."""
import ctypes,hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mpmath as mp
from scipy.io import mmread
from scipy.sparse import coo_matrix
from progressive.prolongation import frames
from shirt_demo.report import plt,load_run,draw
from trimesh.triangles import closest_point

OUT=Path('output/shirt_L2_diag_20261119')
RUN=Path('output/shirt_L2_feasibility_fix_20261118/L2_guard_v4_diag/run')
meta=json.loads((RUN/'run.json').read_text());x=np.load(RUN/'positions.npy',mmap_mode='r');f=np.load(RUN/'faces.npy');poses=np.load(RUN/'robot.npy',mmap_mode='r')
steps=json.loads((RUN/'steps.json').read_text());timing=np.load(RUN/'timing.npz')['step_seconds']
rest=x[0][f];IB=frames(x[0],f)[2]
stages=[60,1076,1616,1800,1900,1920,1940,1960,1980,2000,2028,2050,2074]
history=[];focus_faces={}
for k in stages:
    tri=x[k][f];F=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=-1)@IB
    sv=np.linalg.svd(F,compute_uv=False)[:,0];gap=1.1-sv;ids=np.argsort(gap)[:8];focus_faces[k]=ids
    rec=dict(step=k,max_stretch=float(sv.max()),cpu_min_gap=float(gap.min()),near_1e8_count=int(sum(gap<1e-8)),near_1e12_count=int(sum(gap<1e-12)),
        worst_faces=ids.tolist(),max_robot_origin_error_mm=float(max(steps[k-1]['body_origin_tracking_errors_m'])*1000),
        seconds=float(timing[k-1]),linear_iterations=steps[k-1]['native']['linear_solver_iterations'],newton_iterations=steps[k-1]['native']['newton_iterations'])
    history.append(rec)

frame=0;capframes={};linear_by_frame={}
for line in (RUN.parent/'run.log').open(encoding='utf-8',errors='replace'):
    m=re.search(r'>>> Begin Frame: (\d+)',line)
    if m:frame=int(m[1])
    m=re.search(r'Iterative linear solver iteration count: (\d+)',line)
    if m:
        it=int(m[1]);linear_by_frame.setdefault(frame,[]).append(it)
        if it>=51690:capframes[frame]=capframes.get(frame,0)+1

def matrix(path):
    raw=mmread(path).tocoo();mask=raw.row//3!=raw.col//3
    assert np.all(raw.row[mask]//3<raw.col[mask]//3)
    return (raw+coo_matrix((raw.data[mask],(raw.col[mask],raw.row[mask])),shape=raw.shape)).tocsr()
def vector(p):return np.asarray(mmread(p)).ravel()

probes=[]
for k in [1900,2028,2074]:
    root=OUT/f'probe_{k}';info=json.loads((root/'probe.json').read_text());rows=[]
    for p in sorted(root.rglob('A.*.mtx')):
        A=matrix(p);b=vector(p.with_name(p.name.replace('A.','b.')));z=vector(p.with_name(p.name.replace('A.','x.')));r=A@z-b
        rows.append(dict(iteration=int(p.stem.split('.')[-1]),matrix_diag_max=float(A.diagonal().max()),matrix_diag_min=float(A.diagonal().min()),
            relative_true_residual=float(np.linalg.norm(r)/np.linalg.norm(b)),abd_relative_true_residual=float(np.linalg.norm(r[:108])/np.linalg.norm(b[:108])),
            max_direction_component=float(abs(z).max())))
    probes.append(dict(saved_step=k,stats=info['stats'],seconds=info['wall_seconds'],linear_solves=rows,
        cloth_max_motion_mm=info['cloth_max_displacement_mm'],cap_samples=[json.loads(s) for s in (root/'cap_audit.jsonl').read_text().splitlines()]))

dll=ctypes.CDLL(str((OUT/'diag_block_replay.dll').resolve()));ptr=ctypes.POINTER(ctypes.c_double)
dll.evaluate.argtypes=[ctypes.c_int,ptr,ptr];dll.evaluate.restype=ctypes.c_int
block_checks=[];mp.mp.dps=80
for k in [1900,2074]:
    root=OUT/f'native_pz_{k}';p=next(root.rglob('A.1.0.mtx'));A=matrix(p)
    rp=next(root.rglob('r.1.0.0.mtx'));r=vector(rp);native_z=vector(rp.with_name('z.1.0.0.mtx'))
    fem=A[108:,108:].tobsr(blocksize=(3,3))
    D=np.array([fem.data[fem.indptr[i]:fem.indptr[i+1]][fem.indices[fem.indptr[i]:fem.indptr[i+1]]==i][0] for i in range(len(x[0]))])
    b=r[108:].reshape(-1,3);z=native_z[108:].reshape(-1,3)
    data=np.ascontiguousarray(np.c_[D.reshape(-1,9),b]);out=np.zeros((len(b),7))
    err=dll.evaluate(len(data),data.ctypes.data_as(ptr),out.ctypes.data_as(ptr));assert err==0
    z_stable=out[:,3:6];eig=np.linalg.eigvalsh((D+D.transpose(0,2,1))*.5)
    residual=lambda zz:np.linalg.norm(np.einsum('nij,nj->ni',D,zz)-b,axis=1)/np.maximum(np.linalg.norm(b,axis=1),1e-30)
    ng=residual(z);sg=residual(z_stable);replayerr=residual(out[:,:3]);samples=[]
    for i in np.argsort(ng)[-5:][::-1]:
        M=mp.matrix(D[i].tolist());rhs=mp.matrix(b[i].tolist());exact=mp.lu_solve(M,rhs)
        def mr(zz):return float(mp.norm(M*mp.matrix(zz.tolist())-rhs)/mp.norm(rhs))
        numpy_z=np.linalg.solve(D[i],b[i])
        samples.append(dict(vertex=int(i),eigenvalues=eig[i].tolist(),native_relative_residual_80digit=mr(z[i]),
            scaled_cholesky_gpu_relative_residual_80digit=mr(z_stable[i]),numpy_pivoted_solve_relative_residual_80digit=mr(numpy_z),
            native_rz=float(b[i]@z[i]),scaled_cholesky_gpu_rz=float(b[i]@z_stable[i]),reference_rz=float((rhs.T*exact)[0])))
    np.savez_compressed(OUT/f'blocks_{k}.npz',D=D,r=b,native_z=z,replay_analytic_z=out[:,:3],scaled_cholesky_z=z_stable)
    block_checks.append(dict(saved_step=k,blocks=len(D),native_total_rz=float(r@native_z),native_fem_negative_rz_blocks=int(sum((b*z).sum(1)<0)),
        scaled_cholesky_negative_rz_blocks=int(sum((b*z_stable).sum(1)<0)),scaled_cholesky_failed_pivots=int(sum(out[:,6]!=1)),
        native_max_relative_residual=float(ng.max()),native_replay_max_relative_residual=float(replayerr.max()),
        scaled_cholesky_max_relative_residual=float(sg.max()),worst_samples=samples,
        max_block_eigenvalue=float(eig[:,-1].max()),max_block_eigenvalue_ratio=float(np.max(eig[:,-1]/eig[:,0]))))

# Existing independently compiled v4 strain kernel: finite domain and stiffness replay.
sys.path.insert(0,str(Path('scripts').resolve()))
from check_psq_saved_feasibility import evaluate
strain_dll_path=Path('output/shirt_L2_feasibility_fix_20261118/probe/coordinate_check.dll')
strain_dll=ctypes.CDLL(str(strain_dll_path.resolve()));gpu=[]
areas=np.linalg.norm(np.cross(rest[:,1]-rest[:,0],rest[:,2]-rest[:,0]),axis=1)*.5
for k in [1900,1940,2028,2074]:
    rows=np.c_[rest.reshape(-1,9),x[k][f].reshape(-1,9)];out=evaluate(strain_dll,rows)
    weight=areas*.0002*meta['dt']**2;norm=out[:,13]*weight;i=int(norm.argmax())
    gpu.append(dict(step=k,nonfinite_energy=int(sum(~np.isfinite(out[:,0]))),max_stretch=float(out[:,2].max()),
        gpu_min_gap=float(1.1-out[:,2].max()),largest_weighted_element_hessian_norm=float(norm[i]),worst_face=i,
        finite_gradient_hessian=bool(np.isfinite(out[:,12:14]).all())))

# Locate the most stretched saved triangles against actual unsimplified robot surfaces.
colliders=[dict(np.load(RUN/f'collider_{j}.npz')) for j in range(8)];locations=[]
for k in [1940,2074]:
    for face in focus_faces[k][:5]:
        tri=x[k,f[face]];best=(float('inf'),None)
        for j,b in enumerate(colliders):
            v=b['vertices']@poses[k,j,:3,:3].T+poses[k,j,:3,3];tris=v[b['faces']]
            for point in tri:
                near=closest_point(tris,np.broadcast_to(point,(len(tris),3)));d=float(np.linalg.norm(near-point,axis=1).min())
                if d<best[0]:best=(d,j)
        locations.append(dict(step=k,face=int(face),nearest_robot_body=best[1],minimum_triangle_vertex_to_robot_surface_mm=best[0]*1000,
            centroid=tri.mean(0).tolist()))

report=dict(status='diag_restored_cause_diagnosed_not_full_fold_fixed',history=history,original_pcg_limit_counts=capframes,
    bounded_diag_probes=probes,diagonal_block_checks=block_checks,strain_gpu_replay=gpu,critical_face_locations=locations,
    limitations=['Snapshot probes rebuild native adaptive/contact/friction history and are not exact checkpoint replays.',
        'Eight-Newton diagnostic cap is deliberate; not a claim that the original 1024-Newton limit was hit.',
        'Scaled Cholesky is an isolated GPU block replay only, not installed in the simulation runtime; full Newton convergence and folding are unverified.',
        'CPU/GPU singular values may differ by rounding near the limit. Independent GPU replay uses production formulas but compilation can differ.',
        'Minimum vertex-to-surface distance localizes faces; it is not a native contact force measurement.'])
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2))

fig,ax=plt.subplots(3,1,figsize=(10,9),layout='constrained')
ks=np.arange(1,len(steps)+1)
ax[0].semilogy(ks,[r['native']['linear_solver_iterations'] for r in steps],lw=.7)
ax[0].set(ylabel='PCG iterations per physical step',xlim=(1750,2076))
ax[1].semilogy([r['step'] for r in history],[max(r['cpu_min_gap'],1e-16) for r in history],'.-')
ax[1].set(ylabel='Stretch gap: 1.1 - sigma_max',xlim=(1750,2076))
ax[2].plot(ks,[max(r['body_origin_tracking_errors_m'])*1000 for r in steps],lw=.8)
ax[2].set(ylabel='Max robot origin target error (mm)',xlabel='Physical step',xlim=(1750,2076))
for aa in ax:aa.grid(alpha=.2);aa.axvline(1940,color='gray',ls='--',alpha=.5)
fig.suptitle('Original L2 diag run: strain boundary, solver cost, and tracking')
fig.savefig(OUT/'diag_deterioration.png',dpi=150);plt.close(fig)
data=load_run(RUN);fig=plt.figure(figsize=(11,5))
for ii,k in enumerate([1940,2074]):
    aa=fig.add_subplot(1,2,ii+1,projection='3d');draw(aa,data,k,f'L2 diag step {k}: red = most stretched faces')
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    aa.add_collection3d(Poly3DCollection(x[k][f[focus_faces[k]]],facecolors='red',edgecolors='red'))
fig.tight_layout();fig.savefig(OUT/'critical_faces.png',dpi=150);plt.close(fig)
print(json.dumps(dict(block_checks=block_checks,gpu=gpu,probe_initial_matrix_max=[(r['saved_step'],r['linear_solves'][0]['matrix_diag_max']) for r in probes]),indent=2))
