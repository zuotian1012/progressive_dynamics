"""Check exported native linear systems using true residuals and MAS block spectra."""
import json,re
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix
ROOT=Path('output/shirt_gripper_solver_diagnosis_20261119')

def matrix(path):
    raw=mmread(path).tocoo()
    # Native sym_spmv mirrors off-diagonal 3x3 BLOCKS, not scalar triangles.
    mask=raw.row//3!=raw.col//3
    assert np.all(raw.row[mask]//3<raw.col[mask]//3)
    A=(raw+coo_matrix((raw.data[mask],(raw.col[mask],raw.row[mask])),shape=raw.shape)).tocsr()
    return A

def vector(p):return np.asarray(mmread(p)).ravel()

reports=[]
for name in ['baseline_v2','no_graph','tight','diag','no_robot_contact','no_robot_contact_diag','mas_matrix_probe']:
    root=ROOT/name
    info=json.loads((root/'probe.json').read_text());rows=[]
    for p in sorted(root.rglob('A.*.mtx')):
        A=matrix(p);b=vector(p.with_name(p.name.replace('A.','b.')));x=vector(p.with_name(p.name.replace('A.','x.')))
        residual=A@x-b
        row=dict(matrix=p.name,b_norm=float(np.linalg.norm(b)),true_relative_residual=float(np.linalg.norm(residual)/np.linalg.norm(b)),
            abd_b_norm=float(np.linalg.norm(b[:108])),abd_true_relative_residual=float(np.linalg.norm(residual[:108])/np.linalg.norm(b[:108])),
            abd_direction_max=float(np.max(abs(x[:108]))),fem_direction_max=float(np.max(abs(x[108:]))),
            diagonal_min=float(A.diagonal().min()),diagonal_max=float(A.diagonal().max()))
        if name in ['no_robot_contact','no_robot_contact_diag','mas_matrix_probe']:
            assert A[:108,108:].nnz==0 or np.max(abs(A[:108,108:].data))==0
            exact=np.linalg.solve(A[:108,:108].toarray(),b[:108])
            row.update(decoupled_abd_reference_relative_residual=float(np.linalg.norm(A[:108,:108]@exact-b[:108])/np.linalg.norm(b[:108])),
                native_abd_direction_error_vs_dense_reference=float(np.linalg.norm(x[:108]-exact)/np.linalg.norm(exact)),
                reference_abd_direction_max=float(abs(exact).max()))
        rows.append(row)
    reports.append(dict(name=name,stats=info['stats'],wall_seconds=info['wall_seconds'],linear_solves=rows,
        robot_origin_displacement_mm=info['robot_origin_displacement_mm']))

def first(name):
    p=next((ROOT/name).rglob('A.1.0.mtx'))
    return matrix(p),vector(p.with_name('b.1.0.mtx'))
controls=[]
for a,b in [('baseline_v2','diag'),('baseline_v2','no_graph'),('baseline_v2','tight'),('no_robot_contact','no_robot_contact_diag')]:
    A,x=first(a);B,y=first(b)
    controls.append(dict(pair=[a,b],relative_matrix_frobenius_difference=float(np.linalg.norm((A-B).data)/np.linalg.norm(A.data)),
        relative_rhs_difference=float(np.linalg.norm(x-y)/np.linalg.norm(x))))

mp=ROOT/'mas_matrix_probe';metadata=json.loads(next(mp.rglob('mas_cluster_meta*.json')).read_text())
H=matrix(next(mp.rglob('mas_cluster_hess*.mtx')));P=matrix(next(mp.rglob('mas_cluster_inv*.mtx')))
dim=metadata['banksize']*3;blocks=[]
for i in range(metadata['num_cluster_blocks']):
    h=H[i*dim:(i+1)*dim,i*dim:(i+1)*dim].toarray()
    inv=P[i*dim:(i+1)*dim,i*dim:(i+1)*dim].toarray()
    eh=np.linalg.eigvalsh((h+h.T)*.5);ei=np.linalg.eigvalsh((inv+inv.T)*.5)
    if ei[0]<-1e-8*max(abs(ei[0]),abs(ei[-1])):
        blocks.append(dict(block=i,hessian_min_eigenvalue=float(eh[0]),hessian_max_eigenvalue=float(eh[-1]),
            stored_inverse_min_eigenvalue=float(ei[0]),stored_inverse_max_eigenvalue=float(ei[-1])))

# Algebraic application of the dumped MAS hierarchy in double precision.
# Not bitwise GPU reproduction: native restrictions/products use float32.
r2p=np.array(metadata['real_to_part']);gn=np.array(metadata['going_next'])
rr=[];cc=[]
for real,part in enumerate(r2p):
    if part<0:continue
    nodes=[int(part)];cur=real
    for _ in range(metadata['level_num']-1):cur=int(gn[cur]);nodes.append(cur)
    for node in nodes:
        for d in range(3):rr.append(3*node+d);cc.append(3*real+d)
R=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(P.shape[0],len(r2p)*3)).tocsr()
A,b=first('mas_matrix_probe');bf=b[108:];rb=R@bf;zb=P@rb;zf=R.T@zb
abd_z=np.linalg.solve(A[:108,:108].toarray(),b[:108])
rz_abd=float(b[:108]@abd_z);rz_fem=float(bf@zf)
mas=dict(blocks=metadata['num_cluster_blocks'],significantly_indefinite_stored_inverse_blocks=blocks,
    cpu_algebraic_initial_rz_abd=rz_abd,cpu_algebraic_initial_rz_fem=rz_fem,
    cpu_algebraic_initial_rz_total=rz_abd+rz_fem,
    limitation='Dumped local inverse spectra are exact analyses of exported native matrices. CPU assembled MAS application uses float64 and does not reproduce native float32 reduction order. Local indefinite blocks alone do not prove the assembled global preconditioner indefinite.')
native_root=ROOT/'native_pz_v2'
native_vectors=[]
for p in sorted(native_root.rglob('r.1.0.*.mtx')):
    r=vector(p);z=vector(p.with_name(p.name.replace('r.','z.')))
    native_vectors.append(dict(file=p.name,r_dot_z=float(r@z),abd_r_dot_z=float(r[:108]@z[:108]),fem_r_dot_z=float(r[108:]@z[108:])))
AA,bb=first('native_pz_v2');AB,bB=first('no_robot_contact')
mas['native_gpu_vector_probe']=dict(samples=native_vectors,
    relative_initial_rhs_difference=float(np.linalg.norm(bb-bB)/np.linalg.norm(bB)),
    relative_initial_matrix_difference=float(np.linalg.norm((AA-AB).data)/np.linalg.norm(AB.data)),
    scope='Native LinearPCG used only to dump the initial r,z from the same MAS preconditioner; tol_rate=0.9 and max_newton=1 bound the diagnostic. Initial preconditioner application precedes and is independent of the linear stopping test. This is not a production solver configuration.')

history=[]
for level,path in enumerate([Path('output/shirt_four_level_full_fold_20261118/L0/run'),Path('output/shirt_four_level_full_fold_20261118/L1/run'),Path('output/shirt_L2_feasibility_fix_20261118/L2/run')]):
    poses=np.load(path/'robot.npy',mmap_mode='r');sides=[]
    for side,j in [('left',2),('right',6)]:
        u,sv,vh=np.linalg.svd(poses[:,[j,j+1],:3,:3]);rot=u@vh
        relative=np.swapaxes(rot[:,0],1,2)@rot[:,1]
        angles=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
        sustained={}
        for deg in [5,10,20,30]:
            found=np.flatnonzero(np.convolve((angles>deg).astype(int),np.ones(5,dtype=int),mode='valid')==5)
            sustained[str(deg)]=None if not len(found) else int(found[0])
        sides.append(dict(side=side,first_five_consecutive_frames_above_degrees=sustained,
            relative_rotation_at_stages={str(k):float(angles[k]) for k in [60,596,1076,1616,2216,2576,2696,2936,3476]}))
    history.append(dict(level=level,sides=sides))
result=dict(probes=reports,controls=controls,mas=mas,
    full_run_rotation_history=history,
    residual_definition='norm(A*x-b)/norm(b), true residual recomputed from full-precision exported matrices and solutions; ABD is DOFs [0,108), independently confirmed from geometry dof_offset metadata.',
    limitations=['Reconstructed snapshot, not historical solver checkpoint; native adaptive/contact/friction caches are rebuilt.',
        'Four-Newton cap bounds most probes; no claim of a completed physical correction.',
        'True Euclidean residual differs from native preconditioned stopping norm. Nonzero residual alone does not locate the entire cause; decoupled ABD comparison isolates the neglected robot equations.'])
(ROOT/'analysis.json').write_text(json.dumps(result,indent=2))
for r in reports:
    last=r['linear_solves'][-1]
    print(r['name'],'converged',r['stats']['converged'],'last ABD relative residual',last['abd_true_relative_residual'],
        'max body movement mm',max(r['robot_origin_displacement_mm']))
print('MAS',mas);print('control differences',controls)
