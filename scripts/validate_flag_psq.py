"""Flag-specific PSQ operator/assembly verification, without a new dynamics run.

Checks the existing 551 -> 35000 ten-level flag, all saved L0 states, and an
actual-decimation curved flag with nonzero normal and tangential residuals.
CUDA is a freshly compiled standalone kernel harness, not a full backend test.
"""
import ctypes as ct
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import load_npz
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from progressive.prolongation import Hierarchy, ShellProlongation
from progressive.shell import Shell
from progressive.solver import IncrementalProblem
from progressive.runtime import numerical_threads
from audit_psq_prolongation import reference
from demo_psq_shallow_pleats import write_json

OUT=ROOT/'output/flag_psq_validation_20261121'
EXISTING=ROOT/'output/pdpp_flag_paper_assets_20261117_v2'
RUN=ROOT/'output/flag_L0_shirt_aligned_20261118'
TOL=2e-11


def hierarchy(assets):
    n=len(list(assets.glob('mesh_*.npz')))
    meshes=[dict(np.load(assets/f'mesh_{k}.npz')) for k in range(n)]
    return Hierarchy([m['vertices'] for m in meshes],[m['faces'] for m in meshes],
        [load_npz(assets/f'map_{k}_{k+1}.npz') for k in range(n-1)],
        [np.load(assets/f'anchors_{k}_{k+1}.npy') for k in range(n-1)],
        [np.load(assets/f'anchors_{k}_{n-1}.npy') for k in range(n-1)]+[None])


def prepare_curved():
    out=OUT/'curved_flag';out.mkdir()
    u,v=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,25))
    x=np.column_stack((.6*u.ravel(),(.012*u*u*np.sin(4*np.pi*v)+.006*np.sin(np.pi*u)*np.sin(np.pi*v)).ravel(),.25+.4*v.ravel()))
    ids=np.arange(len(x)).reshape(25,33)
    a,b,c,d=ids[:-1,:-1].ravel(),ids[:-1,1:].ravel(),ids[1:,1:].ravel(),ids[1:,:-1].ravel()
    f=np.concatenate((np.column_stack((a,b,c)),np.column_stack((a,c,d))))
    np.savez(out/'original.npz',vertices=x,faces=f)
    obj=out/'original.obj'
    with obj.open('w') as fp:
        for row in x:fp.write('v '+' '.join(format(float(t),'.17g') for t in row)+'\n')
        for row in f:fp.write('f '+' '.join(str(int(t)+1) for t in row)+'\n')
    with (out/'prepare.log').open('w') as log:
        subprocess.run([sys.executable,'-m','progressive.prepare_intrinsic','--input',str(obj),
            '--output',str(out/'assets'),'--targets','512','192','--lock-boundary',
            '--executable',str(ROOT/'output/intrinsic_plate_build/Release/intrinsic_builder.exe')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    h=hierarchy(out/'assets')
    np.testing.assert_array_equal(h.rests[-1],x);np.testing.assert_array_equal(h.faces[-1],f)
    assert np.max(np.abs(h.to_finest[0].gamma))>1e-5
    assert np.max(np.abs(h.to_finest[0].tangent))>1e-6
    return h


def deform(rest,s):
    x,y,z=rest.T;width=np.ptp(x);height=np.ptp(z);mid=(z.min()+z.max())/2
    u=(x-x.min())/width;v=(z-mid)/height;angle=.65*s*u
    return np.column_stack((x-.04*width*s*u*u,
        y*np.cos(angle)-(z-mid)*np.sin(angle)+.12*width*s*u*u*np.sin(2*np.pi*v+2*u),
        mid+y*np.sin(angle)+(z-mid)*np.cos(angle)-.06*width*s*u*u))


def maximum(a,b):return float(np.max(np.abs(a-b),initial=0))


def audit_hierarchy(h):
    rng=np.random.default_rng(2111);rows=[]
    rotations=Rotation.random(8,random_state=rng).as_matrix()
    for kind,operators in [('adjacent',h.adjacent),('direct',h.to_finest)]:
        for k,p in enumerate(operators):
            row=dict(kind=kind,level=k,coarse_vertices=len(p.coarse_rest),fine_vertices=len(p.fine_rest),
                row_sum_error=maximum(np.asarray(p.U.sum(axis=1)),1.),
                rest_error_m=maximum(p.position(p.coarse_rest),p.fine_rest),
                normal_residual_max_m=float(np.abs(p.gamma).max()),
                tangent_residual_max_m=float(np.linalg.norm(p.tangent,axis=1).max()))
            assert np.isfinite(p.U.data).all() and row['row_sum_error']<1e-12
            if kind=='adjacent':
                ids=np.repeat(np.arange(p.U.shape[0]),np.diff(p.U.indptr))
                assert np.all(np.any(p.U.indices[:,None]==p.faces[p.anchors[ids]],axis=1))
            row['rigid_max_error_m']=max(maximum(p.position(p.coarse_rest@q.T+[.4,-.3,.7]),p.fine_rest@q.T+[.4,-.3,.7]) for q in rotations)
            row['deformed_reference_max_error_m']=max(maximum(p.position(deform(p.coarse_rest,s)),reference(p,deform(p.coarse_rest,s))) for s in (.1,.4,.8,1.2))
            assert max(row['rest_error_m'],row['rigid_max_error_m'],row['deformed_reference_max_error_m'])<TOL
            rows.append(row)
    x=deform(h.rests[0],.8);linear=x.copy();nonlinear=x.copy()
    for p in h.adjacent:linear=p.U@linear;nonlinear=p.position(nonlinear)
    composition=maximum(h.to_finest[0].U@x,linear);assert composition<TOL
    return dict(passed=True,operators=rows,linear_composition_error_m=composition,
        direct_vs_composed_nonlinear_difference_m=maximum(h.to_finest[0].position(x),nonlinear),
        note='Direct nonlinear P is not required to equal a composition of adjacent nonlinear operators.')


class CUDA:
    def __init__(self,reuse=False):
        out=OUT/'cuda';out.mkdir(exist_ok=reuse)
        for n in ('psq_operator_check.cu','psq_kernels.cuh'):
            if reuse:assert (out/n).read_bytes()==(ROOT/'native/libuipc'/n).read_bytes()
            else:shutil.copy2(ROOT/'native/libuipc'/n,out/n)
        env=os.environ.copy()
        cache=ROOT/'output/flag_L0_repair_20261118/build_env.json'
        if cache.exists():env.update(json.loads(cache.read_text())['env'])
        self.path=out/'psq_check.dll'
        command=['nvcc','--shared','-O2','-std=c++17','-arch=sm_86','-allow-unsupported-compiler',
            '-ccbin','C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64',
            str(out/'psq_operator_check.cu'),'-o',str(self.path)]
        if not reuse:
            with (out/'build.log').open('w') as fp:subprocess.run(command,cwd=out,env=env,stdout=fp,stderr=subprocess.STDOUT,check=True)
            write_json(out/'build.json',dict(command=command,sha256={n:hashlib.sha256((out/n).read_bytes()).hexdigest()
                for n in ('psq_operator_check.cu','psq_kernels.cuh','psq_check.dll')}))
        else:
            for n,sha in json.loads((out/'build.json').read_text())['sha256'].items():
                assert hashlib.sha256((out/n).read_bytes()).hexdigest()==sha
        lib=ct.CDLL(str(self.path));self.fun=lib.psq_check
        self.fun.argtypes=[ct.c_int]*4+[ct.c_void_p]*12;self.fun.restype=ct.c_int

    def evaluate(self,p,x,g):
        ints=[p.U.indptr,p.U.indices,p.faces,p.anchors]
        row,col,faces,anchors=[np.ascontiguousarray(a,dtype=np.int32) for a in ints]
        val,inv,gamma,tangent,x,g=[np.ascontiguousarray(a,dtype=np.float64) for a in (p.U.data,p.inv_dm,p.gamma,p.tangent,x,g)]
        y=np.zeros((len(p.fine_rest),3));gc=np.zeros_like(x)
        buffers=[row,col,val,faces,anchors,inv,gamma,tangent,x,g,y,gc]
        code=self.fun(len(y),len(x),len(faces),p.U.nnz,*[a.ctypes.data for a in buffers])
        return code,y,gc


def assembly_checks(h):
    rng=np.random.default_rng(2112);p=h.to_finest[0]
    x=deform(p.coarse_rest,.08);fixed=np.isclose(p.coarse_rest[:,0],0)
    shell=Shell(p.coarse_rest,p.faces,strain_limit=None)
    finest=Shell(p.fine_rest,h.faces[-1],strain_limit=None)
    xhat=x+[0,.0001,0];dt=.02;gravity=np.array([0.,2.,-9.81])
    problem=IncrementalProblem(shell,finest,p,dt,xhat,p.coarse_rest,gravity=gravity)
    offset=p.offset(x);value,g,H=problem.evaluate(x,offset,True,'coarse')
    ef,gf,_=finest.evaluate(reference(p,x))
    expected=ef+np.sum(shell.mass[:,None]*(x-xhat)**2)/(2*dt**2)-np.sum(shell.mass[:,None]*x*gravity)
    expected_g=p.U.T@gf+shell.mass[:,None]*((x-xhat)/dt**2-gravity)
    gradient_error=maximum(g,expected_g)
    assert abs(value-expected)<1e-10 and gradient_error<1e-9
    from scipy.sparse import diags
    expected_H=shell.evaluate(x,True)[2]+diags(np.repeat(shell.mass/dt**2,3))
    herror=float(np.max(np.abs((H-expected_H).data),initial=0));assert herror<1e-9
    tests=[]
    for j in range(12):
        d=rng.normal(size=x.shape);d[fixed]=0;d/=np.linalg.norm(d)
        analytic=float(np.sum(g*d));frozen_errors=[];full_errors=[];velocity_errors=[]
        full_expected=float(np.sum(gf*p.velocity(x,d)))
        for eps in (1e-5,3e-6,1e-6):
            fd=(problem.evaluate(x+eps*d,offset)[0]-problem.evaluate(x-eps*d,offset)[0])/(2*eps)
            frozen_errors.append(abs(fd-analytic)/max(1.,abs(analytic)))
            fdv=(reference(p,x+eps*d)-reference(p,x-eps*d))/(2*eps)
            velocity_errors.append(maximum(fdv,p.velocity(x,d)))
            fd_full=(finest.evaluate(reference(p,x+eps*d))[0]-finest.evaluate(reference(p,x-eps*d))[0])/(2*eps)
            full_errors.append(abs(fd_full-full_expected)/max(1.,abs(full_expected)))
        assert frozen_errors[-1]<2e-6 and full_errors[-1]<2e-6 and velocity_errors[-1]<2e-7
        tests.append(dict(frozen_directional_relative_errors=frozen_errors,full_directional_relative_errors=full_errors,
            exact_DP_directional_errors=velocity_errors,
            full_minus_frozen_elastic_derivative=full_expected-float(np.sum((p.U.T@gf)*d))))
    identity=h.to_finest[-1];direct=IncrementalProblem(finest,finest,identity,dt,p.fine_rest,p.fine_rest,gravity=(0,0,0))
    probe=deform(p.fine_rest,.03);e,g,_=direct.evaluate(probe,identity.offset(probe))
    ee,gg,_=finest.evaluate(probe);dx=probe-p.fine_rest
    assert abs(e-ee-np.sum(finest.mass[:,None]*dx*dx)/(2*dt**2))<1e-10
    assert maximum(g,gg+finest.mass[:,None]*dx/dt**2)<1e-9
    return dict(passed=True,energy_assembly_abs_error_J=abs(value-expected),gradient_assembly_max_error=gradient_error,
        hessian_preconditioner_max_error=herror,directions=tests,identity_no_double_counting=True,
        distinction='U.T*g is the frozen-offset derivative, not the full nonlinear PSQ energy gradient. Full derivative tested separately through DP.')


def guard_and_negative_controls(h,cuda):
    p=h.to_finest[0];q=Rotation.from_rotvec([.7,.4,-.3]).as_matrix()
    x=p.coarse_rest@q.T;truth=p.fine_rest@q.T;delta=p.fine_rest-p.U@p.coarse_rest
    fixed=p.U@x+delta
    # A normal-omission control uses the production normal residual only for
    # constructing the deliberately incorrect output, never as the oracle.
    tri=x[p.faces];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    omit_normal=p.position(x)-n[p.anchors]*p.gamma[:,None]
    bad_errors=dict(linear=maximum(p.U@x,truth),fixed_world_offset=maximum(fixed,truth),omit_normal=maximum(omit_normal,truth))
    assert min(bad_errors.values())>1e-6
    # Test deformation of tangent residual by F instead of polar R.
    z=p.coarse_rest*[1.25,.85,1.15];tri=z[p.faces]
    F=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)@p.inv_dm
    n=np.cross(F[:,:,0],F[:,:,1]);n/=np.linalg.norm(n,axis=1)[:,None]
    wrong=p.U@z+n[p.anchors]*p.gamma[:,None]+np.einsum('nij,nj->ni',F[p.anchors],p.tangent)
    bad_errors['stretch_tangent_with_F']=maximum(wrong,reference(p,z));assert bad_errors['stretch_tangent_with_F']>1e-6
    bad=p.coarse_rest.copy();face=p.faces[p.anchors[0]];bad[face[1]]=bad[face[0]]
    rejected=False
    try:p.position(bad)
    except ValueError:rejected=True
    code,_,_=cuda.evaluate(p,bad,np.zeros_like(p.fine_rest));assert rejected and code==4
    scales=[]
    for s in (.001,1.,1000.):
        scaled=ShellProlongation(p.coarse_rest*s,p.fine_rest*s,p.faces,p.U,p.anchors)
        error=maximum(scaled.position(x*s),p.position(x)*s)/s;assert error<TOL
        scales.append(dict(scale=s,normalized_error_m=error))
    return dict(passed=True,negative_control_errors_m=bad_errors,degenerate_cpu_rejected=rejected,degenerate_cuda_code=code,unit_scaling=scales)


def figure(h):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from demo_psq_ruffled_cape import Camera
    p=h.to_finest[0];q=Rotation.from_rotvec([.3,0,.8]).as_matrix()
    rest=p.coarse_rest;x=rest@q.T;truth=p.fine_rest@q.T
    states=[p.fine_rest,p.U@rest,p.position(rest),truth,p.U@x,p.position(x)]
    cam=Camera(states);fig,axes=plt.subplots(2,3,figsize=(14,9),facecolor='#f4f6f9')
    titles=['ORIGINAL / REST','LINEAR / REST','PSQ / REST','EXACT RIGID REFERENCE','LINEAR / ROTATED','PSQ / ROTATED']
    for i,(ax,y,title) in enumerate(zip(axes.flat,states,titles)):
        cam.draw(ax,y,h.faces[-1]);ax.set_title(title,fontsize=13,weight='bold',color='#213147')
        target=p.fine_rest if i<3 else truth
        ax.text(.5,-.02,f'Max coordinate error: {maximum(y,target)*1000:.3e} mm',transform=ax.transAxes,ha='center',fontsize=10,color='#526078')
    fig.text(.035,.95,'FLAG / PSQ CORRECTNESS CHECK',fontsize=23,weight='bold',color='#213147')
    fig.text(.035,.9,f'Actual decimation: {len(rest)} -> {len(p.fine_rest)} vertices | Nonzero normal + tangential residuals',fontsize=12,color='#526078')
    fig.text(.035,.035,'Controlled operator tests, not a new physical simulation. No geometry smoothing or displacement amplification.',fontsize=11,color='#526078')
    fig.subplots_adjust(left=.025,right=.975,bottom=.10,top=.82,hspace=.20,wspace=.07)
    fig.savefig(OUT/'verification.png',dpi=120);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-fixtures',action='store_true');args=parser.parse_args()
    start=time.perf_counter();OUT.mkdir(exist_ok=args.reuse_fixtures)
    shutil.copy2(__file__,OUT/'source_snapshot.py')
    with numerical_threads(1):
        curved=hierarchy(OUT/'curved_flag/assets') if args.reuse_fixtures else prepare_curved()
        actual=hierarchy(EXISTING);cuda=CUDA(args.reuse_fixtures)
        results={'existing_flag':audit_hierarchy(actual),'curved_flag':audit_hierarchy(curved)}
        print('Both hierarchy formula audits passed',flush=True)
        rng=np.random.default_rng(2113);gpu=[]
        for label,h in [('existing',actual),('curved',curved)]:
            for k,p in enumerate(h.to_finest):
                x=deform(p.coarse_rest,.6);g=rng.normal(size=p.fine_rest.shape)
                code,y,gc=cuda.evaluate(p,x,g);pe=maximum(y,reference(p,x));ge=maximum(gc,p.U.T@g)
                assert code==0 and pe<TOL and ge<1e-9
                gpu.append(dict(hierarchy=label,level=k,position_max_error_m=pe,frozen_restriction_max_error=ge))
        results['cuda']=dict(passed=True,checks=gpu,scope='Fresh standalone kernel build; not a full libuipc runtime validation.')
        meta=json.loads((RUN/'run.json').read_text());states=np.load(RUN/'positions.npy',mmap_mode='r')
        assert meta['completed_steps']+1==len(states) and Path(meta['assets']).resolve()==EXISTING.resolve()
        p=actual.to_finest[0];saved_error=0.;sample_cuda=[]
        # Verify the operator serialized for the actual saved run, not just
        # another hierarchy with the same number of vertices.
        from scipy.sparse import csr_matrix
        bundle=np.load(RUN/'operator/operator.npz')
        U=csr_matrix((bundle['U_data'],bundle['U_indices'],bundle['U_indptr']),shape=tuple(bundle['U_shape']))
        serialized_error=max(float(np.max(np.abs((U-p.U).data),initial=0)),
            *[maximum(bundle[key],getattr(p,key)) for key in ('coarse_rest','fine_rest','gamma','tangent','inv_dm')])
        assert serialized_error<TOL
        pins=p.coarse_rest[:,0]<1e-4;fine_pins=np.isclose(p.fine_rest[:,0],0)
        assert pins.sum()==meta['pins']
        proxy_pin_error=0.;coarse_pin_error=0.;worst_frame=0
        for i,x in enumerate(states):
            prolonged=p.position(x)
            saved_error=max(saved_error,maximum(prolonged,reference(p,x)))
            coarse_pin_error=max(coarse_pin_error,maximum(x[pins],p.coarse_rest[pins]))
            pin_error=float(np.max(np.linalg.norm(prolonged[fine_pins]-p.fine_rest[fine_pins],axis=1)))
            if pin_error>proxy_pin_error:proxy_pin_error=pin_error;worst_frame=i
            if i%50==0 or i==206:
                code,y,gc=cuda.evaluate(p,x,np.zeros_like(p.fine_rest));err=maximum(y,reference(p,x))
                assert code==0 and err<TOL;sample_cuda.append(dict(frame=i,error_m=err))
        assert saved_error<TOL
        historical=json.loads((RUN/'audit.json').read_text())
        results['saved_states']=dict(passed=True,checked_states=len(states),position_max_error_m=saved_error,cuda_samples=sample_cuda,
            serialized_operator_max_error=serialized_error,
            historical_termination_counts=historical['termination_counts'],
            scope='Read-only operator evaluation on stored states. Historical preview stops are not Newton convergence; trajectory correctness is not certified.')
        curved_pin_error=max(maximum(p.position(deform(p.coarse_rest,1.2))[np.isclose(p.fine_rest[:,0],0)],
            p.fine_rest[np.isclose(p.fine_rest[:,0],0)]) for p in curved.to_finest)
        assert curved_pin_error<TOL
        results['boundary']=dict(existing_coarse_pin_coordinate_error_m=coarse_pin_error,
            existing_prolonged_proxy_pin_distance_m=proxy_pin_error,worst_saved_frame=worst_frame,
            existing_coarse_pins=int(pins.sum()),existing_fine_boundary_vertices=int(fine_pins.sum()),
            curved_locked_boundary_error_m=curved_pin_error,
            distinction='Proxy-boundary movement is not coarse pin drift or a formula mismatch. Rotated residuals do not automatically preserve all fine boundary constraints.')
        results['assembly']=assembly_checks(curved)
        results['guards_and_negative_controls']=guard_and_negative_controls(curved,cuda)
        figure(curved)
    sources=[Path(__file__),ROOT/'progressive/prolongation.py',ROOT/'progressive/solver.py',ROOT/'progressive/shell.py',
        ROOT/'scripts/audit_psq_prolongation.py',RUN/'positions.npy',RUN/'run.json',RUN/'audit.json',RUN/'operator/operator.npz']
    for assets in (EXISTING,OUT/'curved_flag/assets'):
        sources += sorted(assets.glob('mesh_*.npz'))+sorted(assets.glob('map_*.npz'))+sorted(assets.glob('anchors_*.npy'))+[assets/'manifest.json']
    results.update(operator_checks_passed=True,dynamics_certified=False,seconds=time.perf_counter()-start,
        tolerance_m=TOL,sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    write_json(OUT/'audit.json',results)
    print(json.dumps({k:v for k,v in results.items() if k not in ('sha256','existing_flag','curved_flag')},indent=2),flush=True)


if __name__=='__main__':main()
