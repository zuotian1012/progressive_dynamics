"""Re-solve a recorded step with current code without changing its input run."""
import argparse
import json
from pathlib import Path
import sys
import time
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from progressive.run import prepare_scene
from progressive.shell import Shell
from progressive.affine import AffineObstacles
from progressive.contact import IPCContact
from progressive.integrators import predict
from progressive.solver import IncrementalProblem,NewtonOptions,solve
from progressive.runtime import numerical_threads


def load_replay(root,level,step,paper_aligned=False):
    cfg=json.loads((root/'experiment.json').read_text())
    if cfg['parameters'].get('warm_start','horizontal')=='vertical':
        raise ValueError('This replay reconstructs horizontal starts only; vertical runs require their safe-expansion initializer and are not supported here')
    args=argparse.Namespace(**cfg['parameters'])
    args.assets=Path(args.assets)
    for name,value in [('curvature',0.),('grip_jaw_height',.026),('grip_initial_gap',.001)]:
        if not hasattr(args,name):setattr(args,name,value)
    for name,value in dict(strain_model='legacy',strain_stiffness=1e4,strain_activation=1+(args.strain_limit-1)*.5,
                           elastic_hessian='finite_difference',convergence_norm='max',ccd='tight',hinge_difference='periodic').items():
        if not hasattr(args,name):setattr(args,name,value)
    if paper_aligned:
        args.strain_model='cipc';args.strain_stiffness=1e3;args.elastic_hessian='exact'
        args.convergence_norm='rms';args.ccd='additive'
        args.hinge_difference='principal'
    hierarchy,bodies,fixed,targets,pins,steps,dt,_=prepare_scene(args)
    shells=[Shell(x,f,young=args.young,poisson=getattr(args,'poisson',.3),thickness=args.thickness,density=args.density,
                  strain_limit=args.strain_limit,strain_activation=args.strain_activation,strain_stiffness=args.strain_stiffness,
                  strain_model=args.strain_model,elastic_hessian=args.elastic_hessian,hinge_difference=args.hinge_difference)
            for x,f in zip(hierarchy.rests,hierarchy.faces)]
    tracks=[np.load(root/f'velpro_L{k}'/'positions.npy',mmap_mode='r') for k in range(level+1)]
    meta=json.loads((root/f'velpro_L{level}'/'run.json').read_text())
    if step>meta['completed_steps']:raise ValueError('Input warm start was not accepted')
    shell=shells[level]
    if step and not paper_aligned:
        reports=json.loads((root/f'velpro_L{level}'/'solver.json').read_text())
        if 'strain_stiffness' in reports[step-1]:shell.set_strain_stiffness(reports[step-1]['strain_stiffness'])
    embedding=AffineObstacles(shell.rest,shell.faces,bodies,fixed)
    q=embedding.q0.copy();q[:len(shell.rest)]=tracks[level][step]
    poses=np.load(root/f'velpro_L{level}'/'robot.npy',mmap_mode='r')[step]
    for i,pose in enumerate(poses):q[len(shell.rest)+4*i:len(shell.rest)+4*i+4]=np.vstack((pose[:3,3],pose[:3,:3].T))
    # CollisionMesh rest geometry affects IPC edge-edge mollification. It must
    # remain the ORIGINAL rest geometry, never the deformed checkpoint state.
    contact_rest=embedding.positions(embedding.q0)
    contact=IPCContact(contact_rest,embedding.faces,len(shell.rest),args.dhat,args.dmin,args.contact_stiffness,args.friction,ccd=args.ccd)
    contact.validate(embedding.positions(q))
    embedding.set_targets(targets(step))
    xhat=predict('velpro',level,step,tracks,hierarchy.adjacent,dt,[np.zeros_like(x) for x in hierarchy.rests])
    target=hierarchy.adjacent[level-1].position(tracks[level-1][step+1]) if level else None
    problem=IncrementalProblem(shell,shells[-1],hierarchy.to_finest[level],dt,xhat,q,
                               target=target,weight=args.consistency_weight if level else 0.,
                               contact=contact,embedding=embedding,auxiliary=embedding)
    digest=lambda value:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    inputs={'state_sha256':digest(q),'predictor_sha256':digest(xhat),
            'collision_rest_sha256':digest(contact_rest),'target_sha256':digest(embedding.target)}
    return problem,q,args,inputs


def replay(root,level,step,output,max_iterations,repeats=1,threads=1,initial_state=None,paper_aligned=False):
    if output.exists(): raise ValueError('Diagnostic output already exists')
    output.mkdir(parents=True)
    print(f'Replaying accepted input at L{level}, step {step}',flush=True)
    summaries=[]
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('progressive').glob('*.py')}
    with numerical_threads(threads) as runtime:
        for repeat in range(repeats):
            problem,q,args,inputs=load_replay(root,level,step,paper_aligned)
            search=getattr(args,'offset_linesearch','frozen')
            initial=q if initial_state is None else np.load(initial_state)
            if initial.shape!=q.shape:raise ValueError('Wrong warm-start shape')
            inputs['warm_start_sha256']=hashlib.sha256(initial.tobytes()).hexdigest()
            # Only the optimization initial guess changes: inertia prediction,
            # lagged friction, original rest geometry and driven targets stay fixed.
            options=NewtonOptions(max_iterations,args.tolerance,hessian=args.hessian,offset_linesearch=search,
                                  allow_preview_stall=(not getattr(args,'strict_preview',True) and level<args.levels-1),
                                  preview_min_alpha=getattr(args,'preview_min_alpha',1e-8),convergence_norm=args.convergence_norm)
            result,report=solve(problem,initial,options)
            report.update(source=str(root),level=level,step=step,strain_hessian='analytic spectral',offset_linesearch=search,
                          paper_aligned_override=paper_aligned,elastic_hessian=args.elastic_hessian,ccd=args.ccd,
                          hinge_difference=args.hinge_difference,
                          comparison_scope='Changed barrier model and solver; diagnostic only, not same incremental potential' if paper_aligned else 'Recorded model with current implementation',
                          input_hashes=inputs,runtime=runtime,repeat=repeat,warm_start_source=str(initial_state) if initial_state else None,
                          code_sha256=source_hashes)
            folder=output if repeats==1 else output/f'repeat_{repeat}'
            folder.mkdir(exist_ok=True)
            np.save(folder/'state.npy',result)
            (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            summary={k:report[k] for k in ('accepted','converged','preview_stop','reason','iterations','seconds','energy','gradient_inf','max_stretch','input_hashes')}
            summary['state_sha256']=hashlib.sha256(result.tobytes()).hexdigest()
            summaries.append(summary)
            print(json.dumps(summary),flush=True)
    (output/'repeats.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    if not all(s['accepted'] for s in summaries):raise RuntimeError('At least one replay was not accepted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--level',type=int,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-iterations',type=int,default=150)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--initial-state',type=Path,help='Continue optimization of the SAME incremental problem from this state')
    p.add_argument('--paper-aligned',action='store_true',help='Diagnostic with corrected C-IPC model, exact elastic Hessians, RMS convergence and ACCD; changes the original problem')
    a=p.parse_args()
    replay(a.root,a.level,a.step,a.output,a.max_iterations,a.repeats,a.threads,a.initial_state,a.paper_aligned)
