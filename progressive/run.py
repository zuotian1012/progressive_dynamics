"""Run the paper framework on a flag, a frictional grasp, or Scene527 assets.

python -m progressive.run --scene grip --steps 60 --output output/pd_grip
python -m progressive.run --scene shirt --frames 740 --output output/pd_shirt
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.sparse import load_npz
import trimesh
from cloth.mesh import grid,prolongation
from .prolongation import Hierarchy
from .shell import Shell
from .contact import IPCContact
from .affine import AffineObstacles
from .solver import IncrementalProblem,NewtonOptions,solve
from .integrators import predict,METHODS
from .metrics import consistency_metrics,continuity
from .runtime import numerical_threads
from .checkpoint import validate_resume,restore_level


def mesh_hash(rest,faces):
    return hashlib.sha256(rest.tobytes()+faces.tobytes()).hexdigest()


def robot_targets(robot,settle_steps,substeps,hold_after_frame=None):
    def targets(step):
        if step<settle_steps:return robot.poses(robot.q[0])
        frame,sub=divmod(step-settle_steps,substeps)
        if hold_after_frame is not None:
            frame=min(frame,hold_after_frame)
            if step-settle_steps>=(hold_after_frame+1)*substeps:
                return robot.poses(robot.q[hold_after_frame])
        q=robot.q[max(0,frame-1)]+(sub+1)/substeps*(robot.q[frame]-robot.q[max(0,frame-1)])
        return robot.poses(q)
    return targets


def grip_bodies_and_targets(args):
    """Physical jaws shared by nested and intrinsic-hierarchy diagnostics."""
    bodies=[]
    for side in (-1,1):
        box=trimesh.creation.box(extents=[.1,.005,args.grip_jaw_height])
        t=np.eye(4);t[:3,3]=[0,side*(.0025+args.grip_initial_gap/2),.07]
        bodies.append((str(side),np.asarray(box.vertices),np.asarray(box.faces),t))
    def targets(step):
        t=(step+1)*args.dt
        close=min(t/.15,1.)
        lift=np.clip((t-.2)/.6,0,1)*.025
        result={}
        for name,_,_,initial in bodies:
            half_gap=args.grip_initial_gap/2
            pose=initial.copy();pose[1,3]=int(name)*(.0025+half_gap-max(0.,half_gap-.00025)*close);pose[2,3]+=lift
            result[name]=pose
        return result
    return bodies,targets


def preflight_contacts(hierarchy,bodies,fixed,args):
    """All levels must be feasible before spending time on the first preview."""
    contacts=[]
    for level,(x,f) in enumerate(zip(hierarchy.rests,hierarchy.faces)):
        embedding=AffineObstacles(x,f,bodies,fixed)
        contact=IPCContact(embedding.positions(embedding.q0),embedding.faces,len(x),
                           args.dhat,args.dmin,args.contact_stiffness,args.friction,ccd=args.ccd)
        try:contact.validate(embedding.positions(embedding.q0))
        except ValueError as error:raise ValueError(f'Initial hierarchy L{level} is infeasible: {error}') from error
        contacts.append(contact)
    return contacts


def prepare_scene(args):
    bodies,fixed=[],[]
    robot=None
    adjacent_anchors,finest_anchors=None,None
    if args.scene in ('shirt','intrinsic','intrinsic-grip','bunny'):
        from shirt_demo.assets import Robot
        rests,faces,maps=[],[],[]
        for k in range(args.levels):
            with np.load(args.assets / f'mesh_{k}.npz') as data:
                rests.append(data['vertices']);faces.append(data['faces'])
            if k: maps.append(load_npz(args.assets / f'map_{k-1}_{k}.npz'))
        def anchor_file(name):
            path=args.assets/name
            return np.load(path) if path.exists() else None
        adjacent_anchors=[anchor_file(f'anchors_{k}_{k+1}.npy') for k in range(args.levels-1)]
        finest_anchors=[anchor_file(f'anchors_{k}_{args.levels-1}.npy') for k in range(args.levels)]
        if args.scene=='shirt':
            robot=Robot()
            poses=robot.poses(robot.q[0])
            bodies=[(name,v,f,poses[name]) for name,v,f in robot.collisions()]
            box=trimesh.creation.box(extents=[1,2,.8])
            fixed=[(np.asarray(box.vertices)+[.65,0,.4],np.asarray(box.faces))]
            steps=args.settle_steps+args.frames*args.substeps
            dt=1/(60*args.substeps)
            targets=robot_targets(robot,args.settle_steps,args.substeps,getattr(args,'hold_after_frame',None))
        else:
            steps,dt=args.steps,args.dt
            targets=lambda step: {}
            if args.scene=='intrinsic-grip':bodies,targets=grip_bodies_and_targets(args)
            if args.scene=='bunny':
                with np.load(args.assets/'bunny.npz',allow_pickle=False) as data:
                    fixed=[(data['vertices'],data['faces'])]
        pins=lambda rest: np.zeros(len(rest),bool)
    else:
        meshes=[grid(args.base_segments*2**k) for k in range(args.levels)]
        maps=[prolongation(a,b) for a,b in zip(meshes,meshes[1:])]
        faces=[m.faces for m in meshes]
        if args.scene=='grip':
            rests=[np.column_stack((m.rest[:,0]*.08,np.zeros(len(m.rest)),m.rest[:,1]*.08)) for m in meshes]
            bodies,targets=grip_bodies_and_targets(args)
            pins=lambda rest: np.zeros(len(rest),bool)
        else:
            rests=[np.column_stack((m.rest[:,0]*.3,args.curvature*np.sin(np.pi*m.rest[:,0]),.25+m.rest[:,1]*.3)) for m in meshes]
            pins=lambda rest: np.isclose(rest[:,0],-.15)
            targets=lambda step: {}
        steps,dt=args.steps,args.dt
    return Hierarchy(rests,faces,maps,adjacent_anchors,finest_anchors),bodies,fixed,targets,pins,steps,dt,robot


def run(args,runtime=None):
    # Programmatic callers from earlier tests retain the horizontal default.
    for key,value in {'warm_start':'horizontal','expansion_seed':527,'expansion_gap_cap':.001,'expansion_min_gap':None,
                      'expansion_fixed_policy':'unsplit-lineage','poisson':.3,'config':None,
                      'only_level':None,'coarse_from':None}.items():
        if not hasattr(args,key):setattr(args,key,value)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    hierarchy,bodies,fixed,targets,pins,steps,dt,robot=prepare_scene(args)
    shells=[Shell(x,f,young=args.young,poisson=args.poisson,thickness=args.thickness,density=args.density,
                  strain_limit=args.strain_limit,strain_activation=args.strain_activation,
                  strain_stiffness=args.strain_stiffness,strain_model=args.strain_model,elastic_hessian=args.elastic_hessian,
                  hinge_difference=args.hinge_difference)
            for x,f in zip(hierarchy.rests,hierarchy.faces)]
    options=NewtonOptions(args.max_iterations,args.tolerance,hessian=args.hessian,
                          offset_linesearch=args.offset_linesearch,preview_min_alpha=args.preview_min_alpha,
                          convergence_norm=args.convergence_norm)
    material={k:getattr(args,k) for k in ('young','poisson','thickness','density','strain_limit')}
    config={'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'dt':dt,'steps':steps,'material':material,'mesh_hashes':[mesh_hash(x,f) for x,f in zip(hierarchy.rests,hierarchy.faces)],
            'algorithm':'PD/PD++ shared-finest shell energy; PSQ P=Ux+a with lagged offsets; current-level IPC/friction/strain',
            'hierarchy':'input maps and anchor files; intrinsic QSlim/self-parameterization or nested midpoint, identified by asset_manifest when available',
            'warm_start':args.warm_start+'; vertical uses exported intrinsic histories and safe refinement',
            'preview_policy':'strict Newton convergence' if args.strict_preview else 'PD 2024 Sec. 4.6: non-descent preview stops accepted and explicitly logged; finest/direct must converge',
            'solver':'inexact projected Newton; '+args.elastic_hessian+' elastic Hessians; '+args.convergence_norm+' Newton decrement',
            'strain_hessian':'analytic spectral Hessian; '+args.strain_model+' barrier normalization and stiffness policy',
            'line_search':args.offset_linesearch+' PSQ offsets; gradients use per-Newton frozen offsets',
            'psq_nonzero_offset_levels':[not p.zero_offset for p in hierarchy.to_finest],
            'contact':'IPC Toolkit; '+args.ccd+' CCD with finite gap; friction lagged once per timestep',
            'grasp':'contact/friction only; affine jaws with orthogonality and pose-drive energies',
            'backend':'CPU reference, not the native CUDA demo',
            'python':platform.python_version(),'platform':platform.platform(),
            'versions':{n:importlib.metadata.version(n) for n in ('numpy','scipy','ipctk','trimesh')},
            'completed':False,'runs':[]}
    config['runtime']=runtime
    config['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).parent.glob('*.py')}
    if args.scene in ('shirt','intrinsic','intrinsic-grip','bunny') and (args.assets/'manifest.json').exists():
        config['asset_manifest']=json.loads((args.assets/'manifest.json').read_text(encoding='utf-8'))
    if args.config:
        config['input_config']={'path':str(args.config),'sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
                                'content':json.loads(args.config.read_text(encoding='utf-8'))}
    if args.scene=='bunny':
        config['grasp']='None; free cloth falls under gravity onto a fixed bunny, without pins or prescribed cloth motion'
        config['asset_files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(args.assets.iterdir()) if p.suffix in ('.npz','.npy')}
    if robot: config['robot_q_sha256']=hashlib.sha256(robot.q.tobytes()).hexdigest()
    if args.warm_start=='vertical':
        config['expansion_history_sha256']={f'expansion_{k}_{k+1}.json':hashlib.sha256(
            (args.assets/f'expansion_{k}_{k+1}.json').read_bytes()).hexdigest() for k in range(args.levels-1)}
    if args.resume_from:
        config['continuation']=validate_resume(args.resume_from,config,args.resume_allow_code_change)
    if args.coarse_from:
        # A completed lower trajectory is an explicit input, not a continuation
        # of the new L1 solve. Preserve and disclose its different tolerance.
        previous=json.loads((args.coarse_from/'experiment.json').read_text())
        comparison=dict(config,parameters=dict(config['parameters']))
        comparison['parameters']['tolerance']=previous['parameters']['tolerance']
        provenance=validate_resume(args.coarse_from,comparison)
        source_meta=json.loads((args.coarse_from/f'{args.methods[0]}_L0'/'run.json').read_text())
        if not source_meta['completed'] or source_meta['completed_steps']!=steps:
            raise ValueError('L1-only requires a completed L0 trajectory on the same time grid')
        config['coarse_input']=dict(provenance,level=0,tolerance=previous['parameters']['tolerance'],
            new_level_tolerance=args.tolerance,scope='Existing L0 input; L1 starts from rest, no old L1 frames reused',
            positions_sha256=hashlib.sha256((args.coarse_from/f'{args.methods[0]}_L0'/'positions.npy').read_bytes()).hexdigest())
    def save_config():
        (args.output/'experiment.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    save_config()
    histories=[]
    try:
        contacts=preflight_contacts(hierarchy,bodies,fixed,args) if args.scene!='flag' else [None]*len(shells)
        if args.warm_start=='vertical':
            from .expansion_history import ExpansionHistory
            if args.levels!=2:
                raise ValueError('The vertical dynamics driver currently requires two levels so refinement uses the global finest proxy; multilevel horizontal runs remain supported')
            if args.scene not in ('shirt','intrinsic','intrinsic-grip'):
                raise ValueError('Vertical initialization requires intrinsic assets with exported expansion histories')
            for level in range(args.levels-1):
                path=args.assets/f'expansion_{level}_{level+1}.json'
                histories.append(ExpansionHistory.load(path,hierarchy.rests[level],hierarchy.faces[level],
                                                        hierarchy.rests[level+1],hierarchy.faces[level+1]))
            config['expansion_histories']=[h.summary() for h in histories]
    except Exception as error:
        config['error']=repr(error);config['failure_phase']='hierarchy_preflight';save_config()
        raise
    # Reuse the common proxy coarsest solve for progressive methods, but NEVER
    # reuse an independent coarse solve whose elastic discretization differs.
    shared_coarse=None
    for method in args.methods:
        tracks=[];pose_tracks=[];velocities=[np.zeros_like(x) for x in hierarchy.rests]
        for level,shell in enumerate(shells):
            if args.only_level is not None and level>args.only_level:
                break
            # Adaptive material state belongs to a trajectory, never to a
            # previously simulated method (in particular the Direct control).
            shell.set_strain_stiffness(args.strain_stiffness)
            options.allow_preview_stall=(not args.strict_preview and method!='direct' and level<len(shells)-1)
            folder=args.output/f'{method}_L{level}'
            if level==0 and method!='direct' and shared_coarse is not None:
                tracks.append(shared_coarse['track'])
                pose_tracks.append(shared_coarse['poses'])
                config['runs'].append({'method':method,'level':0,'reused_from':shared_coarse['folder'],
                                       'physics_seconds':shared_coarse['seconds']})
                continue
            folder.mkdir()
            embedding=AffineObstacles(shell.rest,shell.faces,bodies,fixed)
            q=embedding.q0.copy()
            contact=contacts[level]
            # Direct control uses each mesh's OWN elastic energy.
            transfer=hierarchy.to_finest[level]
            elastic=shells[-1]
            if method=='direct':
                from scipy.sparse import eye
                from .prolongation import ShellProlongation
                transfer=ShellProlongation(shell.rest,shell.rest,shell.faces,eye(len(shell.rest)))
                elastic=shell
            track=np.lib.format.open_memmap(folder/'positions.npy',mode='w+',dtype='float64',shape=(steps+1,len(shell.rest),3))
            track[0]=shell.rest
            tracks.append(track)
            poses=np.lib.format.open_memmap(folder/'robot.npy',mode='w+',dtype='float64',shape=(steps+1,len(bodies),4,4))
            poses[0]=embedding.transforms(q)
            pose_tracks.append(poses)
            np.save(folder/'faces.npy',shell.faces)
            for i,(_,v,f,_) in enumerate(bodies):
                np.savez_compressed(folder/f'collider_{i}.npz',vertices=v,faces=f)
            for i,(v,f) in enumerate(fixed):
                np.savez_compressed(folder/f'fixed_collider_{i}.npz',vertices=v,faces=f)
            fixed_mask=np.zeros(q.shape,bool);fixed_mask[:len(shell.rest)]=pins(shell.rest)[:,None]
            reports=[];metrics=[];timings=[]
            meta={'method':method,'level':level,'vertices':len(shell.rest),'faces':len(shell.faces),
                  'completed':False,'completed_steps':0,'preview_stopped_steps':0,'steps':steps,'dt':dt,'backend':'CPU paper framework',
                  'energy':'own shell' if method=='direct' else 'shared finest proxy',
                  'physics_seconds':0.,'consistency_weight':args.consistency_weight if level and method!='direct' else 0.}
            start_step=0
            restore_root=args.coarse_from if level==0 and args.coarse_from else args.resume_from
            if restore_root:
                restored=restore_level(restore_root,method,level,track,poses,embedding,steps,args.resume_interrupted)
                if restored:
                    start_step,q,reports,metrics,timings=restored
                    if reports and 'strain_stiffness' in reports[-1]:shell.set_strain_stiffness(reports[-1]['strain_stiffness'])
                    if contact:contact.validate(embedding.positions(q))
                    meta.update(completed_steps=start_step,physics_seconds=sum(timings),inherited_steps=start_step,
                                preview_stopped_steps=sum(r.get('preview_stop',False) for r in reports))
                    if level==0 and args.coarse_from:
                        meta.update(inherited_from=str(args.coarse_from.resolve()),
                                    tolerance=previous['parameters']['tolerance'])
            def save_level():
                track.flush();poses.flush()
                (folder/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
                (folder/'solver.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
                (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
                np.savez(folder/'timing.npz',step_seconds=timings)
            save_level()
            try:
                for step in range(start_step,steps):
                    if (args.output/'STOP').exists():
                        raise InterruptedError('Stopped at an accepted checkpoint by the output/STOP marker')
                    tick=time.perf_counter()
                    xhat=predict(method,level,step,tracks,hierarchy.adjacent,dt,velocities)
                    target=hierarchy.adjacent[level-1].position(tracks[level-1][step+1]) if level and method!='direct' else None
                    embedding.set_targets(targets(step))
                    gravity=np.array([0.,0.,-9.81])
                    if args.scene=='flag': gravity+=np.array([0.,8*np.sin((step+1)*dt*5),0.])
                    problem=IncrementalProblem(shell,elastic,transfer,dt,xhat,q,gravity,target,meta['consistency_weight'],
                                               contact,embedding,embedding if bodies else None)
                    initial=q
                    initialization=None
                    if args.warm_start=='vertical' and level and method!='direct':
                        from .safe_expansion import initialize
                        from .refinement_contact import ExpansionOptions,RefinementFailure
                        initial=q.copy()
                        for i,t in enumerate(pose_tracks[level-1][step+1]):
                            initial[embedding.n+4*i:embedding.n+4*i+4]=np.vstack((t[:3,3],t[:3,:3].T))
                        world=embedding.positions(initial);obstacle_faces=embedding.faces[len(shell.faces):]-embedding.n
                        obstacles=[(world[embedding.n:],obstacle_faces)] if len(obstacle_faces) else []
                        def refinement_progress(batch):
                            if (args.output/'STOP').exists():raise InterruptedError('Stopped during vertical initialization before accepting a physical step')
                            if batch['color']%10==0:
                                print(f'{method} L{level} step {step}: expansion color {batch["color"]+1}/{len(histories[level-1].colors)}',flush=True)
                        try:
                            initial[:embedding.n],initialization=initialize(
                                histories[level-1],tracks[level-1][step+1],target,obstacles=obstacles,
                                options=ExpansionOptions(seed=args.expansion_seed+step,gap_cap=args.expansion_gap_cap,
                                                         minimum_gap=args.expansion_min_gap,fixed_policy=args.expansion_fixed_policy,dt=dt),
                                material=dict(young=args.young,poisson=args.poisson,thickness=args.thickness,density=args.density,
                                              strain_activation=args.strain_activation,strain_stiffness=args.strain_stiffness,
                                              strain_model=args.strain_model,elastic_hessian=args.elastic_hessian,
                                              hinge_difference=args.hinge_difference),
                                dmin=args.dmin,contact_stiffness=args.contact_stiffness,strain_limit=args.strain_limit,
                                on_batch=refinement_progress)
                        except RefinementFailure as error:
                            (folder/f'initialization_failure_{step:05d}.json').write_text(json.dumps(error.report,indent=2),encoding='utf-8')
                            if hasattr(error,'failed_state'):
                                np.savez_compressed(folder/f'initialization_failure_{step:05d}.npz',
                                                    positions=error.failed_state,rest=error.failed_rest,
                                                    faces=error.failed_faces,cloth_vertices=error.failed_cloth_vertices)
                            raise
                        (folder/f'initialization_{step:05d}.json').write_text(json.dumps(initialization,indent=2),encoding='utf-8')
                    next_q,report=solve(problem,initial,options,fixed_mask)
                    if initialization is not None:
                        report['vertical_initialization']={'accepted':True,'seconds':initialization['seconds'],
                                                           'log':f'initialization_{step:05d}.json'}
                    report['step']=step
                    reports.append(report)
                    timings.append(time.perf_counter()-tick)
                    # A paper-authorized coarse preview stop is NOT reported as
                    # Newton convergence. Finest/direct states remain strict.
                    if not report['accepted']:
                        raise RuntimeError(f'{method} L{level} step {step}: {report["reason"]}, gradient={report["gradient_inf"]:.3g}')
                    meta['preview_stopped_steps']+=int(report['preview_stop'])
                    q=next_q
                    track[step+1]=q[:len(shell.rest)]
                    poses[step+1]=embedding.transforms(q)
                    metric={'step':step+1,'max_stretch':report['max_stretch']}
                    if level:
                        metric.update(consistency_metrics(tracks[level-1][step+1],track[step+1],hierarchy.adjacent[level-1],shells[level-1].mass,shell.mass))
                    if step:
                        _,gradient,_=problem.evaluate(q,transfer.offset(q[:len(shell.rest)]))
                        potential_gradient=gradient[:len(shell.rest)]-shell.mass[:,None]*(track[step+1]-xhat)/dt**2
                        physical_gradient=potential_gradient.copy()
                        if target is not None:
                            physical_gradient-=2*meta['consistency_weight']*shell.mass[:,None]*(track[step+1]-target)
                        metric.update(continuity(track[step-1],track[step],track[step+1],xhat,physical_gradient,
                                                  shell.mass,dt,potential_gradient,~pins(shell.rest)))
                    metrics.append(metric)
                    meta.update(completed_steps=step+1,physics_seconds=sum(timings))
                    if (step+1)%args.checkpoint_every==0 or step+1==steps:
                        print(f'{method} L{level} {step+1}/{steps}: {sum(timings):.2f}s, Newton {report["iterations"]}, stretch {report["max_stretch"]:.5f}',flush=True)
                        save_level()
                if contact: contact.validate(embedding.positions(q))
                meta['completed']=True
                if level==0 and method!='direct':
                    shared_coarse={'track':track,'poses':poses,'folder':folder.name,'seconds':sum(timings)}
            except Exception as error:
                meta['error']=repr(error)
                raise
            finally:
                meta['physics_seconds']=sum(timings)
                meta['timing_scope']='predictor, contact/friction setup, vertical initialization (including its log write), energy assembly and Newton solve; excludes metrics, other disk I/O, scene setup and rendering'
                save_level()
                config['runs'].append(dict(meta,folder=folder.name))
                save_config()
    config['completed']=args.only_level is None
    if args.only_level is not None:
        config['selected_level_completed']=True
        config['status']='selected_level_completed'
        config['scope']='Only L1 simulated; L0 inherited; L2-L7 not simulated'
    save_config()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=('flag','grip','shirt','intrinsic','intrinsic-grip','bunny'),default='flag')
    p.add_argument('--config',type=Path,help='JSON parameters used as defaults; explicit CLI arguments take precedence')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,help='Copy accepted prefixes from this experiment; output must still be new')
    p.add_argument('--resume-allow-code-change',action='store_true',help='Explicit diagnostic continuation across recorded code changes, not a benchmark')
    p.add_argument('--resume-interrupted',action='store_true',help='Recover a stopped process with no final error record; stop that source process first')
    p.add_argument('--only-level',type=int,choices=[1],help='Simulate only L1, retaining the full finest-energy hierarchy')
    p.add_argument('--coarse-from',type=Path,help='Completed L0 input for --only-level 1; does not reuse any L1 frames')
    p.add_argument('--threads',type=int,default=1,help='Fixed IPC and BLAS thread budget for reproducible reference runs')
    p.add_argument('--checkpoint-every',type=int,default=1,help='Save accepted states/logs every this many steps')
    p.add_argument('--assets',type=Path,default=Path('output/shirt_assets'))
    p.add_argument('--methods',nargs='+',choices=METHODS,default=['velpro','direct'])
    p.add_argument('--levels',type=int,default=2)
    p.add_argument('--base-segments',type=int,default=3)
    p.add_argument('--curvature',type=float,default=0.,help='Flag rest curvature amplitude in metres; exercises nonlinear PSQ offsets')
    p.add_argument('--warm-start',choices=('horizontal','vertical'),default='horizontal')
    p.add_argument('--expansion-seed',type=int,default=527)
    p.add_argument('--expansion-gap-cap',type=float,default=.001,help='Conservative gap bound for safe refinement; must exceed dmin')
    p.add_argument('--expansion-min-gap',type=float,help='PSQ Algorithm 3 epsilon, strictly above dmin; omitted preserves each measured pre-expansion gap')
    p.add_argument('--expansion-fixed-policy',choices=('unsplit-lineage','coarse-indices'),default='unsplit-lineage',
                   help='Fix surviving coarse vertex identities; coarse-indices reproduces the historical storage-slot policy')
    p.add_argument('--grip-jaw-height',type=float,default=.026,
                   help='Metres; tall plates isolate friction from support by jaw edges in the grip diagnostic')
    p.add_argument('--grip-initial-gap',type=float,default=.001,help='Initial gap between the two inner jaw faces in metres')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--dt',type=float,default=.01)
    p.add_argument('--frames',type=int,default=740)
    p.add_argument('--hold-after-frame',type=int,help='After reaching this original control frame, hold the physical jaws at that pose')
    p.add_argument('--substeps',type=int,default=2)
    p.add_argument('--settle-steps',type=int,default=60)
    p.add_argument('--young',type=float,default=30000.)
    p.add_argument('--poisson',type=float,default=.3)
    p.add_argument('--thickness',type=float,default=.0004)
    p.add_argument('--density',type=float,default=200.)
    p.add_argument('--strain-limit',type=float,default=1.1)
    p.add_argument('--strain-activation',type=float,help='Default: halfway between rest stretch and strain limit')
    p.add_argument('--strain-stiffness',type=float,default=1e3,help='Pa; C-IPC initial stiffness, adapted up to 1e5')
    p.add_argument('--strain-model',choices=('cipc','legacy'),default='cipc')
    p.add_argument('--elastic-hessian',choices=('exact','finite_difference'),default='exact')
    p.add_argument('--hinge-difference',choices=('principal','periodic'),default='principal',help='C-IPC signed-angle difference; periodic is a historical alternative')
    p.add_argument('--convergence-norm',choices=('rms','max'),default='rms',help='C-IPC RMS velocity decrement; max is the historical control')
    p.add_argument('--ccd',choices=('additive','tight'),default='additive')
    p.add_argument('--consistency-weight',type=float,default=0.)
    p.add_argument('--friction',type=float,default=1.)
    p.add_argument('--dhat',type=float,default=.0015)
    p.add_argument('--dmin',type=float,default=.0002)
    p.add_argument('--contact-stiffness',type=float,default=None,
                   help='Default: 1e5 for the thin layered shirt, 1e4 for the grip test')
    p.add_argument('--max-iterations',type=int,default=None)
    p.add_argument('--tolerance',type=float,default=1e-3)
    p.add_argument('--hessian',choices=('coarse','proxy'),default=None,
                   help='Default: coarse, following PSQ Sec. 4.5; proxy is an explicit ablation')
    p.add_argument('--offset-linesearch',choices=('nonlinear','frozen'),default='nonlinear',
                   help='Nonlinear evaluates the full PSQ potential (paper Algorithm 1); frozen searches only the local lagged-offset model')
    p.add_argument('--strict-preview',action='store_true',help='Require full Newton convergence even for coarse previews; default follows PD 2024 Sec. 4.6')
    p.add_argument('--preview-min-alpha',type=float,default=1e-8,help='PSQ Algorithm 1 gamma: explicitly logged small-step stop for coarse previews only')
    config_parser=argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config',type=Path)
    supplied,_=config_parser.parse_known_args(argv)
    if supplied.config:
        defaults=json.loads(supplied.config.read_text(encoding='utf-8'))['parameters']
        allowed={a.dest for a in p._actions}-{'help','output','config','resume_from','resume_allow_code_change','resume_interrupted'}
        if set(defaults)-allowed:p.error(f'Unknown or run-specific config parameters: {sorted(set(defaults)-allowed)}')
        # argparse converts string defaults through the declared type, e.g. Path.
        p.set_defaults(**defaults)
    args=p.parse_args(argv)
    if (args.only_level is None)!=(args.coarse_from is None):
        p.error('--only-level 1 and --coarse-from must be supplied together')
    if args.only_level is not None and (args.levels<3 or len(args.methods)!=1 or args.methods[0]=='direct' or args.resume_from or args.warm_start!='horizontal'):
        p.error('L1-only requires >=3 hierarchy levels, one progressive method, horizontal stepping, and no resume')
    for action in p._actions:
        value=getattr(args,action.dest,None)
        if action.choices and value is not None:
            values=value if isinstance(value,list) else [value]
            if any(v not in action.choices for v in values):p.error(f'Invalid configured {action.dest}: {value}')
    if args.strain_activation is None:args.strain_activation=1+(args.strain_limit-1)*.5
    if args.max_iterations is None:args.max_iterations=200 if args.scene=='shirt' else 100
    if args.hessian is None: args.hessian='coarse'
    if args.contact_stiffness is None: args.contact_stiffness=1e5 if args.scene=='shirt' else 1e4
    if args.levels<1 or args.steps<2 or args.dt<=0 or args.substeps<1 or args.settle_steps<0 or not 1<=args.frames<=4373:
        p.error('Invalid time grid or hierarchy')
    if args.consistency_weight<0 or args.strain_limit<=1: p.error('Invalid consistency or strain parameters')
    if not 1<=args.strain_activation<args.strain_limit or args.strain_stiffness<=0:p.error('Invalid strain barrier configuration')
    if not 0<args.preview_min_alpha<1:p.error('Invalid preview step threshold')
    if args.expansion_gap_cap<=args.dmin or args.expansion_seed<0:p.error('Invalid expansion settings')
    if args.expansion_min_gap is not None and not args.dmin<args.expansion_min_gap<=args.expansion_gap_cap:
        p.error('Expansion minimum gap must be above dmin and at most gap-cap')
    if args.threads<1 or args.checkpoint_every<1:p.error('Thread and checkpoint intervals must be positive')
    if args.hold_after_frame is not None and (args.scene!='shirt' or not 0<=args.hold_after_frame<args.frames):
        p.error('Hold frame must lie inside the shirt control interval')
    if args.grip_jaw_height<=0 or args.grip_initial_gap<=2*args.dmin:
        p.error('Grip geometry must have positive height and initial clearance greater than twice dmin')
    with numerical_threads(args.threads) as runtime:
        run(args,runtime)


if __name__=='__main__': main()
