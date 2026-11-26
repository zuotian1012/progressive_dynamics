"""Bounded snapshot reconstruction for solver diagnosis, NOT exact checkpoint replay."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,default=Path('output/shirt_L2_feasibility_fix_20261118/L2/run'))
p.add_argument('--runtime',type=Path,default=Path('output/shirt_L2_feasibility_fix_20261118/runtime/src'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--step',type=int,default=2936)
p.add_argument('--variant',choices=['baseline','diag','tight','no_robot_contact','no_robot_contact_diag','no_graph','no_robot_contact_native_pz','native_pz'],default='baseline')
p.add_argument('--newton-limit',type=int,default=4)
p.add_argument('--dump-mas',action='store_true')
p.add_argument('--steps',type=int,default=1,help='Bounded continuation; stops on a nonconverged or line-search-exhausted step')
p.add_argument('--no-matrix-dump',action='store_true')
p.add_argument('--equilibrate',action='store_true',help='Experimental symmetric variable scaling; requires a supporting runtime')
p.add_argument('--checkpoint-every',type=int,default=0,help='Save accepted states periodically; native caches are not checkpointed')
p.add_argument('--initial-state',type=Path,help='Accepted state saved by this probe; source run and absolute step must match')
a=p.parse_args()
if a.equilibrate:os.environ['UIPC_LINEAR_EQUILIBRATE']='1'
else:os.environ.pop('UIPC_LINEAR_EQUILIBRATE',None)
if a.output.exists() and any(a.output.iterdir()):raise ValueError('Output must be empty')
a.output.mkdir(parents=True,exist_ok=True)
from scripts.libuipc_native_runtime import load,loaded_libraries
uipc=load(a.runtime)
import numpy as np
import trimesh as tm
from uipc import view,builtin
from uipc.core import Engine,Scene,World,FiniteElementStateAccessorFeature,AffineBodyStateAccessorFeature
from uipc.constitution import AffineBodyConstitution,SoftTransformConstraint,NeoHookeanShell,DiscreteShellBending,ElasticModuli2D
from uipc.geometry import label_surface
from shirt_demo.simulate import trimesh
from shirt_demo.assets import Robot,LINKS
meta=json.loads((a.run/'run.json').read_text());cfg=copy.deepcopy(meta['config'])
cfg['newton']['max_iter']=a.newton_limit
cfg['extras']['debug']['dump_linear_system']=0 if a.no_matrix_dump else 1
if a.dump_mas:cfg['extras']['debug']['dump_mas_matrices']=1
if a.variant in ['diag','no_robot_contact_diag']:cfg['linear_system']['fem_preconditioner']='diag'
if a.variant=='tight':cfg['linear_system']['tol_rate']=1e-8
if a.variant=='no_graph':cfg['linear_system']['use_cuda_graph']=0
if a.variant in ['no_robot_contact_native_pz','native_pz']:
    # Only inspect the INITIAL native preconditioner application; no convergence claim.
    cfg['linear_system']['solver']='linear_pcg'
    cfg['linear_system']['tol_rate']=.9
    cfg['extras']['debug']['dump_linear_pcg']=1
os.environ['UIPC_PSQ_BUNDLE']=meta['psq_bundle']
os.environ['UIPC_PSQ_MODE']='proxy'
os.environ['UIPC_PSQ_STRAIN_CAP']='directional'
os.environ['UIPC_PSQ_REJECT_INF_TRIAL']='1'
os.environ['UIPC_PSQ_CAP_AUDIT']=str((a.output/'cap_audit.jsonl').resolve())
for key in ['UIPC_PSQ_PREVIEW_STOP','UIPC_PSQ_FORCE_UPDATE','UIPC_PSQ_CONSISTENCY_AUDIT']:
    os.environ.pop(key,None)
uipc.Logger.set_level(uipc.Logger.Level.Info);uipc.Timer.disable_all()
saved=np.load(a.run/'positions.npy',mmap_mode='r');f=np.load(a.run/'faces.npy')
poses=np.load(a.run/'robot.npy',mmap_mode='r')
start_x=np.array(saved[a.step]);start_robot=np.array(poses[a.step])
if a.initial_state:
    state=np.load(a.initial_state)
    assert int(state['step'])==a.step and str(state['source_run'])==str(a.run.resolve())
    assert state['positions'].shape==start_x.shape and state['robot'].shape==start_robot.shape
    start_x=np.array(state['positions']);start_robot=np.array(state['robot'])
velocity=np.load(a.run/'input_velocities.npy',mmap_mode='r')[a.step]
robot=Robot(meta['trajectory_path'])
step=a.step;settle=meta['parameters']['settle_steps'];sub=meta['parameters']['substeps']
frame,s=divmod(step-settle,sub);previous=robot.q[max(0,frame-1)]
q=previous+(s+1)/sub*(robot.q[frame]-previous)
targets=robot.poses(q)
engine=Engine('cuda',str((a.output/'backend').resolve()));world=World(engine);scene=Scene(cfg)
ct=scene.contact_tabular();ct.default_model(1.,1e7)
cloth_ct=ct.create('cloth');robot_ct=ct.create('robot');table_ct=ct.default_element()
ct.insert(cloth_ct,cloth_ct,2.,1e7,True);ct.insert(cloth_ct,table_ct,1.,1e7,True)
ct.insert(cloth_ct,robot_ct,float(np.sqrt(2)),1e7,not a.variant.startswith('no_robot_contact'))
ct.insert(robot_ct,table_ct,float(np.sqrt(2)),1e7,True);ct.insert(robot_ct,robot_ct,0.,1e7,False)
cloth=trimesh(np.array(saved[0]),f);label_surface(cloth)
NeoHookeanShell().apply_to(cloth,ElasticModuli2D.youngs_poisson(meta['parameters']['young'],.3),200.,meta['material_parameters']['full_thickness']/2)
DiscreteShellBending().apply_to(cloth,meta['parameters']['young'],.3);cloth_ct.apply_to(cloth)
scene.objects().create('shirt').geometries().create(cloth)
abd=AffineBodyConstitution();drive=SoftTransformConstraint()
box=tm.creation.box(extents=[1,2,.8]);table=trimesh(box.vertices+[.65,0,.4],box.faces)
label_surface(table);abd.apply_to(table,1e8,1000)
view(table.instances().find(builtin.is_fixed))[:]=1;table_ct.apply_to(table)
scene.objects().create('table').geometries().create(table)
slots=[]
for j,name in enumerate(LINKS):
    m=np.load(a.run/f'collider_{j}.npz');geo=trimesh(m['vertices'],m['faces']);label_surface(geo)
    abd.apply_to(geo,1e8,1000);view(geo.instances().find(builtin.is_dynamic))[:]=0
    view(geo.transforms())[0]=robot.poses(robot.q[0])[name]
    drive.apply_to(geo,np.array(meta['robot_drive_strength']))
    view(geo.instances().find(builtin.is_constrained))[:]=1;robot_ct.apply_to(geo)
    obj=scene.objects().create(name);slot,_=obj.geometries().create(geo);slots.append(slot)
    def animate(info,name=name):view(info.geo_slots()[0].geometry().instances().find(builtin.aim_transform))[0]=targets[name]
    scene.animator().insert(obj,animate)
world.init(scene)
assert world.is_valid()
fem=world.features().find(FiniteElementStateAccessorFeature);fs=fem.create_geometry()
fs.vertices().create(builtin.position,np.zeros(3));fs.vertices().create(builtin.velocity,np.zeros(3));fem.copy_to(fs)
view(fs.positions())[:]=start_x.reshape(-1,3,1)
view(fs.vertices().find(builtin.velocity))[:]=velocity.reshape(-1,3,1);fem.copy_from(fs)
af=world.features().find(AffineBodyStateAccessorFeature);ast=af.create_geometry()
ast.instances().create(builtin.transform,np.eye(4));af.copy_to(ast)
mapping={}
for j,slot in enumerate(slots):
    offset=int(view(slot.geometry().meta().find('backend_abd_body_offset'))[0]);mapping[LINKS[j]]=offset
    view(ast.transforms())[offset]=start_robot[j]
af.copy_from(ast);af.copy_to(ast);fem.copy_to(fs)
assert np.array_equal(view(fs.positions()).reshape(-1,3),start_x)
assert np.array_equal(view(fs.vertices().find(builtin.velocity)).reshape(-1,3),velocity)
assert np.array_equal(np.stack([view(ast.transforms())[mapping[n]] for n in LINKS]),start_robot)
world.retrieve()
info=dict(source=str(a.run.resolve()),snapshot_step=a.step,target_step=a.step+1,variant=a.variant,config=cfg,
    mapping=mapping,runtime_hashes={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in loaded_libraries().items()},
    body_dof_offsets={n:int(view(slot.geometry().meta().find(builtin.dof_offset))[0]) for n,slot in zip(LINKS,slots)},
    exact_position_velocity_transform_readback=True,
    limitation='Reconstructed scene with original rest geometry and saved position, velocity, ABD transforms. Native adaptive/contact/friction history and PSQ offset caches are rebuilt, not restored; this is not an exact historical checkpoint replay.',
    newton_limit_is_diagnostic=True,linear_equilibration=a.equilibrate,
    initial_state=str(a.initial_state.resolve()) if a.initial_state else None,
    initial_state_sha256=hashlib.sha256(a.initial_state.read_bytes()).hexdigest() if a.initial_state else None)
(a.output/'probe.json').write_text(json.dumps(info,indent=2))
assert a.steps>0 and a.step+a.steps<=len(robot.q)*sub+settle
info['requested_probe_steps']=a.steps
history=[];all_x=[start_x.copy()];all_robot=[start_robot.copy()]
velocity_only=fem.create_geometry();velocity_only.vertices().create(builtin.velocity,np.zeros(3))
lower=np.load(Path(meta['lower_run'])/'positions.npy',mmap_mode='r')
from scipy.sparse import load_npz
transfer=load_npz(Path(meta['parameters']['assets'])/f'map_{meta["level"]-1}_{meta["level"]}.npz')
t=time.perf_counter()
for local in range(a.steps):
    physical=a.step+local
    frame,s=divmod(physical-settle,sub);previous=robot.q[max(0,frame-1)]
    targets=robot.poses(previous+(s+1)/sub*(robot.q[frame]-previous))
    vv=transfer@((lower[physical]-lower[physical-1])/meta['dt'])
    if local==0:np.testing.assert_allclose(vv,velocity,atol=1e-12,rtol=0)
    view(velocity_only.vertices().find(builtin.velocity))[:]=vv.reshape(-1,3,1);fem.copy_from(velocity_only)
    tick=time.perf_counter();world.advance();world.retrieve();fem.copy_to(fs);af.copy_to(ast)
    stats=engine.frame_stats();rr=np.stack([view(ast.transforms())[mapping[n]] for n in LINKS])
    all_x.append(view(fs.positions()).reshape(-1,3).copy());all_robot.append(rr)
    history.append(dict(physical_step=physical+1,stats=stats,seconds=time.perf_counter()-tick,
        max_robot_origin_target_error_mm=float(np.linalg.norm(rr[:,:3,3]-np.stack([targets[n][:3,3] for n in LINKS]),axis=1).max()*1000)))
    (a.output/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(history[-1]),flush=True)
    if not world.is_valid() or not stats['converged'] or stats['hit_line_search_limit']:break
    if a.checkpoint_every>0 and ((local+1)%a.checkpoint_every==0 or local+1==a.steps):
        np.savez_compressed(a.output/f'checkpoint_{physical+1}.npz',step=physical+1,
            source_run=str(a.run.resolve()),positions=all_x[-1],robot=rr)
info['wall_seconds']=time.perf_counter()-t;info['stats']=engine.frame_stats();info['valid']=world.is_valid()
info['completed_probe_steps']=len(history)
info['all_probe_steps_converged']=len(history)==a.steps and all(h['stats']['converged'] and not h['stats']['hit_line_search_limit'] for h in history)
np.save(a.output/'probe_positions.npy',np.array(all_x));np.save(a.output/'probe_robot.npy',np.array(all_robot))
after=view(fs.positions()).reshape(-1,3).copy();after_robot=np.stack([view(ast.transforms())[mapping[n]] for n in LINKS])
np.savez_compressed(a.output/'states.npz',before=start_x,after=after,robot_before=start_robot,robot_after=after_robot,
    targets=np.stack([targets[n] for n in LINKS]))
info['cloth_max_displacement_mm']=float(np.linalg.norm(after-start_x,axis=1).max()*1000)
info['robot_origin_displacement_mm']=(np.linalg.norm(after_robot[:,:3,3]-start_robot[:,:3,3],axis=1)*1000).tolist()
(a.output/'probe.json').write_text(json.dumps(info,indent=2));print(json.dumps(info['stats']),flush=True)
