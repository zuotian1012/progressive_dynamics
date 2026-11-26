"""Author intrinsic self-parameterization -> PSQ-ready, coarse-to-fine assets.

The original finest shirt topology is retained. No nearest-point map is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.sparse import coo_matrix,save_npz
from .prolongation import Hierarchy
from .expansion_history import ExpansionHistory

DEFAULT_EXECUTABLE = Path(__file__).resolve().parents[1] / 'build/intrinsic/Release/intrinsic_builder.exe'

def prepare(output,targets,input_path=None,executable=DEFAULT_EXECUTABLE,
            quadric='probabilistic',normal_stddev=.01,position_stddev_relative=.01,contact_clearance=.0002,
            scene527_transform=False,lock_boundary=False):
    from shirt_demo.assets import read_obj,BUNDLE
    if contact_clearance<=0:raise ValueError('Contact clearance must be positive')
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing nonempty output')
    if not executable.exists():raise RuntimeError('Build native/intrinsic_builder first')
    output.mkdir(parents=True,exist_ok=True)
    source=input_path or BUNDLE/'assets/cloth/short-shirt-8000f.obj'
    rest,faces=read_obj(source)
    transformed = input_path is None or scene527_transform
    if transformed:
        rest=rest[:,[0,2,1]]*[.001,.001,-.001]+[.667,.015,.93]
    intermediate=output/'input.obj'
    with intermediate.open('w',encoding='ascii') as stream:
        for v in rest:stream.write('v '+' '.join(f'{a:.17g}' for a in v)+'\n')
        for f in faces:stream.write('f '+' '.join(str(int(i)+1) for i in f)+'\n')
    start=time.perf_counter()
    with (output/'preprocess.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(executable.resolve()),str(intermediate.resolve()),str((output/'raw').resolve()),
                        *map(str,targets),'--quadric',quadric,'--normal-stddev',str(normal_stddev),
                        '--position-stddev-relative',str(position_stddev_relative),
                        *(['--lock-boundary'] if lock_boundary else [])],stdout=log,stderr=subprocess.STDOUT,check=True)
    seconds=time.perf_counter()-start
    count=len(targets)+1
    rests=[];topologies=[];maps=[];adjacent=[];direct=[]
    for k,d in enumerate(reversed(range(count))):
        x,f=read_obj(output/'raw'/f'mesh_{d}.obj')
        if d == 0:
            if not np.array_equal(f,faces) or not np.allclose(x,rest,atol=1e-14,rtol=0):
                raise ValueError('Finest source topology/geometry changed')
            # Native text serialization can round by an ULP; retain exact input.
            x=rest.copy()
        rests.append(x);topologies.append(f)
        np.savez_compressed(output/f'mesh_{k}.npz',vertices=x,faces=f)
        if d:
            path=output/'raw'/f'map_{d}.txt'
            with path.open() as stream:shape=tuple(map(int,stream.readline().split()[:2]))
            data=np.loadtxt(path,skiprows=1,ndmin=2)
            U=coo_matrix((data[:,2],(data[:,0].astype(int),data[:,1].astype(int))),shape=shape).tocsr()
            U.eliminate_zeros();maps.append(U)
            save_npz(output/f'map_{k}_{k+1}.npz',U)
            anchor=np.loadtxt(output/'raw'/f'anchors_{d}.txt',dtype=int,ndmin=1)
            finest=np.loadtxt(output/'raw'/f'finest_anchors_{d}.txt',dtype=int,ndmin=1)
            adjacent.append(anchor);direct.append(finest)
            np.save(output/f'anchors_{k}_{k+1}.npy',anchor)
            np.save(output/f'anchors_{k}_{count-1}.npy',finest)
        else:direct.append(None)
    h=Hierarchy(rests,topologies,maps,adjacent,direct)
    expansion_checks=[]
    for k in range(count-1):
        source_history=output/'raw'/f'expansion_{count-1-k}.json'
        history=ExpansionHistory.load(source_history,rests[k],topologies[k],rests[k+1],topologies[k+1])
        saved_history=output/f'expansion_{k}_{k+1}.json'
        saved_history.write_bytes(source_history.read_bytes())
        expansion_checks.append(dict(history.summary(),file=saved_history.name,
                                     sha256=hashlib.sha256(saved_history.read_bytes()).hexdigest()))
    if not np.array_equal(topologies[-1],faces) or not np.allclose(rests[-1],rest,atol=1e-14,rtol=0):
        raise ValueError('Finest source topology/geometry changed')
    checks=[]
    import ipctk
    for k,p in enumerate(h.to_finest):
        error=float(np.max(np.abs(p.position(rests[k])-rests[-1])))
        if error>1e-12:raise ValueError('PSQ rest reconstruction failed')
        f=topologies[k]
        edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
        collision_mesh=ipctk.CollisionMesh(rests[k],edges.astype(np.int32),f.astype(np.int32))
        intersections=bool(ipctk.has_intersections(collision_mesh,rests[k]))
        collisions=ipctk.NormalCollisions();collisions.build(collision_mesh,rests[k],contact_clearance)
        distance=None if collisions.empty() else float(np.sqrt(collisions.compute_minimum_distance(collision_mesh,rests[k])))
        checks.append({'level':k,'vertices':len(rests[k]),'faces':len(f),'rest_max_error_m':error,
                       'nonzero_psq_offset':not p.zero_offset,'rest_self_intersections':intersections,
                       'contact_clearance_m':contact_clearance,'minimum_distance_below_clearance_m':distance,
                       'rest_thickness_feasible':not intersections and (distance is None or distance>contact_clearance),
                       'max_offset_m':float(np.linalg.norm(p.offset(rests[k]),axis=1).max())})
    manifest={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'coordinate_transform':{'input_units':'mm','output_units':'m','axis_order':[0,2,1],
                                      'axis_scale':[.001,.001,-.001],'translation_m':[.667,.015,.93]} if transformed else {'kind':'identity'},
              'method':quadric+' plane quadrics + joint successive self-parameterization (Liu et al. 2021), PSQ nonlinear offset (Zhang et al. 2023)',
              'quadric_parameters':json.loads((output/'raw'/'metric.json').read_text()),
              'quadric_parameter_scope':'Explicit local settings; PSQ author noise parameters were not published in the accessed paper',
              'executable_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
              'quadric_reference_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'third_party/probabilistic_quadrics/probabilistic-quadrics.hh').read_bytes()).hexdigest(),
              'source_commit':'adc49abe10a97200a3a5735bd370e98560091a37',
              'original_finest_topology_preserved':True,'nearest_point_mapping':False,
              'targets_fine_to_coarse':targets,'construction_seconds':seconds,'levels':checks,
              'expansion_histories':expansion_checks,
              'nonintersecting_rest_hierarchy':not any(c['rest_self_intersections'] for c in checks),
              'thickness_feasible_rest_hierarchy':all(c['rest_thickness_feasible'] for c in checks),
              'limitation':'Exports reversible histories for CPU safe expansion; static hierarchy feasibility is checked, not automatically repaired'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--targets',type=int,nargs='+',default=[4000,1000]);p.add_argument('--input',type=Path)
    p.add_argument('--executable',type=Path,default=DEFAULT_EXECUTABLE)
    p.add_argument('--quadric',choices=('ordinary','probabilistic'),default='probabilistic')
    p.add_argument('--normal-stddev',type=float,default=.01)
    p.add_argument('--position-stddev-relative',type=float,default=.01)
    p.add_argument('--contact-clearance',type=float,default=.0002,help='Metres; test finite-thickness self clearance, not only intersections')
    p.add_argument('--scene527-transform',action='store_true',help='Apply the original shirt millimetre-to-world transform to an explicit input')
    p.add_argument('--lock-boundary',action='store_true',help='Preserve every boundary vertex; requires a supporting native builder')
    a=p.parse_args();prepare(a.output,a.targets,a.input,a.executable,a.quadric,a.normal_stddev,a.position_stddev_relative,a.contact_clearance,a.scene527_transform,a.lock_boundary)
