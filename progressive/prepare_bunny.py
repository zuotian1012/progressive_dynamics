"""Reconstruct PD++ Fig. 11--12 inputs with explicit provenance and unknowns.

The bundled C-IPC bunny is a substitute mesh, not an authenticated PD++ asset.
Debug meshes test the pipeline; only the paper-size preset matches the stated
endpoint vertex counts approximately. No simulation is launched by this module.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from cloth.mesh import grid
from shirt_demo.assets import read_obj
from .prepare_intrinsic import prepare
from .runtime import numerical_threads


SOURCE=Path('external/Codim-IPC-main/Projects/FEMShell/input/bunny3K.obj')
PAPER='https://pcs-sim.github.io/pd%2B%2B/PD%2B%2B_main.pdf'


def transformed_bunny(source,height):
    vertices,faces=read_obj(source)
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Bunny input must be a closed, consistently oriented triangle mesh')
    # Source is Y-up. This proper rotation maps +Y to world +Z.
    rotation=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    vertices=vertices @ rotation.T
    scale=height/np.ptp(vertices[:,2])
    center=(vertices.min(axis=0)+vertices.max(axis=0))/2
    translation=np.array([-center[0],-center[1],-vertices[:,2].min()])*scale
    vertices=vertices*scale+translation
    return vertices,faces,{'rotation':rotation.tolist(),'uniform_scale':float(scale),'translation_m':translation.tolist()}


def prepare_scene(output,profile='debug',source=SOURCE,segments=None,targets=None,
                  height=.5,width=1.,clearance=.1,offset=(.18,0.)):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing nonempty bunny asset directory')
    if min(height,width,clearance)<=0:raise ValueError('Geometry dimensions and clearance must be positive')
    if profile not in ('debug','paper-size'):raise ValueError('Unknown bunny profile')
    if segments is None:segments=24 if profile=='debug' else 299
    if targets is None:
        # Targets descend from the original finest cloth. All eight actual
        # meshes, mappings and anchor faces come from intrinsic preprocessing.
        targets=[800,560,400,280,200,140,100] if profile=='debug' else [100000,56000,31000,17000,9500,5200,2800]
    bunny,bunny_faces,transform=transformed_bunny(source,height)
    cloth=grid(segments)
    positions=np.column_stack(((cloth.uv-.5)*width+np.asarray(offset),
                               np.full(len(cloth.uv),bunny[:,2].max()+clearance)))
    # A separate source file is needed because prepare() requires empty output.
    output.parent.mkdir(parents=True,exist_ok=True)
    cloth_source=output.parent/(output.name+'_cloth_input.obj')
    if cloth_source.exists():raise ValueError('Refusing to replace existing source mesh')
    with cloth_source.open('w',encoding='ascii') as stream:
        for vertex in positions:stream.write('v '+' '.join(f'{a:.17g}' for a in vertex)+'\n')
        for face in cloth.faces:stream.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    prepare(output,targets,cloth_source,contact_clearance=.00007)
    np.savez_compressed(output/'bunny.npz',vertices=bunny,faces=bunny_faces)
    manifest_path=output/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['benchmark']={
        'name':'PD++ Cloth Drop Over Bunny, Figures 11--12', 'paper':PAPER,
        'profile':profile,'status':'reconstructed inputs; original author scene package not obtained',
        'paper_parameters':{'thickness_m':.00007,'max_principal_stretch':1.068,'friction':.3,
                            'dt_s':.01,'levels':8,'coarse_vertices_approx':1500,'fine_vertices_approx':90000},
        'source_search':{'project_page':'https://pcs-sim.github.io/pd%2B%2B/',
                         'author_page':'https://eriszhang.github.io/',
                         'website_repository':'https://github.com/pcs-sim/pcs-sim.github.io',
                         'finding':'No original bunny simulation configuration/mesh download identified in the reviewed public links'},
        'unresolved_author_inputs':['exact bunny mesh and transform','exact cloth mesh and eight-level hierarchy',
                                    'cloth dimensions and initial placement/velocity','Young modulus, Poisson ratio, density',
                                    'contact activation distance and stiffness','exact thickness offset convention',
                                    'simulation duration and frame-to-time convention'],
        'local_geometry':{'cloth_width_m':width,'bunny_height_m':height,'cloth_offset_xy_m':list(offset),
                          'cloth_clearance_above_bunny_m':clearance,'cloth_initial_velocity':'zero',
                          'cloth_pins':False,'ground_plane':False,
                          'scope':'explicit reconstruction choices, not recovered author settings'},
        'bunny':{'source':str(source),'source_url':'https://github.com/ipc-sim/Codim-IPC/blob/main/Projects/FEMShell/input/bunny3K.obj',
                 'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'vertices':len(bunny),'faces':len(bunny_faces),
                 'transform':transform,'scope':'unaltered source topology, proper rotation/uniform scale/translation only'},
        'not_a_paper_reproduction_certificate':True}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'assets':str(output),'profile':profile,'vertices':[r['vertices'] for r in manifest['levels']],
                      'bunny_vertices':len(bunny),'unresolved_author_inputs':manifest['benchmark']['unresolved_author_inputs']},indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=('debug','paper-size'),default='debug')
    p.add_argument('--bunny',type=Path,default=SOURCE)
    p.add_argument('--segments',type=int)
    p.add_argument('--targets',type=int,nargs='+')
    p.add_argument('--bunny-height',type=float,default=.5)
    p.add_argument('--cloth-width',type=float,default=1.)
    p.add_argument('--clearance',type=float,default=.1)
    p.add_argument('--offset',type=float,nargs=2,default=(.18,0.))
    a=p.parse_args()
    with numerical_threads(1):
        prepare_scene(a.output,a.profile,a.bunny,a.segments,a.targets,a.bunny_height,a.cloth_width,a.clearance,a.offset)


if __name__=='__main__':main()
