"""Replay an initializer and attribute inherited-edge distortion to stages."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from progressive.run import prepare_scene
from progressive.affine import AffineObstacles
from progressive.contact import IPCContact
from progressive.expansion_history import ExpansionHistory
from progressive.refinement_contact import ExpansionOptions,RefinementFailure,edges_of,minimum_gap
from progressive.shell import Shell
from progressive.runtime import numerical_threads
import progressive.safe_expansion as expansion


def diagnose(experiment,step,output,fixed_policy='coarse-indices'):
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing nonempty output')
    config=json.loads((experiment/'experiment.json').read_text())
    parameters=config['parameters'].copy();parameters['assets']=Path(parameters['assets'])
    args=SimpleNamespace(**parameters)
    hierarchy,bodies,fixed,_,_,_,dt,_=prepare_scene(args)
    if len(hierarchy.rests)!=2:raise ValueError('Two-level diagnostic only')
    h=ExpansionHistory.load(args.assets/'expansion_0_1.json',hierarchy.rests[0],hierarchy.faces[0],hierarchy.rests[1],hierarchy.faces[1])
    coarse=np.load(experiment/'velpro_L0/positions.npy')[step+1]
    poses=np.load(experiment/'velpro_L0/robot.npy')[step+1]
    embedding=AffineObstacles(h.fine_rest,h.fine_faces,bodies,fixed);q=embedding.q0.copy()
    for i,t in enumerate(poses):q[embedding.n+4*i:embedding.n+4*i+4]=np.vstack((t[:3,3],t[:3,:3].T))
    world=embedding.positions(q);of=embedding.faces[len(h.fine_faces):]-h.n
    obstacles=[(world[h.n:],of)] if len(of) else []
    target=hierarchy.adjacent[0].position(coarse)
    material={k:parameters[k] for k in ('young','thickness','density','strain_activation','strain_stiffness','strain_model','elastic_hessian','hinge_difference')}
    shell=Shell(h.fine_rest,h.fine_faces,strain_limit=args.strain_limit,**material)
    contact=IPCContact(embedding.positions(embedding.q0),embedding.faces,h.n,args.dhat,args.dmin,args.contact_stiffness,args.friction)
    target_world=np.vstack((target,world[h.n:]));target_error=None
    try:contact.validate(target_world)
    except ValueError as error:target_error=str(error)
    edges=edges_of(h.fine_faces);inherited=np.zeros(h.n,bool);inherited[h.coarse_ids]=True
    edges=edges[inherited[edges].all(axis=1)]
    lengths=np.linalg.norm(h.fine_rest[edges[:,1]]-h.fine_rest[edges[:,0]],axis=1)
    report={'experiment':str(experiment.resolve()),'step':step,'target_stretch':shell.max_stretch(target),
            'diagnostic_fixed_policy':fixed_policy,
            'target_contact_error':target_error,'target_gap_lower_bound':minimum_gap(contact.mesh,target_world,.001),
            'target_feasible':shell.feasible(target) and target_error is None,'stages':[],
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('progressive').glob('*.py')}}
    snapshots={};current={};original_compact=expansion.compact_world
    def observe(label,positions):
        full=np.full((h.n,3),np.nan);full[current['ids']]=positions[:len(current['ids'])]
        ratios=np.linalg.norm(full[edges[:,1]]-full[edges[:,0]],axis=1)/lengths
        report['stages'].append({'stage':label,'vertices':len(current['ids']),
                                'inherited_edge_max_ratio':float(ratios.max()),
                                'bad_inherited_edges':edges[ratios>=args.strain_limit].tolist()})
        snapshots[f'stage_{len(report["stages"])-1:03d}']=full
    def compact(*a):
        result=original_compact(*a);current['ids']=result[0];return result
    original_ratio=expansion.ratio_advance;original_relax=expansion.patch_relaxation
    original_strain=expansion.strain_relaxation
    def ratio(mesh,x,*a):
        observe('after_perturb',x);out,records=original_ratio(mesh,x,*a);observe('after_ratio_advance',out);return out,records
    def relax(shell,x,*a,**kw):
        out,records=original_relax(shell,x,*a,**kw);observe('after_patch_relaxation',out);return out,records
    def strain(shell,x,target,mask,*a):
        if fixed_policy=='unsplit-lineage':
            mask=mask.copy()
            # A split retires its parent; both endpoints are born in refinement,
            # even when one endpoint reuses the parent's storage index.
            for p in h.patches:mask[p.split]=False
        report['strain_fixed_cloth_ids']=np.flatnonzero(mask[:h.n]).tolist()
        out,records=original_strain(shell,x,target,mask,*a)
        observe('after_strain_correction',out)
        return out,records
    output.mkdir(parents=True,exist_ok=True)
    try:
        with patch.object(expansion,'compact_world',compact),patch.object(expansion,'ratio_advance',ratio),patch.object(expansion,'patch_relaxation',relax),patch.object(expansion,'strain_relaxation',strain):
            result,init=expansion.initialize(h,coarse,target,obstacles=obstacles,
                options=ExpansionOptions(seed=args.expansion_seed+step,gap_cap=args.expansion_gap_cap,
                                         minimum_gap=args.expansion_min_gap,fixed_policy=fixed_policy,dt=dt),
                material=material,dmin=args.dmin,contact_stiffness=args.contact_stiffness,strain_limit=args.strain_limit)
        report['accepted']=True;report['initialization']=init;snapshots['result']=result
    except RefinementFailure as error:
        report['accepted']=False;report['initialization']=error.report
    np.savez_compressed(output/'stages.npz',target=target,rest=h.fine_rest,faces=h.fine_faces,
                        inherited_edges=edges,**snapshots)
    (output/'diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('target feasible:',report['target_feasible'],'target max stretch:',report['target_stretch'])
    for s in report['stages']:print(s)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-policy',choices=('coarse-indices','unsplit-lineage'),default='coarse-indices')
    a=p.parse_args()
    with numerical_threads(1):diagnose(a.experiment,a.step,a.output,a.fixed_policy)
