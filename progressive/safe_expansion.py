"""PSQ safe vertical initialization on exported intrinsic collapse histories.

CPU reference implementation. Collision pairs and local solves are processed
serially inside topology-independent color batches; no GPU speedup is claimed.
"""
import time
import numpy as np
import ipctk
from .shell import Shell
from .contact import IPCContact
from .refinement_contact import (ExpansionOptions,RefinementFailure,collision_mesh,
                                 minimum_gap,full_rank,safe_perturb,ratio_advance)
from .refinement_relaxation import patch_relaxation,strain_relaxation


def compact_world(rest,x,face_dict,obstacles):
    faces=np.array(list(face_dict.values()),int);ids=np.unique(faces)
    lookup=np.full(len(x),-1,int);lookup[ids]=np.arange(len(ids))
    cloth_faces=lookup[faces];world_x=[x[ids]];world_rest=[rest[ids]];world_faces=[cloth_faces]
    count=len(ids)
    for positions,triangles in obstacles:
        positions=np.asarray(positions,float);triangles=np.asarray(triangles,int)
        world_x.append(positions);world_rest.append(positions);world_faces.append(triangles+count);count+=len(positions)
    return ids,lookup,cloth_faces,np.vstack(world_rest),np.vstack(world_x),np.vstack(world_faces)


def initialize(history,coarse_x,prolonged_target,*,obstacles=(),options=None,
               material=None,dmin=.0002,contact_stiffness=1e5,strain_limit=1.1,on_batch=None):
    options=options or ExpansionOptions();settings=options.validate();started=time.perf_counter()
    material=dict(material or {});material.pop('strain_limit',None)
    if dmin<0 or options.gap_cap<=dmin:raise ValueError('Gap cap must exceed the requested final thickness')
    if options.minimum_gap is not None and options.minimum_gap<=dmin:
        raise ValueError('Minimum expansion gap must strictly exceed the requested final thickness')
    prolonged_target=np.asarray(prolonged_target,float)
    if prolonged_target.shape!=history.fine_rest.shape or not np.isfinite(prolonged_target).all():raise ValueError('Invalid prolonged target')
    rng=np.random.default_rng(options.seed);x,rest,faces=history.initial(coarse_x)
    expanded=set();finest=Shell(history.fine_rest,history.fine_faces,strain_limit=None,**material)
    report={'accepted':False,'history':history.summary(),'options':settings,'batches':[],
            'relaxation_energy':'shared finest elastic proxy through dynamic intrinsic PSQ map + local incremental/contact terms',
            'backend':'serial CPU operations within independent topology color batches',
            'requested_final_dmin':dmin}
    try:
        ids,lookup,cf,wr,wx,wf=compact_world(rest,x,faces,obstacles)
        mesh=collision_mesh(wr,wf,len(ids))
        if not full_rank(wx,wf) or ipctk.has_intersections(mesh,wx):raise RefinementFailure('Invalid coarse starting geometry')
        if minimum_gap(mesh,wx,options.gap_cap)<=dmin:raise RefinementFailure('Coarse state violates requested thickness')
        for color_index,color in enumerate(history.colors):
            report['active_color']=color_index
            # Measure gap BEFORE making the collapsed vertices coincident.
            gap=minimum_gap(mesh,wx,options.gap_cap)
            if gap<=1e-12:raise RefinementFailure('Insufficient coarse gap for perturbation')
            # Algorithm 3 distinguishes measured g0 (perturbation/ratio CCD)
            # from epsilon (minimum gap triggering local relaxation). Restoring
            # exactly g0 is a conservative optional policy, not the only one.
            required_gap=gap if options.minimum_gap is None else options.minimum_gap
            if gap<required_gap*(1-1e-8):raise RefinementFailure('Input batch gap is below the configured minimum expansion gap')
            targets={i:history.patches[i].targets(x) for i in color}
            history.update_topology(x,rest,faces,color)
            expanded.update(color)
            perturbations=[];movable=set();split_vertices=[]
            for i in color:
                p=history.patches[i]
                perturbations.append(safe_perturb(x,p.split,targets[i],p.pre_faces,
                                                  options.perturbation_fraction*gap,rng,options.sampling_budget))
                movable.update(map(int,p.ids));split_vertices.extend(p.split.tolist())
            ids,lookup,cf,wr,wx,wf=compact_world(rest,x,faces,obstacles)
            mesh=collision_mesh(wr,wf,len(ids))
            if not full_rank(wx,wf) or ipctk.has_intersections(mesh,wx):raise RefinementFailure('Global batch perturbation check failed')
            target=wx.copy()
            for i in color:target[lookup[history.patches[i].split]]=targets[i]
            wx,advances=ratio_advance(mesh,wx,target,gap,options)
            # Intermediate expansion uses zero-thickness IPC because topological
            # insertion initially creates distances far below final dmin.
            # Restore gap before the next batch, then certify actual thickness.
            contact=IPCContact(wr,wf,len(ids),dhat=2*gap,dmin=0,
                               stiffness=contact_stiffness,friction=0,ccd='additive')
            shell=Shell(wr[:len(ids)],cf,strain_limit=None,**material)
            proxy=None
            if minimum_gap(mesh,wx,required_gap)<required_gap*(1-1e-8) or not shell.feasible(wx[:len(ids)]):
                proxy=history.dynamic_prolongation(rest,faces,expanded)
            wx,relax=patch_relaxation(shell,wx,target,[lookup[v] for v in sorted(movable)],contact,required_gap,options,
                                      transfer=proxy,finest=finest if proxy is not None else shell)
            contact.validate(wx)
            x[ids]=wx[:len(ids)]
            report['batches'].append({'color':color_index,'splits':len(color),'gap_before':gap,
                                      'required_gap':required_gap,
                                      'gap_after':minimum_gap(mesh,wx,options.gap_cap),
                                      'perturbations':perturbations,'advances':advances,'relaxations':relax})
            if on_batch is not None:on_batch(report['batches'][-1])
        ids,lookup,cf,wr,wx,wf=compact_world(rest,x,faces,obstacles)
        if not np.array_equal(ids,np.arange(history.n)):raise RefinementFailure('Incomplete topology expansion')
        # Reorder faces to the source order for material arrays and diagnostics.
        cf=history.fine_faces;wf=np.vstack((cf,wf[len(cf):]))
        final_contact=IPCContact(wr,wf,history.n,dhat=options.gap_cap,dmin=dmin,
                                stiffness=contact_stiffness,friction=0,ccd='additive')
        final_contact.validate(wx)
        shell=Shell(history.fine_rest,history.fine_faces,strain_limit=strain_limit,**material)
        inherited=history.unsplit_coarse_ids if options.fixed_policy=='unsplit-lineage' else history.coarse_ids
        fixed=np.ones(len(wx),bool);fixed[:history.n]=False;fixed[inherited]=True
        inherited_start=wx[inherited].copy()
        report['strain_fixed_cloth_ids']=inherited.tolist()
        target=wx.copy();target[:history.n]=prolonged_target
        wx,strain= strain_relaxation(shell,wx,target,fixed,final_contact,options)
        if not np.array_equal(wx[inherited],inherited_start):raise RefinementFailure('Strain correction moved inherited vertices')
        if not shell.feasible(wx[:history.n]):raise RefinementFailure('Final strain constraints not satisfied')
        final_contact.validate(wx)
        report.update(accepted=True,strain_correction=strain,max_stretch=shell.max_stretch(wx[:history.n]),
                      final_gap_lower_bound=minimum_gap(final_contact.mesh,wx,options.gap_cap),
                      seconds=time.perf_counter()-started)
        return wx[:history.n].copy(),report
    except Exception as error:
        report.update(error=repr(error),seconds=time.perf_counter()-started)
        if hasattr(error,'diagnostics'):report['failure_diagnostics']=error.diagnostics
        failure=RefinementFailure(str(error));failure.report=report
        if hasattr(error,'failed_state'):
            failure.failed_state=error.failed_state
            failure.failed_rest=wr;failure.failed_faces=wf;failure.failed_cloth_vertices=len(ids)
        raise failure from error
