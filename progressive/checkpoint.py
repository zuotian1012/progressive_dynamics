"""Resume accepted prefixes into a NEW output, retaining original IPC rest data."""
import json
from pathlib import Path
import numpy as np


def validate_resume(root,config,allow_code_change=False):
    root=Path(root)
    previous=json.loads((root/'experiment.json').read_text())
    old,new=previous['parameters'],config['parameters']
    ignored={'output','config','frames','steps','max_iterations','threads','resume_from','resume_allow_code_change','resume_interrupted','checkpoint_every','only_level','coarse_from'}
    legacy={'offset_linesearch':'frozen','strict_preview':True,'preview_min_alpha':1e-8,
            'strain_model':'legacy','strain_stiffness':1e4,'strain_activation':1+(old['strain_limit']-1)*.5,
            'elastic_hessian':'finite_difference','convergence_norm':'max','ccd':'tight','hinge_difference':'periodic',
            'warm_start':'horizontal','expansion_seed':527,'expansion_gap_cap':.001,
            'expansion_fixed_policy':'coarse-indices','poisson':.3}
    for key in (old.keys()|new.keys())-ignored:
        if key=='expansion_fixed_policy' and old.get('warm_start','horizontal')==new.get('warm_start','horizontal')=='horizontal':continue
        a,b=old.get(key,legacy.get(key)),new.get(key,legacy.get(key))
        if key=='assets':a,b=str(Path(a).resolve()),str(Path(b).resolve())
        if a!=b:raise ValueError(f'Cannot resume with a different {key}: {a!r} -> {b!r}')
    for key in ('mesh_hashes','dt','robot_q_sha256','versions'):
        if previous.get(key)!=config.get(key):raise ValueError(f'Resume {key} mismatch')
    if new.get('scene')=='bunny' and previous.get('asset_files_sha256')!=config.get('asset_files_sha256'):
        raise ValueError('Resume bunny geometry or intrinsic-map asset mismatch')
    if new.get('warm_start','horizontal')=='vertical' and previous.get('expansion_history_sha256')!=config.get('expansion_history_sha256'):
        raise ValueError('Resume expansion history mismatch')
    kernels=('affine.py','contact.py','integrators.py','prolongation.py','shell.py','solver.py','differentials.py')
    if new.get('warm_start','horizontal')=='vertical':
        kernels+=('expansion_history.py','refinement_contact.py','refinement_relaxation.py','safe_expansion.py')
    changed=[k for k in kernels if previous.get('code_sha256',{}).get(k)!=config['code_sha256'][k]]
    if changed and not allow_code_change:raise ValueError(f'Changed physics code {changed}; use an explicit diagnostic continuation if intended')
    return {'source':str(root.resolve()),'source_completed':previous['completed'],
            'changed_kernel_files':changed,'source_code_sha256':previous.get('code_sha256',{}),
            'timing_warning':'Inherited and new timings use different runs; do not use this continuation for performance claims'}


def restore_level(root,method,level,track,poses,embedding,steps,allow_interrupted=False):
    root=Path(root)
    config=json.loads((root/'experiment.json').read_text())
    entry=next((r for r in config['runs'] if r['method']==method and r['level']==level),None)
    # A process can stop after flushing a level checkpoint but before appending
    # the level to the root manifest. Recover that accepted prefix explicitly.
    folder=root/((entry.get('reused_from') or entry['folder']) if entry else f'{method}_L{level}')
    if not (folder/'run.json').exists():return None
    meta=json.loads((folder/'run.json').read_text())
    if not meta['completed'] and 'error' not in meta and not allow_interrupted:
        raise ValueError('Refusing a possibly still-running source checkpoint')
    count=meta['completed_steps']
    if count>steps:raise ValueError('Resume target ends before the accepted checkpoint')
    x=np.load(folder/'positions.npy',mmap_mode='r');p=np.load(folder/'robot.npy',mmap_mode='r')
    if x.shape[1:]!=track.shape[1:] or p.shape[1:]!=poses.shape[1:]:raise ValueError('Checkpoint state shape mismatch')
    track[:count+1]=x[:count+1];poses[:count+1]=p[:count+1]
    q=embedding.q0.copy();q[:embedding.n]=track[count]
    for i,t in enumerate(poses[count]):
        q[embedding.n+4*i:embedding.n+4*i+4]=np.vstack((t[:3,3],t[:3,:3].T))
    reports=json.loads((folder/'solver.json').read_text())[:count]
    metrics=json.loads((folder/'metrics.json').read_text())[:count]
    timings=np.load(folder/'timing.npz')['step_seconds'][:count].tolist()
    if len(reports)!=count or len(metrics)!=count or len(timings)!=count:
        raise ValueError('Incomplete checkpoint logs')
    if not all(r.get('accepted',r['converged']) for r in reports):raise ValueError('Unaccepted frame in checkpoint prefix')
    if not np.isfinite(track[:count+1]).all() or not np.isfinite(q).all():raise ValueError('Nonfinite checkpoint')
    return count,q,reports,metrics,timings
