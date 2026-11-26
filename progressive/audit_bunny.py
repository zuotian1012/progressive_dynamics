"""Recheck accepted bunny states, including geometry; never inspect unwritten tails."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .affine import AffineObstacles
from .contact import IPCContact
from .runtime import numerical_threads
from .shell import Shell


def audit(root,accepted_prefix=False):
    root=Path(root)
    cfg=json.loads((root/'experiment.json').read_text())
    p=cfg['parameters']
    if p['scene']!='bunny':raise ValueError('This audit requires a bunny scene')
    if not cfg['completed'] and not accepted_prefix:raise ValueError('Incomplete experiment; request --accepted-prefix explicitly')
    assets=Path(p['assets'])
    for name,digest in cfg['asset_files_sha256'].items():
        if hashlib.sha256((assets/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Asset changed since simulation: {name}')
    for name in ('affine.py','contact.py','shell.py','differentials.py'):
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=cfg['code_sha256'][name]:
            raise ValueError(f'Audit kernel differs from simulation: {name}')
    result={'experiment_completed':cfg['completed'],'accepted_prefix_only':accepted_prefix,
            'scope':'All saved accepted states including rest; discrete intersection/thickness and recomputed strain checks. Does not independently certify between-frame CCD or original-paper reproduction.',
            'runs':[]}
    with np.load(assets/'bunny.npz') as b:fixed=[(b['vertices'],b['faces'])]
    for path in sorted(root.glob('*_L*/run.json')):
        meta=json.loads(path.read_text());folder=path.parent;count=meta['completed_steps']
        with np.load(assets/f'mesh_{meta["level"]}.npz') as data:rest,faces=data['vertices'],data['faces']
        shell=Shell(rest,faces,young=p['young'],poisson=p.get('poisson',.3),density=p['density'],thickness=p['thickness'],
                    strain_limit=p['strain_limit'],strain_activation=p['strain_activation'])
        embedding=AffineObstacles(rest,faces,fixed=fixed)
        contact=IPCContact(embedding.positions(embedding.q0),embedding.faces,len(rest),p['dhat'],p['dmin'],p['contact_stiffness'],p['friction'])
        # The source can be preallocated past the last accepted checkpoint.
        states=np.load(folder/'positions.npy',mmap_mode='r')[:count+1]
        logs=json.loads((folder/'solver.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())[:count]
        timing=np.load(folder/'timing.npz')['step_seconds'][:count]
        if len(states)!=count+1 or len(logs)<count or len(metrics)!=count or len(timing)!=count:
            raise ValueError(f'{folder.name}: truncated accepted prefix')
        if not np.isfinite(states).all() or not np.isfinite(timing).all() or (timing<0).any():raise ValueError('Nonfinite or invalid accepted data')
        if not np.array_equal(states[0],rest):raise ValueError('Saved initial geometry differs from original rest')
        preview=0
        for entry in logs[:count]:
            if not entry.get('accepted',False):raise ValueError('Unaccepted frame in prefix')
            if not entry['converged']:
                if meta['method']=='direct' or meta['level']==p['levels']-1 or not entry.get('preview_stop'):
                    raise ValueError('Strict solve did not converge')
                preview+=1
        stretch=[];active_frames=0
        for i,state in enumerate(states):
            maximum=shell.max_stretch(state);stretch.append(maximum)
            if maximum>=p['strain_limit']:raise ValueError(f'{folder.name} accepted frame {i}: strain limit violated')
            world=embedding.positions(state)
            try:contact.validate(world)
            except ValueError as error:raise ValueError(f'{folder.name} accepted frame {i}: {error}') from error
            active_frames+=int(contact.barrier(contact.normal(world),contact.mesh,world)>0)
            if i and not np.isclose(maximum,metrics[i-1]['max_stretch'],rtol=1e-10,atol=1e-12):
                raise ValueError(f'{folder.name}: logged stretch differs from trajectory')
        result['runs'].append({'method':meta['method'],'level':meta['level'],'completed':meta['completed'],
            'accepted_steps':count,'requested_steps':meta['steps'],'newton_converged_steps':count-preview,
            'preview_stopped_steps':preview,'max_principal_stretch':max(stretch),
            'accepted_physics_seconds':float(timing.sum()),'max_accepted_newton_iterations':max((r['iterations'] for r in logs[:count]),default=0),
            'geometry_frames_checked':len(states),'frames_with_active_contact':active_frames,
            'failure':meta.get('error'),'failed_solver_report':logs[count] if len(logs)>count else None})
    if not result['runs']:raise ValueError('No saved levels')
    if cfg['completed']:
        expected=p['levels']*len(p['methods'])
        if len(cfg['runs'])!=expected or any(not r['completed'] or r['accepted_steps']!=cfg['steps'] for r in result['runs']):
            raise ValueError('Completed manifest lacks required trajectories')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--accepted-prefix',action='store_true')
    args=parser.parse_args()
    with numerical_threads(1):result=audit(args.root,args.accepted_prefix)
    target=args.root/('audit_bunny_prefix.json' if args.accepted_prefix else 'audit_bunny.json')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
