"""Audit and visualize matched physical-grip runs for the lineage correction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def report(positive,negative,output):
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing nonempty output')
    output.mkdir(parents=True,exist_ok=True)
    roots={'mu1':positive,'mu0':negative};audit={'scope':'Small physical grip, not full shirt folding or a performance benchmark','runs':{}}
    tracks={}
    kernels=['affine.py','contact.py','integrators.py','prolongation.py','shell.py','solver.py',
             'differentials.py','expansion_history.py','refinement_contact.py','refinement_relaxation.py','safe_expansion.py']
    for label,root in roots.items():
        config=json.loads((root/'experiment.json').read_text())
        if not config['completed']:raise ValueError(f'Incomplete run: {root}')
        if config['parameters']['expansion_fixed_policy']!='unsplit-lineage':raise ValueError('Wrong fixed policy')
        for key in kernels:
            if config['code_sha256'][key]!=hashlib.sha256((Path('progressive')/key).read_bytes()).hexdigest():
                raise ValueError(f'Changed kernel: {key}')
        audit.setdefault('source_experiments',{})[label]={'path':str(root.resolve()),
            'sha256':hashlib.sha256((root/'experiment.json').read_bytes()).hexdigest()}
        for run in config['runs']:
            folder=root/run['folder'];x=np.load(folder/'positions.npy');poses=np.load(folder/'robot.npy')
            solvers=json.loads((folder/'solver.json').read_text())
            if not all(s['converged'] for s in solvers):raise ValueError('Nonconverged physical step')
            inits=[json.loads(p.read_text()) for p in sorted(folder.glob('initialization_*.json'))]
            if not all(i['accepted'] for i in inits):raise ValueError('Unaccepted initializer')
            lift=(x[:,x[0,:,2]>.06,2]-x[0,x[0,:,2]>.06,2]).mean(axis=1)
            key=label+'/'+folder.name;tracks[key]=(x,poses,lift,folder)
            audit['runs'][key]={'steps':run['completed_steps'],'vertices':len(x[0]),'all_physical_steps_converged':True,
                'accepted_initializers':len(inits),'top_material_definition':'rest z > 0.06 m',
                'top_mean_lift_m':float(lift[-1]),'jaw_mean_lift_m':float((poses[-1,:,2,3]-poses[0,:,2,3]).mean()),
                'max_stretch':max(s['max_stretch'] for s in solvers),'physics_seconds':run['physics_seconds'],
                'initialization_seconds':sum(i['seconds'] for i in inits),
                'min_initialization_gap_lower_bound_m':min((i['final_gap_lower_bound'] for i in inits),default=None)}
    a=json.loads((positive/'experiment.json').read_text());b=json.loads((negative/'experiment.json').read_text())
    ignored={'output','friction'}
    if {k:v for k,v in a['parameters'].items() if k not in ignored}!={k:v for k,v in b['parameters'].items() if k not in ignored}:
        raise ValueError('Controls differ beyond friction and output')
    if a['mesh_hashes']!=b['mesh_hashes']:raise ValueError('Different control meshes')
    audit['controls_match_except_friction']=True
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for label,ax in zip(roots,axes):
        for key,(x,poses,lift,folder) in tracks.items():
            if not key.startswith(label+'/'):continue
            ax.plot(np.arange(len(x))*.01,lift*1000,label=f'{folder.name} ({len(x[0])} vertices)')
        ax.set(title=f'Friction = {label[-1]}',xlabel='Simulated time (s)',ylabel='Upper material mean lift (mm)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Matched contact/friction controls; independent vertical scales')
    fig.savefig(output/'displacement.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(12,7));items=[]
    for i,(key,title) in enumerate([(f'{mu}/{name}',f'{mu}: {name}') for mu in roots for name in ('velpro_L0','velpro_L1','direct_L1')]):
        x,poses,lift,folder=tracks[key];faces=np.load(folder/'faces.npy');ax=fig.add_subplot(2,3,i+1,projection='3d')
        ax.set(xlim=(-.055,.055),ylim=(-.02,.02),zlim=(-.005,.115) if key.startswith('mu1') else (-1.8,.115),
               xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=title)
        ax.set_xticks([-.04,0,.04]);ax.set_yticks([-.01,.01]);ax.set_box_aspect((1,.4,1.2));ax.view_init(18,-65)
        cloth=Poly3DCollection(x[0][faces],facecolors='#79bdd9',edgecolors='#18516e',linewidths=.5,alpha=.9);ax.add_collection3d(cloth)
        jaws=[]
        for j in range(2):
            with np.load(folder/f'collider_{j}.npz') as c:v,f=c['vertices'],c['faces']
            body=Poly3DCollection((v @ poses[0,j,:3,:3].T+poses[0,j,:3,3])[f],facecolors='#eeae53',edgecolors='#8f5d10',linewidths=.3,alpha=.4)
            ax.add_collection3d(body);jaws.append((body,v,f))
        items.append((cloth,jaws,x,faces,poses))
    title=fig.suptitle('');fig.subplots_adjust(hspace=.28)
    def update(frame):
        title.set_text(f'Physical jaws, no cloth pins | t = {frame*.01:.2f} s | playback 0.2x\nRows have different z scales; lower row includes full fall distance')
        for cloth,jaws,x,faces,poses in items:
            cloth.set_verts(x[frame][faces])
            for j,(body,v,f) in enumerate(jaws):body.set_verts((v @ poses[frame,j,:3,:3].T+poses[frame,j,:3,3])[f])
        return []
    update(60);fig.savefig(output/'final.png',dpi=130)
    FuncAnimation(fig,update,frames=range(0,61,2),interval=100,blit=False).save(output/'grip_controls.gif',writer=PillowWriter(fps=10),dpi=90)
    plt.close(fig)
    audit['current_kernel_sha256']={k:hashlib.sha256((Path('progressive')/k).read_bytes()).hexdigest() for k in kernels}
    (output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit['runs'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--positive',type=Path,required=True)
    p.add_argument('--negative',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.positive,a.negative,a.output)
