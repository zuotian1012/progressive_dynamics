"""Compare horizontal and safe-refined seeds for the SAME fine incremental step.

This is an initializer sensitivity experiment, not a folding benchmark. Gravity
is zero and the previous state is feasible; neither cloth component is pinned.
The unsafe direct prolongation is never used as a solver initial state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import eye
from progressive.contact import IPCContact
from progressive.prolongation import ShellProlongation
from progressive.refinement_contact import minimum_gap
from progressive.runtime import numerical_threads
from progressive.shell import Shell
from progressive.solver import IncrementalProblem,NewtonOptions,solve


def compare_seeds(rest,faces,previous,refined,dt=.01,max_iterations=300,tolerance=1e-3):
    results={};states={}
    for name,seed in [('horizontal',previous),('refined',refined)]:
        # Independent material/contact state: adaptive stiffness and friction
        # history from one solve must never leak into the other.
        shell=Shell(rest,faces)
        transfer=ShellProlongation(rest,rest,faces,eye(len(rest)))
        contact=IPCContact(rest,faces,len(rest),stiffness=1e5,friction=1.)
        contact.validate(previous);contact.validate(seed)
        problem=IncrementalProblem(shell,shell,transfer,dt,previous.copy(),previous.copy(),
                                   gravity=(0,0,0),contact=contact)
        initial_energy=problem.evaluate(seed,transfer.offset(seed))[0]
        final,report=solve(problem,seed,NewtonOptions(max_iterations=max_iterations,velocity_tolerance=tolerance))
        contact.validate(final)
        report.update(initial_energy=initial_energy,
                      gap_lower_bound=minimum_gap(contact.mesh,final,.001))
        results[name]=report;states[name]=final
        print(f'{name}: converged={report["converged"]}, iterations={report["iterations"]}, '
              f'seconds={report["seconds"]:.3f}, energy={report["energy"]:.9g}',flush=True)
    # Re-evaluate both results under the SAME final stiffness, even if the two
    # adaptive Newton paths changed it differently. Report that change explicitly.
    kappa=max(r['strain_stiffness'] for r in results.values())
    shell=Shell(rest,faces,strain_stiffness=kappa)
    contact=IPCContact(rest,faces,len(rest),stiffness=1e5,friction=1.)
    transfer=ShellProlongation(rest,rest,faces,eye(len(rest)))
    common=IncrementalProblem(shell,shell,transfer,dt,previous,previous,gravity=(0,0,0),contact=contact)
    energies={k:common.evaluate(v,transfer.offset(v))[0] for k,v in states.items()}
    displacement=np.linalg.norm(states['refined']-states['horizontal'],axis=1)
    initial_displacement=np.linalg.norm(refined-previous,axis=1)
    summary={'both_converged':all(r['converged'] for r in results.values()),
             'same_final_stiffness':results['horizontal']['strain_stiffness']==results['refined']['strain_stiffness'],
             'common_evaluation_stiffness':kappa,'common_final_energies':energies,
             'final_energy_difference':energies['refined']-energies['horizontal'],
             'initial_seed_difference_rms_m':float(np.sqrt(np.mean(initial_displacement**2))),
             'initial_seed_difference_max_m':float(initial_displacement.max()),
             'final_solution_difference_rms_m':float(np.sqrt(np.mean(displacement**2))),
             'final_solution_difference_max_m':float(displacement.max())}
    return states,results,summary


def validate(source,output,max_iterations=300):
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing nonempty output')
    metadata=json.loads((source/'validation.json').read_text())
    if not metadata['accepted']:raise ValueError('Source initialization was not accepted')
    assets=Path(metadata['assets'])
    fine=np.load(assets/'mesh_1.npz');rest=fine['vertices'];faces=fine['faces']
    if metadata['case']=='self-contact':
        nf=len(rest)
        rest=np.vstack((rest,[[-2.,-1.,-.2],[2.,-1.,-.2],[2.,2.,-.2],[-2.,2.,-.2]]))
        faces=np.vstack((faces,np.array([[0,1,2],[0,2,3]])+nf))
        previous=rest.copy();previous[:nf,2]=0;previous[nf:,2]=-.002
    else:previous=rest.copy()
    saved=np.load(source/'states.npz')
    np.testing.assert_array_equal(faces,saved['fine_faces'])
    output.mkdir(parents=True,exist_ok=True)
    report={'source':str(source.resolve()),'case':metadata['case'],
            'scope':'one identical fine incremental step; zero gravity, zero previous velocity, no pins; initializer sensitivity only',
            'parameters':{'dt':.01,'max_iterations':max_iterations,'velocity_tolerance':1e-3},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'validation.json',source/'states.npz')},
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('progressive').glob('*.py')},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    states,results,summary=compare_seeds(rest,faces,previous,saved['safe'],max_iterations=max_iterations)
    report.update(summary=summary,runs=results)
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(output/'states.npz',rest=rest,faces=faces,previous=previous,refined_seed=saved['safe'],**states)
    make_plot(previous,saved['safe'],states,faces,output/'comparison.png')
    print(json.dumps(summary,indent=2),flush=True)
    return report


def make_plot(previous,seed,states,faces,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained',sharex=True,sharey=True)
    for ax,x,title in zip(axes,[seed,states['horizontal'],states['refined']],
                         ['Safe initialization','Horizontal fine solution','Refined fine solution']):
        segments=x[edges][:,:,[0,2]].copy();segments[:,:,1]*=1000
        ax.add_collection(LineCollection(segments,colors='#226b9e',linewidths=.8));ax.autoscale()
        ax.set(title=title,xlabel='x (m)');ax.grid(alpha=.2)
    axes[0].set_ylabel('z (mm)')
    fig.suptitle('Identical fine incremental problem; full height range, y suppressed')
    fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-iterations',type=int,default=300)
    a=p.parse_args()
    with numerical_threads(1):validate(a.source,a.output,a.max_iterations)
