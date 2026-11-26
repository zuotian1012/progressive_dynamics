"""Separate reproducibility of a failed prefix from successful convergence."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(root,continuation):
    reports=[json.loads(p.read_text()) for p in sorted(root.glob('repeat_*/report.json'))]
    if len(reports)!=3:raise ValueError('Need all three completed repeat reports')
    states=[np.load(p) for p in sorted(root.glob('repeat_*/state.npy'))]
    same_states=all(np.array_equal(states[0],s) for s in states[1:])
    same_trace=all(r['trace']==reports[0]['trace'] for r in reports[1:])
    same_inputs=all(r['input_hashes']==reports[0]['input_hashes'] for r in reports[1:])
    later=json.loads((continuation/'report.json').read_text())
    output={'repeats':len(reports),'identical_input_hashes':same_inputs,'identical_final_state_bytes':same_states,
            'identical_iteration_traces':same_trace,'prefix_converged':[r['converged'] for r in reports],
            'prefix_iterations':[r['iterations'] for r in reports],
            'continuation_converged':later['converged'],'continuation_iterations':later['iterations'],
            'total_iteration_checks':reports[0]['iterations']+later['iterations'],
            'scope':'Three identical 150-iteration prefixes, then continuation of the unchanged incremental problem; not proof of the complete grasp sequence'}
    if not (same_states and same_trace and same_inputs and later['converged']):
        raise ValueError('Repeatability/convergence checks failed')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i,r in enumerate(reports):
        ax.semilogy([v['iteration'] for v in r['trace']],[v['velocity_decrement'] for v in r['trace']],label=f'Correct rest, repeat {i+1}',alpha=.7)
    n=len(reports[0]['trace'])
    ax.semilogy([n+v['iteration'] for v in later['trace']],[v['velocity_decrement'] for v in later['trace']],c='black',marker='o',label='Same problem continued')
    ax.axhline(.001,ls='--',c='red',label='Unchanged velocity tolerance')
    ax.set(xlabel='Newton iteration',ylabel='Velocity decrement (m/s)',title='Scene527 level 1, step 22: deterministic replay')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(root/'convergence.png',dpi=150);plt.close(fig)
    (root/'validation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('continuation',type=Path);a=p.parse_args()
    analyze(a.root,a.continuation)
