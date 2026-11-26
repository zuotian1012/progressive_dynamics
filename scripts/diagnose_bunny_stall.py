"""Read-only replay of the failed coarse incremental problem; no simulation changes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse.linalg import spsolve
from scripts.replay_progressive_step import load_replay
from progressive.runtime import numerical_threads
from progressive.solver import solve,NewtonOptions

out=Path('output/bunny_failure_diagnosis')
out.mkdir(exist_ok=False)
with numerical_threads(1):
    problem,q,args,inputs=load_replay(Path('output/bunny_paper_scale/velpro'),0,56)
    opt=NewtonOptions(max_iterations=80,velocity_tolerance=args.tolerance,hessian=args.hessian,
        offset_linesearch=args.offset_linesearch,allow_preview_stall=True,
        preview_min_alpha=args.preview_min_alpha,convergence_norm=args.convergence_norm)
    x,report=solve(problem,q,opt)
    np.save(out/'diagnostic_iterate.npy',x)
    (out/'replay.json').write_text(json.dumps(report,indent=2))
    offset=problem.transfer.offset(x[:problem.n])
    value,g,H=problem.evaluate(x,offset,True,args.hessian)
    d=spsolve(H.tocsc(),-g.ravel()).reshape(x.shape)
    d/=np.linalg.norm(d)
    derivatives=[]
    for h in [1e-5,1e-6,1e-7,1e-8]:
        frozen=[];nonlinear=[]
        for sign in [1,-1]:
            trial=x+sign*h*d
            frozen.append(problem.evaluate(trial,offset)[0])
            nonlinear.append(problem.evaluate(trial,problem.transfer.offset(trial[:problem.n]))[0])
        derivatives.append({'epsilon':h,'frozen_fd':(frozen[0]-frozen[1])/(2*h),
            'nonlinear_fd':(nonlinear[0]-nonlinear[1])/(2*h)})
    result={'scope':'Diagnostic optimization iterate, not an accepted trajectory frame',
        'source_step_zero_based':56,'replay_iterations':report['iterations'],'replay_reason':report['reason'],
        'last_trace':report['trace'][-1],'input_hashes':inputs,
        'predicted_directional_derivative':float(np.sum(g*d)),
        'direction_norm':float(np.linalg.norm(d)),'directional_derivatives':derivatives,
        'zero_offset':problem.transfer.zero_offset}
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
