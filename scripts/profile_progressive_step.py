"""Profile an unchanged saved incremental problem before optimizing kernels."""
import argparse
import cProfile
from pathlib import Path
import pstats
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.replay_progressive_step import load_replay
from progressive.runtime import numerical_threads
from progressive.solver import solve,NewtonOptions


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--level',type=int,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--initial-state',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refusing existing profile output')
    a.output.mkdir(parents=True)
    with numerical_threads(1):
        problem,q,args,inputs=load_replay(a.root,a.level,a.step)
        if a.initial_state:q=np.load(a.initial_state)
        profile=cProfile.Profile();profile.enable()
        _,result=solve(problem,q,NewtonOptions(5,args.tolerance,hessian=args.hessian,
                                               offset_linesearch=getattr(args,'offset_linesearch','frozen')))
        profile.disable();profile.dump_stats(str(a.output/'profile.pstats'))
        with (a.output/'profile.txt').open('w',encoding='utf-8') as stream:
            pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(45)
        print({k:result[k] for k in ('converged','iterations','seconds')})
