"""Run CUDA PSQ shirt dynamics, optionally driven by a lower-level velocity track."""
import argparse
from pathlib import Path
import sys
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.libuipc_native_runtime import load
runtime_parser=argparse.ArgumentParser(add_help=False)
runtime_parser.add_argument('--runtime-package',type=Path)
runtime_args,_=runtime_parser.parse_known_args()
load(runtime_args.runtime_package)
from shirt_demo.simulate import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-package',type=Path,help='Explicit isolated uipc package root containing uipc/_native')
    p.add_argument('--assets',type=Path,default=Path('output/shirt55k_psq_ready'))
    p.add_argument('--psq-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=740)
    p.add_argument('--level',type=int,default=0,help='Level index validated against the asset hierarchy')
    p.add_argument('--lower',type=Path,help='Completed immediate lower-level run for VelPro velocity prediction')
    p.add_argument('--trajectory',type=Path,help='Explicit alternative joint trajectory; preserves the original initial robot state')
    p.add_argument('--settle-steps',type=int,default=60)
    p.add_argument('--max-iterations',type=int,default=100)
    p.add_argument('--force-updates',action='store_true')
    p.add_argument('--strain-cap',choices=['legacy','directional'],default='legacy')
    p.add_argument('--cap-audit',type=Path,help='Read-only sample of actual Newton directions and binding faces (JSONL)')
    p.add_argument('--relaxed-solver',action='store_true',help='Use native defaults, including semi-implicit termination; no custom forced acceptance')
    p.add_argument('--disable-semi-implicit',action='store_true',help='With --relaxed-solver, change only newton.semi_implicit.enable to 0')
    p.add_argument('--line-search-max-iterations',type=int,help='Explicit PSQ backtracking budget override; convergence and energy acceptance stay unchanged')
    p.add_argument('--fem-preconditioner',choices=['diag','mas'],help='Explicit native linear preconditioner override; physical energies and convergence tolerances stay unchanged')
    p.add_argument('--consistency-audit',action='store_true',help='Diagnostic energy probes with trial-state restoration, plus ABD component gradients')
    p.add_argument('--coarse-preview',action='store_true',help='Non-finest level: verified non-descent exit; not Newton convergence')
    p.add_argument('--require-convergence',action='store_true',help='Stop before saving any nonconverged or budget-exhausted step; requires semi-implicit OFF and no preview')
    p.add_argument('--event-fallback',action='store_true',help='Ordered GPU PCG; switch to full PSQ gradient on non-descent and require convergence at every level')
    p.add_argument('--linear-tol-rate',type=float,help='Event mode PCG squared residual ratio (default 1e-10)')
    p.add_argument('--newton-velocity-tol',type=float,help='Event mode RMS velocity tolerance (default .001 m/s)')
    p.add_argument('--record-input-velocities',action='store_true',help='Read-only recording of native cloth velocity immediately before each advance')
    p.add_argument('--diagnostic-contact',choices=['full','no-self','off'],default='full',help='L0-only causal probe; altered contacts are diagnostic, not a production fix')
    p.add_argument('--diagnostic-zero-gravity',action='store_true',help='L0-only rest-state causal probe')
    p.add_argument('--robot-drive-scale',type=float,default=1.,help='Multiply only the robot soft-target translation/rotation strengths; baseline is 1')
    p.add_argument('--reject-inf-trials',action='store_true',help='Reject PSQ +inf trial energies by backtracking; initial/non-PSQ nonfinite energies remain fatal')
    p.add_argument('--stop-after-consecutive-newton-limits',type=int,default=0,
                   help='Diagnostic run guard: stop after N consecutive capped steps (0 disables); does not change a solve')
    p.set_defaults(substeps=2,young=30000.,shear_young=4.,bending=.00001)
    a=p.parse_args()
    os.environ['UIPC_PSQ_STRAIN_CAP']=a.strain_cap
    if a.consistency_audit:os.environ['UIPC_PSQ_CONSISTENCY_AUDIT']='1'
    if a.cap_audit:
        a.cap_audit.parent.mkdir(parents=True,exist_ok=True)
        os.environ['UIPC_PSQ_CAP_AUDIT']=str(a.cap_audit.resolve())
    if not 0<=a.frames<=4373 or a.settle_steps<0 or a.settle_steps+2*a.frames<=0:p.error('Invalid frame range; frames=0 runs settling only')
    if a.stop_after_consecutive_newton_limits<0:p.error('Newton-limit stop count must be nonnegative')
    run(a)
