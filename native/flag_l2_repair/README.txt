Flag L2: safeguarded energy-based backtracking

The new flag_line_search.h fits a quadratic to E(0) and two failed finite trials.
For positive curvature and a negative estimated slope it proposes the minimizer,
clamped to [0.1,0.5] times the last alpha. Otherwise it retains halving.
Every proposal is evaluated by the existing full-energy line search, with
unchanged max_iter=8 and existing feasibility/CCD/CFL restrictions.
The original swept cap is explicitly retained for non-descent probes; it must
not be reconstructed with ldexp once shrink factors can differ from 1/2.
Native convergence semantics, PSQ gradients, material and time step are unchanged.

Opt-in: UIPC_PSQ_ADAPTIVE_SEARCH=1 via --adaptive-search.
Build: .venv/Scripts/python.exe scripts/build_flag_l2_fix.py --variant rescue
Run via scripts/run_flag_l2_fix.py and the explicitly selected independent DLL.
No existing L0/L1/shirt code or runtime is overwritten.

Results: output/flag_L2_fix_20261118/RESULTS.txt
