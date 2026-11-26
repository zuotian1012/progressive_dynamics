Flag L1: bounded refinement of non-descent diagnostic probes

This isolated copy changes only psq_preview_stop.inl relative to the L0 repair code.
When the original three positive resolved slopes fail the 25% consistency test,
shift the stencil down one decade, at most three times. Retain the 64-epsilon
noise floor and 25% test; stop refining if the changes become unresolved.
Probe points are never accepted motion updates. Restore the current Newton
iterate for a verified preview stop, otherwise restore the last tested alpha.

Opt-in: UIPC_PSQ_REFINE_PROBES=1, exposed by --refined-probe.
Build: .venv/Scripts/python.exe scripts/build_flag_l1_fix.py --variant rescue
Run via scripts/run_flag_l1_fix.py and the explicitly selected new runtime.
The completed L0 input keeps its original backend provenance. The L1 backend
change is explicit in metadata; original L0/shirt files are not overwritten.

Results and commands: output/flag_L1_fix_20261118/RESULTS.txt
