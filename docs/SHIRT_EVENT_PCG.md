# Shirt event-fallback GPU PCG

The opt-in `--event-fallback` mode reuses the verified bunny event runtime.
Every level, including L0, must terminate every accepted step with native Newton
convergence. Non-descent, small-step, semi-implicit, and budget-exhaustion exits
are not accepted. Existing CPU/preview results remain historical controls.

The internal coarse-level preview detector remains active to trigger recovery:
restore the current iterate, check its energy, switch to the full PSQ gradient
for the rest of that step, and require at least three full-gradient directions.
The next step starts with frozen offsets again. This is not preview acceptance.
The identity finest level has no coarse preview detector.

Defaults for this mode are PCG squared preconditioned residual ratio `1e-10`
and free-vertex RMS Newton velocity tolerance `0.001 m/s`. The existing ABD
checker also remains active for the grippers. Explicit tolerances may tighten,
but not loosen, those defaults. Pass `--max-iterations 1500` to match the bunny
iteration budget; the iteration budget never authorizes accepting a failed step.
The shirt's time step, contact/friction, material, strain limit, and robot path
are unchanged. Only the solver/acceptance protocol changes.

From the project root, the initial L0 settling test is:

```powershell
.\.venv\Scripts\python.exe scripts/supervise_shirt_psq.py -- `
  --runtime-package output/flag_L3_fix_20261118/bunny_event_fallback_runtime/src `
  --assets output/shirt_four_level_pickup_20261117/candidate_f17000_n0.3_p0.01 `
  --psq-bundle output/shirt_four_level_pickup_20261117/L0.bin --level 0 `
  --trajectory output/shirt_L0_grasp_20261117/laydown_v2/trajectory.npz `
  --frames 0 --settle-steps 60 --strain-cap directional `
  --relaxed-solver --disable-semi-implicit --reject-inf-trials `
  --record-input-velocities --line-search-max-iterations 32 `
  --event-fallback --max-iterations 1500 --output NEW_OUTPUT_DIRECTORY
```

Use a fresh output directory. Do not add `--coarse-preview`. To cover closing
and its hold period from the same initial state, use `--frames 298` (656 physical
steps including settling). A complete fold uses `--frames 1708` (3476 steps).
Higher levels require a completed immediate parent from this same runtime and
strict protocol; the old preview hierarchy is deliberately rejected.

Audit a completed result with:

```powershell
.\.venv\Scripts\python.exe scripts/audit_shirt_event.py NEW_OUTPUT_DIRECTORY
```

This checks all accepted statuses, actual solver/runtime, saved source hashes,
energy/line-search records, per-step fallback transitions, every saved cloth
state's strain and self-intersection, and (on higher levels) VelPro transfer.
It does not independently prove gripper-cloth clearance or continuous-time
collision freedom. Rendering alone is not acceptance.
