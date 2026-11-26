"""Isolated 10-step linear-solver equivalence/performance probe; not a suite result."""
import json
import shutil
from pathlib import Path
import numpy as np
import run_flag_parallel_wind as runner

root=runner.ROOT/'output/flag_pcg_probe_20261122'
source=runner.ROOT/'output/flag_to_L9_20261122/L2'
root.mkdir(exist_ok=False)
shutil.copytree(source/'assets',root/'assets')
for name in ['bundle.bin','scene.json']:shutil.copy2(source/name,root/name)
runner.ORDERED_PACKAGE=runner.ROOT/'output/flag_L3_fix_20261118/bunny_ordered_pcg_runtime/src'
runner.STEPS=10
runner.simulate(root,.001,3.,'corners',2,'semi',runner.ROOT/'output/flag_L1_two_pins_wind300_20261122/L1_semi','proxy')
a=np.load(source/'L2_semi/positions.npy')[:11];b=np.load(root/'L2_semi/positions.npy')
stats=json.loads((source/'L2_semi/steps.json').read_text())[:10]
meta=json.loads((root/'L2_semi/run.json').read_text())
result=dict(steps=10,max_vertex_difference_m=float(np.linalg.norm(a-b,axis=2).max()),
    baseline_step_seconds=sum(s['seconds'] for s in stats),probe_seconds=meta['elapsed_s'],
    production_results_unchanged=True)
(root/'comparison.json').write_text(json.dumps(result,indent=2))
print(result,flush=True)
