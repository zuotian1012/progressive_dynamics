"""Measure original-runtime replay variability before interpreting solver differences."""
import json
import argparse
import shutil
import numpy as np
import run_flag_parallel_wind as runner
p=argparse.ArgumentParser();p.add_argument('--cached',action='store_true');args=p.parse_args()
root=runner.ROOT/('output/flag_cached_probe_20261122' if args.cached else 'output/flag_ldlt_repeat_20261122');source=runner.ROOT/'output/flag_to_L9_20261122/L3'
if args.cached:runner.ORDERED_PACKAGE=runner.ROOT/'output/flag_L3_fix_20261118/bunny_ordered_cached_runtime/src'
root.mkdir(exist_ok=False);shutil.copytree(source/'assets',root/'assets')
for name in ['bundle.bin','scene.json']:shutil.copy2(source/name,root/name)
runner.STEPS=10
runner.simulate(root,.001,3.,'corners',3,'semi',runner.ROOT/'output/flag_to_L9_20261122/L2/L2_semi','proxy')
a=np.load(source/'L3_semi/positions.npy')[:11];b=np.load(root/'L3_semi/positions.npy')
c=np.load(runner.ROOT/'output/flag_mkl_probe_v2_20261122/L3_semi/positions.npy')
result=dict(steps=10,repeat_max_vertex_difference_m=float(np.linalg.norm(a-b,axis=2).max()),
    mkl_vs_repeat_max_vertex_difference_m=float(np.linalg.norm(c-b,axis=2).max()),production_results_unchanged=True)
result['probe_seconds']=json.loads((root/'L3_semi/run.json').read_text())['elapsed_s']
result['baseline_step_seconds']=sum(s['seconds'] for s in json.loads((source/'L3_semi/steps.json').read_text())[:10])
(root/'comparison.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
