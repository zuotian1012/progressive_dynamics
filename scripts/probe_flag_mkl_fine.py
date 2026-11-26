"""L7 independent native-Direct short test at the actual larger matrix size."""
import json
import os
import ctypes
import shutil
import numpy as np
import run_flag_parallel_wind as runner

os.environ['MKL_NUM_THREADS']='8';os.environ['MKL_DYNAMIC']='FALSE'
dirs=['C:/Program Files (x86)/Intel/oneAPI/mkl/latest/redist/intel64',
    'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler']
handles=[os.add_dll_directory(p) for p in dirs]
os.environ['PATH']=';'.join(dirs)+';'+os.environ['PATH']
library=ctypes.WinDLL(dirs[0]+'/mkl_rt.2.dll')
root=runner.ROOT/'output/flag_mkl_fine_probe_20261122'
source=runner.ROOT/'output/flag_to_L9_20261122/L7'
root.mkdir(exist_ok=False);shutil.copytree(source/'assets',root/'assets')
for name in ['local_bundle.bin','scene.json']:shutil.copy2(source/name,root/name)
runner.CACHED_PACKAGE=runner.ROOT/'output/flag_L3_fix_20261118/bunny_ordered_mkl_runtime/src'
runner.STEPS=10
runner.simulate(root,.001,3.,'corners',7,'direct',None,'local')
a=np.load(source/'L7_direct_local/positions.npy',mmap_mode='r')[:11]
b=np.load(root/'L7_direct_local/positions.npy')
stats=json.loads((source/'L7_direct_local/steps.json').read_text())[:10]
meta=json.loads((root/'L7_direct_local/run.json').read_text())
result=dict(steps=10,max_vertex_difference_m=float(np.linalg.norm(a-b,axis=2).max()),
    baseline_step_seconds=sum(s['seconds'] for s in stats),probe_seconds=meta['elapsed_s'],
    acceptance_m=1e-5,production_results_unchanged=True)
result['agreement_passed']=result['max_vertex_difference_m']<result['acceptance_m']
(root/'comparison.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
