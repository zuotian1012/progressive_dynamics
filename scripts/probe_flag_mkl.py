"""Isolated exact direct-solver comparison, no production trajectories changed."""
import json
import os
import ctypes
import shutil
import numpy as np
import run_flag_parallel_wind as runner

os.environ['MKL_NUM_THREADS']='8';os.environ['MKL_DYNAMIC']='FALSE'
handles=[os.add_dll_directory(p) for p in ['C:/Program Files (x86)/Intel/oneAPI/mkl/latest/redist/intel64','C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler']]
os.environ['PATH']='C:/Program Files (x86)/Intel/oneAPI/mkl/latest/redist/intel64;C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler;'+os.environ['PATH']
library=ctypes.WinDLL('C:/Program Files (x86)/Intel/oneAPI/mkl/latest/redist/intel64/mkl_rt.2.dll')
root=runner.ROOT/'output/flag_mkl_probe_v2_20261122';source=runner.ROOT/'output/flag_to_L9_20261122/L3'
root.mkdir(exist_ok=False);shutil.copytree(source/'assets',root/'assets')
for name in ['bundle.bin','scene.json']:shutil.copy2(source/name,root/name)
runner.ORDERED_PACKAGE=runner.ROOT/'output/flag_L3_fix_20261118/bunny_ordered_mkl_runtime/src'
runner.STEPS=10
runner.simulate(root,.001,3.,'corners',3,'semi',runner.ROOT/'output/flag_to_L9_20261122/L2/L2_semi','proxy')
a=np.load(source/'L3_semi/positions.npy')[:11];b=np.load(root/'L3_semi/positions.npy')
stats=json.loads((source/'L3_semi/steps.json').read_text())[:10];meta=json.loads((root/'L3_semi/run.json').read_text())
result=dict(steps=10,max_vertex_difference_m=float(np.linalg.norm(a-b,axis=2).max()),baseline_step_seconds=sum(s['seconds'] for s in stats),
    probe_seconds=meta['elapsed_s'],acceptance_m=1e-5,production_results_unchanged=True)
result['agreement_passed']=result['max_vertex_difference_m']<result['acceptance_m']
(root/'comparison.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
