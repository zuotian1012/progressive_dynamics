"""Compare a restarted PD suffix with the corresponding continuous debug run."""
import json
from pathlib import Path
import subprocess

import numpy as np

from tools.pd.run_scene527_five_stages import DEMO, OUT, runtime_environment, write


def check(case):
    result=case/'resume'
    result.mkdir(exist_ok=True)
    cmd=json.loads((case/'command.json').read_text())
    cmd=cmd[:cmd.index('--save-checkpoint-source-frames')]
    cmd[cmd.index('--output')+1]=str(result)
    cmd[cmd.index('--output',cmd.index('--')+1)+1]=str(result/'fold.mp4')
    cmd[cmd.index('--pd-cloth-state-history')+1]=str(result/'state.npz')
    start=4
    cmd+=['--pd-cloth-resume-start-row',str(start),'--load-third-fold-checkpoint',str(case/'checkpoints/source_000000.pkl')]
    write(result/'command.json',cmd)
    (result/'physics_source_snapshot.py').write_bytes(DEMO.read_bytes())
    with (result/'physics.log').open('w') as log:
        subprocess.run(cmd,env=runtime_environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    with np.load(case/'state.npz') as reference, np.load(result/'state.npz') as resumed:
        errors={}
        for field in ('x','v'):
            continuous=reference[field][start:]
            actual=resumed[field]
            np.testing.assert_equal(continuous.shape,actual.shape)
            np.testing.assert_allclose(actual[0],continuous[0],atol=1e-12,rtol=0)
            errors[field]=float(np.max(np.linalg.norm(actual-continuous,axis=2)))
        write(result/'comparison.json',errors)
        # Both tolerances express the native 0.01 m/s motion checker, in x and v.
        if errors['x']>0.01/120 or errors['v']>0.01:
            raise RuntimeError(f'Checkpoint continuation mismatch: {errors}')
        print(case.name,errors,flush=True)


if __name__=='__main__':
    for name in ['L3','L4']:
        check(OUT/'debug'/name)
