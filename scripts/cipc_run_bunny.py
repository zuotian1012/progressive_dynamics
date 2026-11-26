"""Scene/logging adapter for the UNMODIFIED upstream native C-IPC solver.

Run in Ubuntu/WSL. No progressive module, ipctk, or custom Newton code is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('/opt/cipc-baseline/source'))
    p.add_argument('--build', type=Path, default=Path('/opt/cipc-baseline/build'))
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    inputs = a.inputs.resolve()
    input_manifest = json.loads((inputs/'manifest.json').read_text())
    for name, meta in input_manifest['meshes'].items():
        if hashlib.sha256((inputs/(name+'.obj')).read_bytes()).hexdigest() != meta['obj_sha256']:
            raise RuntimeError(f'Input hash mismatch: {name}')
    if input_manifest['meshes']['cloth']['vertices'] != 90000:
        raise ValueError('This driver requires the actual 90K finest mesh')
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(a.threads)
    os.environ['OMP_PROC_BIND'] = 'false'
    sys.path.insert(0, str(a.source.resolve()/'Python'))
    sys.path.insert(0, str(a.build.resolve()))
    import JGSL
    import Drivers
    from JGSL import Vector3d, Vector2d
    binary = Path(JGSL.__file__).resolve()
    state = {'backend': 'upstream native Codim-IPC C++ / JGSL',
        'source_commit': '9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a',
        'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'cloth_vertices': 90000, 'steps': a.steps, 'completed_steps': 0, 'completed': False,
        'status': 'initializing', 'dt': .01, 'requested_physical_time': a.steps*.01,
        'threads': a.threads, 'input_manifest': input_manifest,
        'material': {'density':200., 'young':30000., 'poisson':.3, 'thickness':.00007},
        'strain': {'limit':1.068, 'activation':1.034, 'initial_stiffness':1000.},
        'contact': {'friction':.3, 'activation_distance_parameter':.001, 'offset':.00007,
                    'stiffness_policy':'native Initialize_OIPC default mass-based initialization and native updates'},
        'newton': {'tolerance':1e-4, 'norm':'native RMS velocity decrement; maximum of last three iterations',
                   'iteration_cap':'upstream default (no injected cap)', 'preview_early_stop':False},
        'limitations': ['Reconstructed PD++ bunny inputs, not authenticated author scene',
            'Material and contact activation are declared local assumptions',
            '4 s is chosen observation duration, not recovered author duration',
            'Native contact stiffness and solver behavior differ from previous Python implementation'],
        'physics_seconds':0., 'wall_seconds':0.}
    atomic_json(out/'run.json', state)
    start = time.perf_counter()
    raw = out/'raw'
    raw.mkdir()
    class NativeScene(Drivers.FEMDiscreteShellBase):
        def register_logger(self):
            self.output_folder = str(raw)+'/'
            (raw/'log').mkdir(exist_ok=True)
            JGSL.Set_Parameter('Basic.log_folder', str(raw/'log')+'/')
    # Upstream SimulationBase uses argv to construct its default log path.
    # Keep CLI arguments out of that path; subclass changes logging only.
    sys.argv = ['cipc_native_bunny']
    os.chdir(out)
    (out/'output').mkdir(exist_ok=True)
    try:
        sim = NativeScene('double', 3)
        zero = Vector3d(0,0,0)
        axis = Vector3d(1,0,0)
        sim.add_shell_3D(str(inputs/'cloth.obj'), zero, zero, axis, 0)
        bunny_range = sim.add_shell_3D(str(inputs/'bunny.obj'), zero, zero, axis, 0)
        sim.set_DBC_with_range(Vector3d(-.1,-.1,-.1), Vector3d(1.1,1.1,1.1),
                               zero, zero, axis, 0, bunny_range)
        sim.gravity = Vector3d(0,0,-9.81)
        sim.dt = sim.frame_dt = .01
        sim.frame_num = a.steps
        sim.PNTol = 1e-4
        sim.withCollision = True
        sim.mu = .3
        sim.initialize(200., 30000., .3, .00007, 0)
        sim.s = Vector2d(1.068, 0)
        sim.sHat = Vector2d(1.034, 1)
        sim.kappa_s = Vector2d(1000., 0)
        sim.initialize_OIPC(.001, .00007)
        state['contact']['initial_kappa'] = [sim.kappa[i] for i in range(3)]
        state['contact']['dHat2'] = sim.dHat2
        state['setup_seconds'] = time.perf_counter()-start
        sim.write(0)
        state['status'] = 'running'
        atomic_json(out/'run.json', state)
        residual_path = raw/'residual.txt'
        for step in range(1, a.steps+1):
            offset = residual_path.stat().st_size if residual_path.exists() else 0
            pn0 = sim.PNIterCount
            tick = time.perf_counter()
            sim.current_frame = step
            sim.advance_one_time_step(.01)
            seconds = time.perf_counter()-tick
            with residual_path.open() as stream:
                stream.seek(offset)
                rows = [[float(v) for v in line.split()] for line in stream if line.strip()]
            last3 = [row[-1] for row in rows[-3:]]
            if len(last3) != 3 or not all(0 <= value <= 1e-4 for value in last3):
                raise RuntimeError(f'Native step returned without three converged residuals: {last3}')
            if JGSL.Get_Parameter('Terminate', False):
                raise RuntimeError('Native solver set Terminate')
            sim.write(step)
            JGSL.TIMER_FLUSH(step, a.steps, .01, .01)
            record = {'step':step, 'physical_time':step*.01, 'seconds':seconds,
                'newton_iterations':sim.PNIterCount-pn0, 'last_three_rms_velocity_decrements':last3,
                'converged':True, 'preview_stop':False, 'frame':f'raw/shell{step}.obj'}
            with (out/'steps.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            state['completed_steps'] = step
            state['physics_seconds'] += seconds
            state['wall_seconds'] = time.perf_counter()-start
            state['physical_time'] = step*.01
            atomic_json(out/'run.json', state)
            print(f'CIPC_ACCEPTED {step}/{a.steps} native_Newton={record["newton_iterations"]} seconds={seconds:.3f}', flush=True)
        state.update(completed=True, status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        state['wall_seconds'] = time.perf_counter()-start
        atomic_json(out/'run.json', state)

if __name__ == '__main__':
    main()
