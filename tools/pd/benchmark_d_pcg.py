"""Paired same-clock PCG tolerance test on the saved D VelPro chain."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import math

import numpy as np

from tools.pd.run_integrator_benchmark import predictor_input, read_p
from tools.pd.run_square_fold import runtime_environment
from tools.pd.run_velpro_drape import DT

LEVELS = ("proxy_coarse", "velpro_middle", "velpro_fine")


def held_velpro_input(previous, matrix, steps, dt, path):
    """Hold the last available parent velocity at each child step's start."""
    with np.load(previous, allow_pickle=False) as state:
        v = state["v"]
        parent_steps = len(v) - 1
        if not math.isclose(parent_steps * float(state["dt"]), steps * dt, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Parent and child must cover the same physical interval")
        if v.shape[1:] != (matrix.shape[1] // 3, 3):
            raise ValueError("Parent vertex order is incompatible with prolongation")
        indices = np.arange(steps, dtype=np.int64) * parent_steps // steps
        transferred = matrix @ v[indices].reshape(steps, -1).T
    transferred.T.astype("<f8").tofile(path)
    np.save(path.with_name("parent_step_indices.npy"), indices)


def run(source, target, name, steps, rate, previous=None, profile=False, *, dt=DT, velocity_tol=1e-3,
        multirate=False, preconditioner="diag", backend=None):
    target.mkdir(parents=True, exist_ok=True)
    cmd = json.loads((source / name / "command.json").read_text())
    cmd[0] = sys.executable
    cmd[cmd.index("--case") + 1] = str(target)
    cmd[cmd.index("--steps") + 1] = str(steps)
    cmd[cmd.index("--dt") + 1] = str(dt)
    if velocity_tol != 1e-3:
        cmd.extend(["--velocity-tol", str(velocity_tol)])
    cmd.extend(["--pcg-tol-rate", str(rate)])
    if preconditioner != "diag":
        cmd.extend(["--fem-preconditioner", preconditioner])
    if backend is not None:
        cmd[cmd.index("--backend")+1] = str(backend)
    if profile:
        cmd.extend(["--profile-step", str(steps)])
    if previous is not None:
        path = target / "predictor.f64"
        cmd[cmd.index("--input") + 1] = str(path)
    command_file = target / "command.json"
    if command_file.is_file() and json.loads(command_file.read_text()) != cmd:
        raise ValueError(f"Different settings already exist in {target}")
    if all((target / name).is_file() for name in ("state.npz", "settings.json", "wall_seconds.json", "transfer_seconds.json")):
        return
    transfer_s = 0.
    start = time.perf_counter()
    if previous is not None:
        level = LEVELS.index(name)
        before = time.perf_counter()
        matrix = read_p(source / "hierarchy", "adjacent", level-1, level)
        if multirate:
            held_velpro_input(previous / "state.npz", matrix, steps, dt, path)
        else:
            predictor_input(previous / "state.npz", matrix, "velpro", steps, path, dt)
        transfer_s = time.perf_counter() - before
    command_file.write_text(json.dumps(cmd, indent=2))
    with (target / "solver.log").open("w") as stream:
        subprocess.run(cmd, env=runtime_environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - start
    (target / "wall_seconds.json").write_text(json.dumps(elapsed))
    (target / "transfer_seconds.json").write_text(json.dumps(transfer_s))
    print(target, f"{elapsed:.2f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("phase", nargs="?", default="daily", choices=["daily", "pilot", "profile-levels", "full", "fine-only", "frequency", "mixed-frequency", "mas-fine", "mas-all"])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.phase == "daily":
        first_fast_level = len(LEVELS) - max(1, len(LEVELS) // 3)
        previous = None
        for level, name in enumerate(LEVELS):
            fine = level >= first_fast_level
            hz = 50 if fine else 100
            target = output / ("100-50Hz" if fine else "100Hz") / name
            run(source, target, name, 120 * hz // 100, 1e-3 if fine else 1e-5,
                previous, dt=1/hz, velocity_tol=1e-5 * hz,
                multirate=level == first_fast_level and previous is not None)
            previous = target
        return
    if args.phase in ("mas-fine", "mas-all"):
        backend = output / "proxy_mas_backend"
        if not (backend / "provenance.json").is_file():
            raise FileNotFoundError(f"Missing verified build in {backend}")
        if args.phase == "mas-fine":
            run(source, output / "100Hz-mas" / LEVELS[-1], LEVELS[-1], 120, 1e-3,
                output / "100Hz" / LEVELS[1], preconditioner="mas", backend=backend)
        else:
            previous = None
            for level, name in enumerate(LEVELS):
                target = output / "100Hz-mas-all" / name
                run(source, target, name, 120, 1e-3 if level == 2 else 1e-5, previous,
                    preconditioner="mas", backend=backend)
                previous = target
        return
    if args.phase == "mixed-frequency":
        for parent_hz, fine_hz in ((100, 50), (50, 100)):
            run(source, output / f"{parent_hz}-{fine_hz}Hz" / LEVELS[-1], LEVELS[-1],
                120 * fine_hz // 100, 1e-3, output / f"{parent_hz}Hz" / LEVELS[1],
                dt=1/fine_hz, velocity_tol=1e-5*fine_hz, multirate=True)
        return
    if args.phase == "frequency":
        for level, name in enumerate(LEVELS):
            for hz in ((50, 100) if level % 2 else (100, 50)):
                parent = output / f"{hz}Hz"
                previous = parent / LEVELS[level-1] if level else None
                run(source, parent / name, name, 120 * hz // 100, 1e-3 if level == 2 else 1e-5,
                    previous, dt=1/hz, velocity_tol=1e-5 * hz)
        return
    if args.phase == "pilot":
        # Same saved precursor and predictor for both runs; crosses initial impact.
        for label, rate in (("diag_baseline", 1e-5), ("diag_candidate", 1e-3)):
            run(source, output / label, LEVELS[-1], 32, rate, profile=True)
        return
    if args.phase == "profile-levels":
        for name in LEVELS[:2]:
            run(source, output / f"profile_{name}", name, 32, 1e-5, profile=True)
        return
    if args.phase == "fine-only":
        run(source, output / "fine_only" / LEVELS[-1], LEVELS[-1], 120, 1e-3,
            output / "base" / LEVELS[1])
        return
    # Alternate the run order by level, keep the two physical histories independent.
    for level, name in enumerate(LEVELS):
        variants = [("base", 1e-5), ("candidate", 1e-3)]
        if level % 2:
            variants.reverse()
        for label, rate in variants:
            parent = output / label
            previous = parent / LEVELS[level-1] if level else None
            run(source, parent / name, name, 120, rate, previous)


if __name__ == "__main__":
    main()
