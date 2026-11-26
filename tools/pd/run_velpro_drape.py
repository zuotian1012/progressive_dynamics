"""Recreate Dynamic Drape with a configurable LOD hierarchy and three integrators."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tools.assets.generate_pd_square_cloth import generate
from tools.assets.prepare_pd_cloth_proxy import prepare
from tools.pd.run_integrator_benchmark import predictor_input, read_p
from tools.pd.run_square_fold import ROOT, runtime_environment


CELLS = (3, 6, 12, 24, 48, 96)
DT = 0.01
BACKEND = ROOT / "outputs/pd_eq5_cuda/native"


def gear_mesh(teeth=12):
    angles = np.arange(4 * teeth) * (2 * np.pi / (4 * teeth))
    radii = np.tile([0.195, 0.25, 0.25, 0.195], teeth)
    xy = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
    count = len(xy)
    top, bottom = 0.26, 0.12
    vertices = np.vstack((np.column_stack((xy, np.full(count, top))),
                          np.column_stack((xy, np.full(count, bottom))),
                          [0, 0, top], [0, 0, bottom]))
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.extend(((2 * count, i, j), (2 * count + 1, count + j, count + i),
                      (i, count + i, j), (j, count + i, count + j)))
    return vertices, np.array(faces, dtype=np.int32)


def run_case(output, name, resolution, mode, args, previous=None, proxy=False):
    directory = output / name
    directory.mkdir(exist_ok=True)
    meshes = output / "hierarchy"
    args_file = directory / "command.json"
    cmd = [sys.executable, "-B", "-m", "tools.pd.run_integrator_benchmark", "--worker",
           "--case", str(directory), "--meshes", str(meshes), "--backend", str(args.backend),
           "--steps", str(args.steps), "--level", str(resolution), "--mode", mode,
           "--dt", str(DT), "--max-iter", str(args.max_iter),
           "--collider", str(output / "gear.npz")]
    if proxy:
        cmd += ["--proxy", str(meshes / f"proxy_L{resolution}.json")]
    solver_info = getattr(args, "solver_info", False)
    if solver_info:
        cmd.append("--solver-info")
    if previous:
        predictor_path = directory / "predictor.f64"
        cmd += ["--input", str(predictor_path)]
    has_outputs = all((directory / name).is_file() for name in ("state.npz", "settings.json"))
    has_timing = not solver_info or (directory / "wall_seconds.json").is_file()
    if args_file.exists() and has_outputs:
        if json.loads(args_file.read_text()) != cmd:
            raise ValueError(f"{name} was run with different settings")
        if has_timing:
            return directory / "state.npz"
    if previous:
        matrix = read_p(meshes, "adjacent", resolution - 1, resolution)
        predictor_input(previous, matrix, mode, args.steps, predictor_path, dt=DT)
    args_file.write_text(json.dumps(cmd, indent=2) + "\n")
    cells = getattr(args, "cells", CELLS)[resolution]
    print(f"{name}: {cells}×{cells}", flush=True)
    start = time.perf_counter()
    with (directory / "solver.log").open("w") as stream:
        subprocess.run(cmd, env=runtime_environment(), stdout=stream, stderr=subprocess.STDOUT, check=True)
    if solver_info:
        (directory / "wall_seconds.json").write_text(json.dumps(time.perf_counter() - start) + "\n")
    return directory / "state.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=BACKEND)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--cells", type=int, nargs="+", default=CELLS)
    parser.add_argument("--modes", nargs="+", choices=("paper_eq5", "velpro"),
                        default=("paper_eq5", "velpro"))
    parser.add_argument("--skip-direct", action="store_true",
                        help="Use a separately run direct reference when analyzing this run")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--smoke", action="store_true", help="Run a short coarse preview and both predictors at level 1")
    args = parser.parse_args()
    if args.steps < 2 or args.max_iter < 1 or len(args.cells) < 2:
        parser.error("Require at least two steps, at least two levels and a positive Newton limit")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meshes = output / "hierarchy"
    if not (output / "gear.npz").exists():
        generate(meshes, args.cells, side_length=1., center=(0.032, -0.018, 0.40))
        for level in range(len(args.cells) - 1):
            prepare(meshes, level, meshes / f"proxy_L{level}.json")
        vertices, faces = gear_mesh()
        np.savez(output / "gear.npz", vertices=vertices, faces=faces)
    manifest = json.loads((meshes / "manifest.json").read_text())
    if [entry["cells_per_side"] for entry in manifest["levels"]] != list(args.cells):
        raise ValueError("Saved hierarchy has different resolutions")
    last = len(args.cells) - 1
    base = run_case(output, "preview_L0", 0, "bdf1", args, proxy=True)
    if not args.smoke and not args.skip_direct:
        run_case(output, "direct_fine", last, "bdf1", args)
    for mode in args.modes:
        previous = base
        for level in range(1, 2 if args.smoke else last + 1):
            previous = run_case(output, f"{mode}_L{level}", level, mode, args, previous, proxy=level < last)


if __name__ == "__main__":
    main()
