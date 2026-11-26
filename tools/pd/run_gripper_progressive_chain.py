"""Run the Scene527 pure-friction first fold on a paper Eq. (5) hierarchy."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.assets.generate_pd_square_cloth import generate
from tools.pd.run_integrator_benchmark import predictor_input, read_p
from tools.pd.run_square_fold import BUNDLE, DEMO, ROOT, runtime_environment


OUTPUT = ROOT / "outputs/pd_gripper_progressive_chain"
MESHES = ROOT / "outputs/pd_square_cloth"
EXTENDED_MESHES = ROOT / "outputs/pd_square_cloth_128"
TRAJECTORY = ROOT / "outputs/pd_inward_transport/center_inward_5pct.npz"
BACKEND = ROOT / "outputs/pd_eq5_cuda/native"
FRAMES = 600
SETTLE = 60
SUBSTEPS = 2
AUDIT_ACTION_FRAME = 300
AUDIT_FRAME = SETTLE + SUBSTEPS * (AUDIT_ACTION_FRAME + 1)
AUDIT_DIRECTORY = f"audit_frame_{AUDIT_FRAME:04d}"
CASES = {0: "pilot_8", 1: "L1_16", 2: "L2_32", 3: "L3_64", 4: "L4_128"}


def case_path(output, level):
    return output / CASES[level]


def mesh_directory(level):
    return EXTENDED_MESHES if level == 4 else MESHES


def base_command(level, directory):
    manifest = json.loads((BUNDLE / "bundle.json").read_text())
    preset = [line.strip() for line in (BUNDLE / manifest["preset"]).read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    cells = 8 << level
    return [
        sys.executable, "-B", str(DEMO), "--sim1-root", str(BUNDLE),
        "--trajectory", str(TRAJECTORY), "--robot-urdf", str(BUNDLE / manifest["robot_urdf"]),
        "--shirt-obj", str(mesh_directory(level) / f"square_L{level}.obj"), *preset,
        "--action-plan-trajectory", "--frames", str(FRAMES), "--action-fps", "60",
        "--substeps", str(SUBSTEPS), "--settle-frames", str(SETTLE),
        "--expected-cloth-vertices", str((cells + 1)**2),
        "--expected-cloth-faces", str(2 * cells**2), "--initial-shirt-z", "0.93",
        "--cloth-stretch-E", "9800", "--cloth-shear-E", "4", "--cloth-kappa", "1e-7",
        "--cloth-strain-rate", "100", "--cloth-self-friction", "2",
        "--pd-cloth-backend-dir", str(BACKEND), "--output", str(directory / "fold.mp4"),
        "--no-keyframe-images",
    ]


def run_case(output, level, source_level):
    directory = case_path(output, level)
    directory.mkdir(parents=True, exist_ok=True)
    if level == 4 and not (EXTENDED_MESHES / "manifest.json").is_file():
        generate(EXTENDED_MESHES, resolutions=(8, 16, 32, 64, 128))
    command = base_command(level, directory) + [
        "--no-record", "--pd-cloth-state-history", str(directory / "state.npz"),
        "--dump-replay-states", str(directory / "replay.npz"),
    ]
    if level < 3:
        command += ["--pd-cloth-proxy", str(MESHES / f"proxy_L{level}.json")]
    if source_level is not None:
        if source_level != level - 1:
            raise ValueError("paper_eq5 requires the immediately preceding mesh level")
        previous = case_path(output, source_level) / "state.npz"
        with np.load(previous, allow_pickle=False) as history:
            if history["settle_steps"] != SETTLE or history["substeps"] != SUBSTEPS:
                raise ValueError("Predecessor settle time or substep count differs")
        path = directory / "predictor.f64"
        predictor_input(previous, read_p(mesh_directory(level), "adjacent", source_level, level),
                        "paper_eq5", SETTLE + SUBSTEPS * FRAMES, path)
        command += ["--pd-cloth-integrator", "paper_eq5", "--pd-cloth-paper-state", str(path)]
        if level == 2:
            command += ["--pd-cloth-audit", str(AUDIT_FRAME), str(output / AUDIT_DIRECTORY / "raw")]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    run(command, directory / "simulation.log")
    with np.load(directory / "replay.npz", allow_pickle=False) as replay:
        np.testing.assert_array_equal(replay["source_frames"], np.arange(FRAMES))


def render(output, level):
    directory = case_path(output, level)
    command = base_command(level, directory) + [
        "--replay-states", str(directory / "replay.npz"), "--record-fps", "60", "--settle-frames", "0",
    ]
    saved_metrics = directory / "fold.json"
    simulation_metrics = saved_metrics.read_bytes()
    try:
        run(command, directory / "render.log")
    finally:
        saved_metrics.write_bytes(simulation_metrics)


def run(command, log_path):
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=runtime_environment(), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("simulate", "render"))
    parser.add_argument("levels", nargs="+", type=int, choices=CASES)
    parser.add_argument("--from-previous", action="store_true",
                        help="Use existing preceding level for the first level listed")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.from_previous and (args.mode != "simulate" or args.levels[0] == 0):
        parser.error("--from-previous requires a simulation starting above level 0")
    if any(target != source + 1 for source, target in zip(args.levels, args.levels[1:])):
        parser.error("List levels in adjacent, increasing order")
    output = args.output.resolve()
    previous = args.levels[0] - 1 if args.from_previous else None
    for level in args.levels:
        print(f"{args.mode} {8 << level}×{8 << level}", flush=True)
        if args.mode == "simulate":
            run_case(output, level, previous)
            previous = level
        else:
            render(output, level)


if __name__ == "__main__":
    main()
