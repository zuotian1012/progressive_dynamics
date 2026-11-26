"""Run a common Scene527 folding sequence with native and progressive square cloth."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "reproduction/scene527_55k_73s"
DEMO = ROOT / "dependencies/scene527-0.0.28/src/run_genesis_ipc.py"
BACKEND = ROOT / "outputs/pd_cloth_cuda/native"


def runtime_environment():
    env = os.environ.copy()
    python_paths = [
        ROOT / "dependencies/genesis-world-v0.0.28",
        ROOT / "dependencies/pyuipc-v0.0.28",
    ]
    nvidia = Path(sys.executable).absolute().parent.parent / (
        f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia"
    )
    cuda_libs = sorted(nvidia.glob("*/lib"))
    for key, paths in (("PYTHONPATH", python_paths), ("LD_LIBRARY_PATH", cuda_libs)):
        entries = [str(path) for path in paths]
        if env.get(key):
            entries.append(env[key])
        env[key] = os.pathsep.join(entries)
    env.update(GENESIS_IPC_CLOTH_PARAMETERIZATION="native",
               GENESIS_IPC_CONVERGENCE_LOGS="1", PYTHONUNBUFFERED="1")
    return env


def launch(output, meshes, spec, name, level, args, previous=None):
    directory = output / name
    directory.mkdir(exist_ok=True)
    history = directory / "state.npz"
    metadata = directory / "run.json"
    settings = {"frames": args.frames, "settle_steps": args.settle_steps,
                "virtual_grasp": args.virtual_grasp, "meshes": str(meshes),
                "trajectory": str(args.trajectory) if args.trajectory else None,
                "backend": str(args.backend_dir)}
    if args.resume and metadata.is_file() and history.is_file():
        record = json.loads(metadata.read_text())
        if record.get("settings", settings) != settings:
            raise ValueError(f"{name} was run with different settings")
        return record
    manifest = json.loads((BUNDLE / "bundle.json").read_text())
    preset = [line.strip() for line in (BUNDLE / manifest["preset"]).read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    command = [
        sys.executable, "-B", str(DEMO), "--sim1-root", str(BUNDLE),
        "--trajectory", str(args.trajectory or BUNDLE / manifest["trajectory"]),
        "--robot-urdf", str(BUNDLE / manifest["robot_urdf"]),
        "--shirt-obj", str(meshes / spec["obj"]), *preset,
        "--action-plan-trajectory", "--frames", str(args.frames), "--action-fps", "60",
        "--settle-frames", str(args.settle_steps), "--expected-cloth-vertices", str(spec["vertices"]),
        "--expected-cloth-faces", str(spec["triangles"]), "--initial-shirt-z", "0.93",
        "--cloth-stretch-E", "9800", "--cloth-shear-E", "4", "--cloth-kappa", "1e-7",
        "--cloth-strain-rate", "100", "--cloth-self-friction", "2",
        "--pd-cloth-backend-dir", str(args.backend_dir), "--pd-cloth-state-history", str(history),
        "--output", str(directory / "fold.mp4"), "--no-keyframe-images", "--no-record",
    ]
    if args.virtual_grasp:
        command.remove("--contact-grasp-test")
        command += ["--virtual-grasp", "--grasp-mode", "soft", "--pd-cloth-soft-grasp",
                    "--grasp-points", "1", "--grasp-strength", "100", "--grasp-radius", "0.065"]
    if name != "native_L0" and level < 3:
        command += ["--pd-cloth-proxy", str(meshes / f"proxy_L{level}.json")]
    if previous is not None:
        p = sparse.load_npz(meshes / f"P_adjacent_L{level-1}_to_L{level}.npz")
        with np.load(output / previous / "state.npz") as prior:
            assert prior["dt"] == 1 / 120 and prior["settle_steps"] == args.settle_steps
            previous_delta = prior["delta_v"]
        steps = args.settle_steps + 2 * args.frames
        if previous_delta.shape[0] != steps:
            raise ValueError(f"Expected {steps} predecessor increments, got {len(previous_delta)}")
        velocity_file = directory / "delta_velocity.f64"
        with velocity_file.open("wb") as stream:
            for start in range(0, steps, 64):
                chunk = previous_delta[start:start + 64].reshape(-1, p.shape[1])
                (p @ chunk.T).T.astype("<f8", copy=False).tofile(stream)
        command += ["--pd-cloth-integrator", "progressive", "--pd-cloth-delta-velocity", str(velocity_file)]
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    started = time.perf_counter()
    with (directory / "simulation.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=runtime_environment())
    with np.load(history) as data:
        steps = args.settle_steps + 2 * args.frames
        assert data["x"].shape == data["v"].shape == (steps + 1, spec["vertices"], 3)
        assert np.all(np.isfinite(data["x"])) and np.all(np.isfinite(data["v"]))
        integration_error = float(np.max(np.abs(np.diff(data["x"], axis=0) / data["dt"] - data["v"][1:])))
    result = {"name": name, "level": level, "settings": settings,
              "elapsed_s": time.perf_counter() - started,
              "state_steps": steps,
              "velocity_reconstruction_error_m_s": integration_error}
    scene = json.loads((directory / "fold.json").read_text())
    result.update(action_wall_s=scene["wall_seconds"], grasp=scene["contact_grasp_summary"],
                  virtual_grasp=scene["grasp_summary"])
    metadata.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def trajectory_error(output, meshes, target):
    p = sparse.load_npz(meshes / f"P_adjacent_L{target-1}_to_L{target}.npz")
    with np.load(output / f"pd_L{target-1}/state.npz") as prev:
        coarse_x = prev["x"]
    with np.load(output / f"pd_L{target}/state.npz") as current:
        fine_x = current["x"]
        action_start = int(current["settle_steps"]) + 1
    sum_squared, maximum = 0.0, 0.0
    action_sum_squared = 0.0
    for start in range(0, len(fine_x), 64):
        coarse = coarse_x[start:start + 64].reshape(-1, p.shape[1])
        fine = fine_x[start:start + 64]
        prolonged = (p @ coarse.T).T.reshape(fine.shape)
        squared = np.sum((fine - prolonged)**2, axis=-1)
        sum_squared += float(squared.sum())
        maximum = max(maximum, float(squared.max()))
        action_sum_squared += float(squared[max(0, action_start - start):].sum())
    return {"target_level": target, "rms_m": (sum_squared / (fine_x.size // 3))**0.5,
            "action_rms_m": (action_sum_squared / (fine_x[action_start:].size // 3))**0.5,
            "max_m": maximum**0.5, "final_rms_m": float(np.sqrt(np.mean(squared[-1])))}


def coarse_fold_metrics(output, meshes, name):
    with np.load(meshes / "mesh_L0.npz") as mesh:
        rest, faces = mesh["rest"], mesh["triangles"]
    with np.load(output / name / "state.npz") as state:
        first = int(state["settle_steps"])
        positions = state["x"][first:]
    start_z = positions[0, :, 2]
    end = positions[-1]
    a, b, c = end[faces].transpose(1, 0, 2)
    cross = np.cross(b - a, c - a)
    face_area = np.linalg.norm(cross, axis=1)
    # A flipped panel is evidence of a fold, unlike a small final bounding box alone.
    flipped = float(face_area[cross[:, 2] < 0].sum() / face_area.sum())
    edges = np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1),
                      axis=0)
    l0 = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=-1)
    length = np.linalg.norm(positions[:, edges[:, 1]] - positions[:, edges[:, 0]], axis=-1)
    return {
        "max_vertex_lift_m": float(np.max(positions[:, :, 2] - start_z)),
        "final_downward_face_area_fraction": flipped,
        "final_height_p95_minus_p05_m": float(np.ptp(np.quantile(end[:, 2], [.05, .95]))),
        "edge_strain_abs_p95": float(np.quantile(np.abs(length / l0 - 1), .95)),
    }


def summarize_run(output, record):
    directory = output / record["name"]
    scene = json.loads((directory / "fold.json").read_text())
    with (directory / "simulation.log").open() as log:
        newton_limits = attachments = releases = 0
        for line in log:
            newton_limits += "Newton Iteration Exits with Max Iteration:" in line
            attachments += line.startswith("grasp_attach ")
            releases += line.startswith("grasp_release ")
    return {**record, "action_wall_s": scene["wall_seconds"], "newton_limit_steps": newton_limits,
            "grasp_attachments": attachments, "grasp_releases": releases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meshes", type=Path, default=ROOT / "outputs/pd_square_cloth")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--last-level", type=int, choices=range(4), default=3)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--backend-dir", type=Path, default=BACKEND)
    parser.add_argument("--virtual-grasp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.settle_steps < 0:
        parser.error("Expected positive action frames and nonnegative settling steps")
    args.backend_dir = args.backend_dir.resolve()
    if args.trajectory is not None:
        args.trajectory = args.trajectory.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meshes = args.meshes.resolve()
    specs = json.loads((meshes / "manifest.json").read_text())["levels"]
    records = [launch(output, meshes, specs[0], "native_L0", 0, args)]
    errors = []
    for level in range(args.last_level + 1):
        records.append(launch(output, meshes, specs[level], f"pd_L{level}", level, args,
                              f"pd_L{level-1}" if level else None))
        if level:
            errors.append(trajectory_error(output, meshes, level))
    metrics = {name: coarse_fold_metrics(output, meshes, name) for name in ("native_L0", "pd_L0")}
    report = {"runs": [summarize_run(output, record) for record in records],
              "errors": errors, "coarse_fold_metrics": metrics}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
