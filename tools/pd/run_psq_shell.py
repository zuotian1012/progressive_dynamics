"""Run a PSQ shell hierarchy with PD 2024 and VelPro in the CUDA solver."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

from tools.assets.prepare_psq_hierarchy import prepare, shell_prolongation
from tools.pd.run_integrator_benchmark import predictor_input, read_mesh, read_p
from tools.pd.run_square_fold import ROOT, runtime_environment
from tools.pd.run_velpro_drape import gear_mesh

DT = 0.01


def curved_shell(path):
    """An irregular triangulation of a one-metre doubly curved open shell."""
    u, v = np.meshgrid(np.linspace(-.5, .5, 15), np.linspace(-.5, .5, 13))
    xy = np.column_stack((u.ravel(), v.ravel()))
    interior = np.max(np.abs(xy), axis=1) < .499
    rng = np.random.default_rng(19)
    xy[interior] += rng.uniform(-.024, .024, size=(interior.sum(), 2))
    x, y = xy.T
    z = .36 + .096 * (1 - 4 * x*x) + .027 * np.sin(2 * np.pi * x) * np.cos(np.pi*y)
    with path.open("w") as file:
        for point in np.column_stack((xy, z)):
            file.write("v " + " ".join(f"{q:.17g}" for q in point) + "\n")
        for face in Delaunay(xy).simplices:
            file.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def load_binding(meshes, kind, level, target):
    intrinsic = read_p(meshes, kind, level, target)[::3, ::3].tocsr()
    with np.load(meshes / f"PSQ_{kind}_L{level}_to_L{target}.npz", allow_pickle=False) as data:
        return intrinsic, data["faces"], data["local"], data["inverse_rest"]


def validate(output, steps, audit, last):
    meshes = output / "hierarchy"
    result = {}
    cases = [("preview", 0), ("direct", last)]
    cases.extend((f"{mode}_L{level}", level) for mode in ("paper_eq5", "velpro")
                 for level in range(1, last + 1))
    for name, level in cases:
        path = output / name
        with np.load(path / "state.npz", allow_pickle=False) as data:
            x, v = data["x"], data["v"]
            if len(x) != steps + 1 or not (np.isfinite(x).all() and np.isfinite(v).all()):
                raise ValueError(f"{name}: incomplete or nonfinite trajectory")
        result[name] = {
            "vertices": x.shape[1],
            "steps": len(x) - 1,
            "max_change_from_rest_m": float(np.max(np.linalg.norm(x - x[:1], axis=-1))),
            "newton_limit_steps": (path / "solver.log").read_text().count("Newton Iteration Exits with Max Iteration:"),
        }
        if level < last:
            binding = load_binding(meshes, "direct", level, last)
            coarse, _ = read_mesh(meshes, level)
            fine, _ = read_mesh(meshes, last)
            rotation = Rotation.from_euler("zyx", [59, -20, 31], degrees=True).as_matrix()
            translation = np.array([.23, -.12, .37])
            expected = fine @ rotation.T + translation
            np.testing.assert_allclose(shell_prolongation(coarse @ rotation.T + translation, *binding),
                                       expected, atol=1e-12, rtol=0)
            folder = path / "audit"
            newton = np.fromfile(folder / "x_newton.f64", dtype="<f8").reshape(-1, 3)
            cuda = np.fromfile(folder / "psq_fine_positions.f64", dtype="<f8").reshape(-1, 3)
            cpu = shell_prolongation(newton, *binding)
            error = float(np.abs(cpu - cuda).max())
            if error > 1e-11:
                raise ValueError(f"{name}: CPU/CUDA prolongation differs by {error} m")
            result[name]["audit_step"] = audit
            result[name]["gpu_cpu_error_m"] = error
    for level in range(1, last + 1):
        parent = "preview" if level == 1 else f"paper_eq5_L{level - 1}"
        with np.load(output / parent / "state.npz", allow_pickle=False) as source:
            positions, velocities = source["x"], source["v"]
        operators = load_binding(meshes, "adjacent", level - 1, level)
        predictor = np.fromfile(output / f"paper_eq5_L{level}" / "predictor.f64", dtype="<f8")
        predictor = predictor.reshape(steps, 2, len(operators[1]), 3)
        index = audit - 1
        np.testing.assert_allclose(predictor[index, 0], shell_prolongation(positions[index], *operators),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(predictor[index, 1], operators[0] @ velocities[index],
                                   rtol=0, atol=1e-12)
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssp", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", type=Path, default=ROOT / "outputs/pd_eq5_cuda/native")
    parser.add_argument("--mesh", type=Path, help="Finest mesh in world metres, initially above the gear")
    parser.add_argument("--face-targets", type=int, nargs="+", default=[156, 64],
                        help="Face counts in fine-to-coarse order")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("Provide at least two steps")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meshes = output / "hierarchy"
    last = len(args.face_targets)
    if not (meshes / "manifest.json").exists():
        fine = args.mesh.resolve() if args.mesh else output / "curved_shell.obj"
        if args.mesh is None:
            curved_shell(fine)
        prepare(fine, args.face_targets, args.ssp, meshes)
    manifest = json.loads((meshes / "manifest.json").read_text())
    if len(manifest["levels"]) != last + 1 or manifest.get("face_targets", args.face_targets) != args.face_targets:
        raise ValueError("Existing hierarchy has different face budgets")
    if args.mesh is not None and Path(manifest["input_mesh"]).resolve() != args.mesh.resolve():
        raise ValueError("Existing hierarchy uses a different source mesh")
    collider = output / "gear.npz"
    vertices, faces = gear_mesh()
    np.savez(collider, vertices=vertices, faces=faces)
    audit = min(args.steps, 25)

    def run(name, level, mode="bdf1", previous=None):
        case = output / name
        case.mkdir(exist_ok=True)
        command = [sys.executable, "-B", "-m", "tools.pd.run_integrator_benchmark",
                   "--worker", "--case", str(case), "--meshes", str(meshes),
                   "--backend", str(args.backend), "--level", str(level), "--steps", str(args.steps),
                   "--dt", str(DT), "--max-iter", "100", "--mode", mode, "--collider", str(collider)]
        if level < last:
            command += ["--proxy", str(meshes / f"proxy_L{level}.json"), "--audit-step", str(audit)]
        if previous:
            path = case / "predictor.f64"
            matrix = read_p(meshes, "adjacent", level - 1, level)
            predictor_input(previous, matrix, mode, args.steps, path, dt=DT,
                            psq=meshes / f"PSQ_adjacent_L{level-1}_to_L{level}.npz")
            command += ["--input", str(path)]
        record = case / "command.json"
        if (case / "state.npz").is_file():
            if json.loads(record.read_text()) != command:
                raise ValueError(f"{case} already has a different run")
        else:
            record.write_text(json.dumps(command, indent=2) + "\n")
            print(name, flush=True)
            with (case / "solver.log").open("w") as stream:
                subprocess.run(command, env=runtime_environment(), stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        return case / "state.npz"

    preview = run("preview", 0)
    for mode in ("paper_eq5", "velpro"):
        previous = preview
        for level in range(1, last + 1):
            previous = run(f"{mode}_L{level}", level, mode, previous)
    run("direct", last)
    validate(output, args.steps, audit, last)


if __name__ == "__main__":
    main()
