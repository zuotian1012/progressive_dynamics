"""Compare Direct IPC and both progressive predictors on two-corner pinned cloth."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import sparse

from tools.assets.generate_pd_square_cloth import generate
from tools.assets.prepare_pd_cloth_proxy import prepare
from tools.pd.run_square_fold import runtime_environment


RESOLUTIONS = (4, 8, 16, 32, 64)
DT = 1 / 120
GRAVITY = [0, 0, -9.81]
MATERIAL = dict(stretch=9800.0, shear=4.0, poisson=0.49, density=800.0,
                thickness=0.0001, strain_rate=100.0, kappa=1e-7)
PREDICTOR_KEYS = {"vel_inc": "pd_cloth_delta_velocity",
                  "vel_inc_gravity": "pd_cloth_delta_velocity",
                  "paper_eq5": "pd_cloth_paper_eq5_state",
                  "velpro": "pd_cloth_velpro_velocity"}


def read_mesh(directory, level):
    with np.load(directory / f"mesh_L{level}.npz", allow_pickle=False) as data:
        return np.ascontiguousarray(data["rest"]), np.ascontiguousarray(data["triangles"], dtype=np.int32)


def read_p(directory, kind, source, target):
    return sparse.load_npz(directory / f"P_{kind}_L{source}_to_L{target}.npz")


def predictor_input(previous, matrix, mode, steps, path, dt=DT, psq=None):
    with np.load(previous, allow_pickle=False) as history:
        if history["dt"] != dt or history["x"].shape != (steps + 1, matrix.shape[1] // 3, 3):
            raise ValueError("Previous trajectory has incompatible time samples or vertex order")
        # File row t predicts the step t -> t+1. Eq. (5) reads v^t;
        # vel_inc reads the difference v^(t+1) - v^t.
        velocity = history["v"]
        if mode == "paper_eq5":
            arrays = (history["x"][:-1], velocity[:-1])
        elif mode == "velpro":
            arrays = (velocity[:-1],)
        else:
            arrays = (np.diff(velocity, axis=0),)
        planes = [(matrix @ array.reshape(steps, -1).T).T for array in arrays]
        if psq is not None and mode == "paper_eq5":
            from tools.assets.prepare_psq_hierarchy import shell_prolongation

            with np.load(psq, allow_pickle=False) as binding:
                support, local, inverse_rest = (binding[key] for key in ("faces", "local", "inverse_rest"))
            intrinsic = matrix[::3, ::3].tocsr()
            for t, coarse in enumerate(arrays[0]):
                planes[0][t] = shell_prolongation(coarse, intrinsic, support, local, inverse_rest).ravel()
        if mode == "vel_inc_gravity":
            # Match the paper_eq5 forcing convention without changing legacy vel_inc.
            planes[0].reshape(steps, -1, 3)[:] += dt * np.asarray(GRAVITY)
        np.stack(planes, axis=1).astype("<f8").tofile(path)


def simulate(args):
    import uipc
    from uipc.constitution import (AffineBodyConstitution, DiscreteShellBending, ElasticModuli2D,
                                   Empty, StrainLimitingBaraffWitkinShell)
    from uipc.core import Engine, FiniteElementStateAccessorFeature, Scene, World

    uipc.init({"module_dir": str(args.backend.resolve())})
    uipc.Timer.disable_all()
    uipc.Logger.set_level(uipc.Logger.Level.Info if args.solver_info else uipc.Logger.Level.Warn)
    config = Scene.default_config()
    config["dt"] = args.dt
    config["gravity"] = [[v] for v in GRAVITY]
    config["contact"]["d_hat"] = 0.0015
    config["newton"].update(max_iter=args.max_iter, velocity_tol=args.velocity_tol, transrate_tol=1e-3)
    config["newton"]["semi_implicit"]["enable"] = 0
    config["linear_system"]["tol_rate"] = args.pcg_tol_rate
    config["linear_system"]["fem_preconditioner"] = args.fem_preconditioner
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)
    engine = Engine("cuda", str(args.case))
    world = World(engine)
    rest, triangles = read_mesh(args.meshes, args.level)
    mesh = uipc.geometry.trimesh(rest, triangles)
    uipc.geometry.label_surface(mesh)
    m = MATERIAL.copy()
    if args.bending_kappa is not None:
        if not np.isfinite(args.bending_kappa) or args.bending_kappa < 0:
            raise ValueError("Bending stiffness must be finite and nonnegative")
        m["kappa"] = args.bending_kappa
    if args.collider:
        m["thickness"] = 0.00007
        with np.load(args.collider, allow_pickle=False) as data:
            obstacle = uipc.geometry.trimesh(data["vertices"], data["faces"].astype(np.int32))
        uipc.geometry.label_surface(obstacle)
        abd = AffineBodyConstitution()
        scene.constitution_tabular().insert(abd)
        abd.apply_to(obstacle, 1e8)
        uipc.view(obstacle.instances().find(uipc.builtin.is_fixed))[:] = 1
        scene.objects().create("gear").geometries().create(obstacle)
    shell = StrainLimitingBaraffWitkinShell()
    shell.apply_to(mesh, stretch_moduli=ElasticModuli2D.youngs_poisson(m["stretch"], m["poisson"]),
                   shear_moduli=ElasticModuli2D.youngs_poisson(m["shear"], m["poisson"]),
                   mass_density=m["density"], thickness=m["thickness"], strain_rate=m["strain_rate"])
    if args.proxy:
        if not args.proxy.is_file():
            raise FileNotFoundError(args.proxy)
        empty = Empty()
        scene.constitution_tabular().insert(empty)
        empty.apply_to(mesh, mass_density=m["density"], thickness=m["thickness"])
        mesh.meta().create("pd_cloth_proxy_asset", str(args.proxy))
        mesh.meta().create("pd_cloth_proxy_kappa", m["kappa"])
        if args.audit_step:
            destination = args.case / "audit"
            destination.mkdir(exist_ok=True)
            mesh.meta().create("pd_cloth_audit_directory", str(destination.resolve()))
            mesh.meta().create("pd_cloth_audit_frame", str(args.audit_step))
    else:
        bend = DiscreteShellBending()
        scene.constitution_tabular().insert(shell)
        scene.constitution_tabular().insert(bend)
        bend.apply_to(mesh, bending_stiffness=m["kappa"])
    stiffness = mesh.meta().find("pd_cloth_proxy_kappa") if args.proxy else mesh.edges().find("bending_stiffness")
    effective_material = {}
    for name, attribute, expected in (
        ("stretch", mesh.triangles().find("lambda"), m["stretch"] * 2 * m["thickness"] / (1 - m["poisson"]**2)),
        ("shear", mesh.triangles().find("mu"), m["shear"] / (2 * (1 + m["poisson"]))),
        ("strain_rate", mesh.triangles().find("strain_rate"), m["strain_rate"]),
        ("kappa", stiffness, m["kappa"]),
        ("thickness", mesh.vertices().find(uipc.builtin.thickness), m["thickness"]),
    ):
        values = np.asarray(attribute.view()).ravel()
        if not np.all(values == values[0]) or not np.isclose(values[0], expected, rtol=1e-14, atol=0):
            raise ValueError(f"Nonuniform or unexpected cloth {name}: expected {expected}")
        effective_material[name] = float(values[0])
    if args.input:
        mesh.meta().create(PREDICTOR_KEYS[args.mode], str(args.input))
    # Shared material-space endpoints on the first edge. Gravity drives motion.
    cells = int(np.sqrt(len(rest))) - 1
    pins = [] if args.collider else [0, cells]
    uipc.view(mesh.vertices().find(uipc.builtin.is_fixed)).reshape(-1)[pins] = 1
    scene.objects().create("cloth").geometries().create(mesh)

    build_start = time.perf_counter()
    world.init(scene)
    build_seconds = time.perf_counter() - build_start
    if not world.is_valid():
        raise RuntimeError("IPC world initialization failed")
    feature = world.features().find(FiniteElementStateAccessorFeature)
    state = feature.create_geometry()
    pos = state.vertices().create(uipc.builtin.position, np.zeros(3))
    vel = state.vertices().create(uipc.builtin.velocity, np.zeros(3))
    x = np.empty((args.steps + 1, *rest.shape))
    v = np.empty_like(x)
    times = np.empty(args.steps)
    readback_seconds = 0.
    sync_seconds = 0.
    captured = 0
    try:
        for t in range(args.steps + 1):
            if t:
                if t == args.profile_step:
                    uipc.Timer.enable_all()
                started = time.perf_counter()
                world.advance()
                before_sync = time.perf_counter()
                world.sync()
                finished = time.perf_counter()
                sync_seconds += finished - before_sync
                times[t - 1] = finished - started
                if t == args.profile_step:
                    profile = uipc.Timer.report_as_json()
                    uipc.Timer.disable_all()
                    (args.case / "profile.json").write_text(json.dumps(profile, indent=2))
            if not world.is_valid() or world.frame() != t:
                raise RuntimeError(f"Expected IPC frame {t}, got {world.frame()}")
            started = time.perf_counter()
            feature.copy_to(state)
            x[t] = pos.view().reshape(-1, 3)
            v[t] = vel.view().reshape(-1, 3)
            readback_seconds += time.perf_counter() - started
            captured = t + 1
            if t and t % 30 == 0:
                print(f"{t}/{args.steps} steps", flush=True)
    except Exception:
        if captured:
            np.savez_compressed(args.case / "partial_state.npz", x=x[:captured], v=v[:captured],
                                dt=args.dt, step_wall_s=times[:captured-1])
        raise
    if not (np.isfinite(x).all() and np.isfinite(v).all()):
        raise RuntimeError("Nonfinite FEM state")
    np.testing.assert_array_equal(x[:, pins], np.broadcast_to(rest[pins], x[:, pins].shape))
    started = time.perf_counter()
    np.savez_compressed(args.case / "state.npz", x=x, v=v, dt=args.dt, step_wall_s=times)
    save_seconds = time.perf_counter() - started
    (args.case / "settings.json").write_text(json.dumps(
        {"config": config, "material": m, "effective_material": effective_material, "cells": cells,
         "initial_pins": rest[pins].tolist(), "mode": args.mode, "build_s": build_seconds,
         "readback_s": readback_seconds, "save_s": save_seconds, "sync_s": sync_seconds,
         "proxy": str(args.proxy) if args.proxy else None,
         "predictor_input": str(args.input) if args.input else None,
         "contact_friction": 0.5, "contact_resistance": 1e9}, indent=2) + "\n")


def launch(directory, meshes, backend, level, mode, steps, previous=None, proxy=False):
    directory.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", "-m", "tools.pd.run_integrator_benchmark",
           "--worker", "--case", str(directory), "--meshes", str(meshes),
           "--backend", str(backend), "--steps", str(steps), "--level", str(level), "--mode", mode]
    if proxy:
        cmd += ["--proxy", str(meshes / f"proxy_L{level}.json")]
    if previous:
        path = directory / "predictor.f64"
        predictor_input(previous, read_p(meshes, "adjacent", level - 1, level), mode, steps, path)
        cmd += ["--input", str(path)]
    (directory / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    env = runtime_environment()
    print(f"Running {directory.name}, {RESOLUTIONS[level]}×{RESOLUTIONS[level]}, {mode}", flush=True)
    with (directory / "solver.log").open("w") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{directory.name} failed. See {directory / 'solver.log'}")
    log = (directory / "solver.log").read_text()
    limits = log.count("Newton Iteration Exits with Max Iteration:")
    if limits:
        print(f"{directory.name}: {limits} steps reached the Newton iteration limit", flush=True)
    return directory / "state.npz"


def lift(matrix, x):
    return (matrix @ x.reshape(len(x), -1).T).T.reshape(len(x), -1, 3)


def rms_curve(x, reference):
    return np.sqrt(np.mean(np.sum((x - reference)**2, axis=-1), axis=1))


def summarize(directory, meshes, cases, steps):
    from matplotlib import pyplot as plt

    curves = {}
    timings = {}
    trajectories = {}
    for name, level, path in cases:
        log = (path.parent / "solver.log").read_text()
        with np.load(path, allow_pickle=False) as data:
            trajectories[name] = data["x"]
            timings[name] = {
                "cells": RESOLUTIONS[level],
                "mean_step_ms": float(data["step_wall_s"].mean() * 1000),
                "median_step_ms": float(np.median(data["step_wall_s"]) * 1000),
                "newton_limit_steps": log.count("Newton Iteration Exits with Max Iteration:"),
                "line_search_limits": log.count("Line Search Exits with Max Iteration:"),
            }
    last = len(RESOLUTIONS) - 1
    reference = trajectories["direct_fine"]
    for name, level, _ in cases:
        prolonged = lift(read_p(meshes, "direct", level, last), trajectories[name])
        curves[f"{name}_vs_direct_fine"] = rms_curve(prolonged, reference)
    modes = [mode for mode in PREDICTOR_KEYS if f"{mode}_L{last}" in trajectories]
    styles = {"vel_inc": "--", "vel_inc_gravity": ":", "paper_eq5": "-", "velpro": "-."}
    for mode in modes:
        for level in range(1, last + 1):
            source_name = "proxy_L0" if level == 1 else f"{mode}_L{level-1}"
            coarse = lift(read_p(meshes, "adjacent", level - 1, level), trajectories[source_name])
            curves[f"{mode}_L{level}_vs_L{level-1}"] = rms_curve(trajectories[f"{mode}_L{level}"], coarse)
    seconds = np.arange(steps + 1) * DT
    keys = list(curves)
    np.savetxt(directory / "rms_curves.csv", np.column_stack((seconds, *curves.values())),
               delimiter=",", header="time_s," + ",".join(keys), comments="")
    report = {key: {"rms_mm": float(np.sqrt(np.mean(value[1:]**2)) * 1000),
                    "max_mm": float(value.max() * 1000),
                    "last_mm": float(value[-1] * 1000)} for key, value in curves.items()}
    (directory / "comparison.json").write_text(json.dumps(
        {"steps": steps, "dt": DT, "cells": RESOLUTIONS, "timings": timings, "errors": report},
        indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    for level in range(1, last + 1):
        for mode in modes:
            axes[0].plot(seconds, 1000 * curves[f"{mode}_L{level}_vs_L{level-1}"],
                         styles[mode], color=f"C{level-1}", label=f"{mode} L{level-1} → L{level}")
    for name, ls in (("direct_coarse", ":"), ("direct_8", "-."), ("proxy_L0", "-")):
        axes[1].plot(seconds, 1000 * curves[f"{name}_vs_direct_fine"], ls, label=name)
    for level in range(1, last + 1):
        for mode in modes:
            axes[1].plot(seconds, 1000 * curves[f"{mode}_L{level}_vs_direct_fine"],
                         styles[mode], color=f"C{level+2}", label=f"{mode} L{level}")
    for ax, title in zip(axes, ("Adjacent-level trajectories", "Compared with Direct IPC 64×64")):
        ax.set(title=title, xlabel="Time (s)", ylabel="Position RMS (mm)")
        ax.set_yscale("symlog", linthresh=1)
        ax.legend(ncols=2, fontsize=8)
        ax.grid(alpha=.25)
    fig.savefig(directory / "rms_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--add-gravity-control", action="store_true",
                        help="Add the forcing-matched control to a completed baseline directory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--solver-info", action="store_true", help="Record IPC iteration and contact diagnostics in worker mode")
    parser.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--meshes", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--level", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("bdf1", *PREDICTOR_KEYS), default="bdf1")
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--proxy", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--velocity-tol", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--pcg-tol-rate", type=float, default=1e-5)
    parser.add_argument("--bending-kappa", type=float, help="Override cloth bending stiffness")
    parser.add_argument("--fem-preconditioner", choices=("diag", "mas"), default="diag")
    parser.add_argument("--profile-step", type=int, help="Collect synchronized CUDA stage timings at one step")
    parser.add_argument("--collider", type=Path, help="Worker fixed closed triangular collider in NPZ")
    parser.add_argument("--audit-step", type=int, help="Worker IPC step for proxy assembly diagnostics")
    args = parser.parse_args()
    if args.worker:
        simulate(args)
        return
    if args.output is None or args.steps < 1 or args.dt != DT or args.max_iter != 300 or args.collider:
        parser.error("--output and a positive --steps are required")
    output = args.output.resolve()
    meshes = output / "hierarchy"
    backend = args.backend.resolve()
    last = len(RESOLUTIONS) - 1

    def run(name, level, mode="bdf1", previous=None, proxy=False):
        return name, level, launch(output / name, meshes, backend, level, mode, args.steps,
                                   previous=previous, proxy=proxy)

    legacy_modes = ("vel_inc", "paper_eq5")
    if args.add_gravity_control:
        names = [("direct_coarse", 0), ("direct_8", 1), ("direct_fine", last), ("proxy_L0", 0)]
        names += [(f"{mode}_L{level}", level) for mode in legacy_modes for level in range(1, last + 1)]
        cases = [(name, level, output / name / "state.npz") for name, level in names]
        for _, _, state in cases:
            command = json.loads(state.with_name("command.json").read_text())
            if command[command.index("--backend") + 1] != str(backend):
                raise ValueError(f"Backend differs from the completed baseline: {state}")
            with np.load(state, allow_pickle=False) as data:
                if data["dt"] != DT or len(data["x"]) != args.steps + 1:
                    raise ValueError(f"Time axis differs from the completed baseline: {state}")
        if (output / "vel_inc_gravity_L1").exists():
            raise FileExistsError("Gravity control output already exists")
        base = cases[3]
        modes = ("vel_inc_gravity",)
    else:
        output.mkdir(parents=True, exist_ok=False)
        generate(meshes, RESOLUTIONS)
        for level in range(last):
            prepare(meshes, level, meshes / f"proxy_L{level}.json")
        cases = [run("direct_coarse", 0), run("direct_8", 1), run("direct_fine", last)]
        base = run("proxy_L0", 0, proxy=True)
        cases.append(base)
        modes = legacy_modes
    for mode in modes:
        prior = base
        for level in range(1, last + 1):
            prior = run(f"{mode}_L{level}", level, mode, prior[2], proxy=level < last)
            cases.append(prior)
    summarize(output, meshes, cases, args.steps)


if __name__ == "__main__":
    main()
