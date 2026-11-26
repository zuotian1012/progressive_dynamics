import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from .energy import ShellEnergy, floor_barrier
from .mesh import grid, lumped_mass, prolongation


METHODS = ("independent", "semi", "full", "velpro")


def predictor(method, level, t, trajectories, maps, h, initial_velocities):
    current = trajectories[level][t]
    if method == "independent" or level == 0:
        prev = trajectories[level][t-1] if t else current - h * initial_velocities[level]
        return 2 * current - prev
    lower = trajectories[level-1][t]
    prev_lower = trajectories[level-1][t-1] if t else lower - h * initial_velocities[level-1]
    p = maps[level-1]
    if method == "semi" or (method == "full" and level == 1):
        return p @ (2 * lower - prev_lower)
    if method == "full":
        older = trajectories[level-2][t-1] if t else trajectories[level-2][0] - h * initial_velocities[level-2]
        return p @ (2 * lower - maps[level-2] @ older)
    if method == "velpro":
        return current + p @ (lower - prev_lower)
    raise ValueError(method)


@dataclass
class Level:
    mesh: object
    mass: np.ndarray
    pins: np.ndarray
    initial: np.ndarray
    to_fine: object


class ClothSystem:
    def __init__(self, config):
        self.config = config
        mat = config["material"]
        segs = config["segments"]
        if not segs or config["dt"] <= 0 or config["steps"] < 2:
            raise ValueError("Require nonempty levels, positive dt, and at least two steps")
        if mat["young"] <= 0 or mat["density"] <= 0 or mat["thickness"] <= 0 or not (-1 < mat["poisson"] < 0.5):
            raise ValueError("Invalid elastic material")
        if config.get("pins", "edge") not in ("none", "edge", "corners", "left_edge"):
            raise ValueError("Unknown pin mode")
        if config.get("orientation", "horizontal") not in ("horizontal", "vertical"):
            raise ValueError("Unknown cloth orientation")
        if config.get("consistency_weight", 0) < 0:
            raise ValueError("consistency_weight must be nonnegative")
        if any(b <= a for a, b in zip(segs, segs[1:])):
            raise ValueError("segments must increase strictly")
        meshes = [grid(s) for s in segs]
        self.maps = [prolongation(a, b) for a, b in zip(meshes, meshes[1:])]
        self.shell = ShellEnergy(meshes[-1], mat["young"], mat["poisson"], mat["thickness"])
        self.levels = []
        self.velocities = []
        for mesh in meshes:
            x = mesh.rest.copy()
            x[:, 2] += config["initial_height"]
            # An affine initial tilt is exactly representable at every level.
            x[:, 2] += config.get("initial_tilt", 0) * mesh.uv[:, 1]
            if config.get("orientation") == "vertical":
                # Rigidly rotate the rest plane about x; keep rest lengths unchanged.
                x = np.column_stack((mesh.uv[:, 0]-0.5, np.zeros(len(mesh.uv)),
                                     config["initial_height"] + mesh.uv[:, 1]))
            pin_mode = config.get("pins", "edge")
            pins = np.isclose(mesh.uv[:, 1], 0) if pin_mode == "edge" else np.zeros(len(x), bool)
            if pin_mode == "corners":
                pins = np.isclose(mesh.uv[:, 1], 0) & (np.isclose(mesh.uv[:, 0], 0) | np.isclose(mesh.uv[:, 0], 1))
            elif pin_mode == "left_edge":
                pins = np.isclose(mesh.uv[:, 0], 0)
            mass = lumped_mass(mesh, mat["density"] * mat["thickness"])
            self.levels.append(Level(mesh, mass, pins, x, prolongation(mesh, meshes[-1])))
            self.velocities.append(np.zeros_like(x))
        floor = config.get("floor")
        if floor and any(np.min(l.initial[:, 2]) <= floor["height"] for l in self.levels):
            raise ValueError("Initial cloth must lie above the floor")

    def potential(self, level, x, next_time):
        l = self.levels[level]
        e, g = self.shell.value_grad(l.to_fine @ x)
        g = l.to_fine.T @ g
        acceleration = np.asarray(self.config.get("gravity", [0, 0, -9.81]), float)
        wind = self.config.get("wind", {})
        amplitude = np.asarray(wind.get("acceleration", [0, 0, 0]), float)
        # Prescribed body acceleration for a reproducible forcing; not aerodynamic drag.
        phase = wind.get("spatial_phase", 0) * l.mesh.uv[:, 0]
        wave = np.sin(2 * np.pi * wind.get("frequency", 1) * next_time - phase)
        acceleration = (acceleration + np.asarray(wind.get("mean_acceleration", [0, 0, 0]), float)
                        + wave[:, None] * amplitude)
        force = l.mass[:, None] * acceleration
        e -= float(np.sum(force * x))
        g -= force
        floor = self.config.get("floor")
        if floor:
            eb, gb = floor_barrier(x, l.mass, **floor)
            e += eb
            g += gb
        return e, g

    def step(self, level, previous, xhat, next_time, target=None):
        l, h = self.levels[level], self.config["dt"]
        free = ~l.pins
        settings = self.config["solver"]
        weight = self.config.get("consistency_weight", 0) if target is not None else 0
        base = l.initial.copy()

        def objective(y):
            x = base.copy()
            x[free] = y.reshape(-1, 3)
            e, g = self.potential(level, x, next_time)
            dx = x - xhat
            e += np.sum(l.mass[:, None] * dx**2) / (2 * h*h)
            g += l.mass[:, None] * dx / (h*h)
            if weight:
                delta = x - target
                e += weight * np.sum(l.mass[:, None] * delta**2)
                g += 2 * weight * l.mass[:, None] * delta
            return e, g[free].ravel()

        bounds = None
        floor = self.config.get("floor")
        if floor:
            bounds = [(None, None), (None, None), (floor["height"] + 1e-7, None)] * int(free.sum())
        start = time.perf_counter()
        result = minimize(objective, previous[free].ravel(), jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": settings["maxiter"], "gtol": settings["gtol"],
                                   "ftol": settings["ftol"], "maxls": 40, "maxcor": 20})
        x = base.copy()
        x[free] = result.x.reshape(-1, 3)
        finite = bool(np.isfinite(x).all() and np.isfinite(result.fun))
        report = {"success": bool(result.success), "finite": finite, "iterations": int(result.nit),
                  "gradient_inf": float(np.max(np.abs(result.jac))), "message": str(result.message),
                  "seconds": time.perf_counter() - start}
        if not finite:
            raise RuntimeError(f"Non-finite state: level {level}, time {next_time}: {report}")
        return x, report

    def simulate(self, method, coarse=None, coarse_reports=None):
        if method not in METHODS:
            raise ValueError(method)
        nsteps, h = self.config["steps"], self.config["dt"]
        trajectories = [np.zeros((nsteps+1, len(l.mass), 3)) for l in self.levels]
        reports = []
        for k, l in enumerate(self.levels):
            trajectories[k][0] = l.initial
            if k == 0 and coarse is not None:
                trajectories[0][:] = coarse
                reports.append(coarse_reports)
                continue
            records = []
            for t in range(nsteps):
                xhat = predictor(method, k, t, trajectories, self.maps, h, self.velocities)
                target = self.maps[k-1] @ trajectories[k-1][t+1] if k and method != "independent" else None
                x, record = self.step(k, trajectories[k][t], xhat, (t+1)*h, target)
                trajectories[k][t+1] = x
                records.append(record)
            reports.append(records)
            failures = sum(not r["success"] for r in records)
            print(f"{method}: level {k}, {len(l.mass)} vertices, {failures}/{nsteps} solver warnings", flush=True)
        return trajectories, reports


def run(config_path, output, methods=METHODS):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    system = ClothSystem(config)
    data, logs = {}, {}
    coarse, coarse_reports = None, None
    for method in methods:
        trajectories, reports = system.simulate(method, coarse, coarse_reports)
        if coarse is None:
            coarse, coarse_reports = trajectories[0], reports[0]
        logs[method] = reports
        for k, trajectory in enumerate(trajectories):
            data[f"{method}_{k}"] = trajectory
    for k, l in enumerate(system.levels):
        data[f"faces_{k}"] = l.mesh.faces
        data[f"mass_{k}"] = l.mass
    np.savez_compressed(output / "trajectories.npz", **data)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "solver.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    environment = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "platform": platform.system(), "dtype": "float64", "methods": list(methods),
                   "optimizer": "L-BFGS-B", "coarse_trajectory_shared": True}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return system, data, logs
