"""Check this computer's CUDA IPC backend and velocity-predictor bridge.

This is an integration probe, not a folding or performance experiment.
"""
import json
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import uipc
from uipc import view
from uipc.constitution import NeoHookeanShell, ElasticModuli2D, DiscreteShellBending
from uipc.core import Engine, Scene, World, FiniteElementStateAccessorFeature
from uipc.geometry import trimesh, label_surface


def main():
    out = Path("output/native_ipc_probe").resolve()
    out.mkdir(parents=True, exist_ok=True)
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    cfg = Scene.default_config()
    cfg["dt"] = 0.01
    cfg["gravity"] = [[0.0], [0.0], [-9.81]]
    cfg["extras"]["strict_mode"]["enable"] = 1
    cfg["newton"]["velocity_tol"] = 1e-5
    cfg["newton"]["semi_implicit"]["enable"] = 0
    engine = Engine("cuda", str(out))
    world = World(engine)
    scene = Scene(cfg)
    x = np.array([[0, 0, 1], [.1, 0, 1], [.1, .1, 1], [0, .1, 1]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = trimesh(x, f)
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(20000, .3), 800, .0001)
    DiscreteShellBending().apply_to(mesh, 20000, .3)
    obj = scene.objects().create("probe")
    obj.geometries().create(mesh)
    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("Native CUDA world initialization failed")
    feature = world.features().find(FiniteElementStateAccessorFeature)
    state = feature.create_geometry()
    state.vertices().create(uipc.builtin.position, np.zeros(3))
    state.vertices().create(uipc.builtin.velocity, np.zeros(3))
    feature.copy_to(state)
    initial = view(state.positions()).reshape(-1, 3).copy()
    # BDF1 consumes this as x_tilde = x + h*v + h*h*g.
    v = np.array([.2, -.1, .3])
    view(state.vertices().find(uipc.builtin.velocity))[:] = v.reshape(1, 3, 1)
    feature.copy_from(state)
    start = time.perf_counter()
    world.advance()
    world.retrieve()
    if not world.is_valid():
        raise RuntimeError("Native CUDA advance failed")
    feature.copy_to(state)
    result = view(state.positions()).reshape(-1, 3).copy()
    expected = initial + .01*v + .01**2*np.array([0, 0, -9.81])
    error = float(np.max(np.abs(result-expected)))
    record = {"package": importlib.metadata.version("pyuipc"), "package_path": uipc.__file__,
              "cuda_world_valid": world.is_valid(), "predictor_error_m": error,
              "seconds_one_step_including_retrieve": time.perf_counter()-start,
              "passed": error < 1e-7,
              "scope": "Rigid translation/free fall only; no contact, no folding benchmark."}
    (out / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not record["passed"]:
        raise RuntimeError("Velocity injection does not match the expected BDF1 predictor")


if __name__ == "__main__":
    main()
