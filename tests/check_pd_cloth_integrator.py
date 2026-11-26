"""Check the CUDA predictors and the horizontal fine-level state update."""

import argparse
from pathlib import Path
import tempfile

import numpy as np
import uipc
from uipc.constitution import Empty
from uipc.core import Engine, FiniteElementStateAccessorFeature, Scene, World


def check(mode):
    dt = 1 / 120
    config = Scene.default_config()
    config["dt"] = dt
    config["contact"]["enable"] = 0
    config["sanity_check"]["enable"] = 0
    config["newton"]["velocity_tol"] = 1e-8
    config["newton"]["transrate_tol"] = 1e-8

    x = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    v = np.array([[.1, -.2, .3], [-.4, .5, .6], [0., 0., 0.]])
    delta_v = np.array([v / 2, -v / 4])
    # These are already prolonged from coarse time t. Both differ from the
    # fine state, and the two samples differ from each other.
    coarse_x = np.stack([x + [.02, -.03, .01], x + [-.01, .04, -.02]])
    coarse_v = np.stack([v * 2, -v * 3])
    paper = mode == "paper_eq5"
    acceleration = np.array([0., 0., -9.81])
    config["gravity"] = acceleration.reshape(3, 1).tolist()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "predictor.f64"
        samples = np.stack((coarse_x, coarse_v), axis=1) if paper else (
            coarse_v if mode == "velpro" else delta_v
        )
        samples.astype("<f8").tofile(path)
        scene = Scene(config)
        engine = Engine("cuda", directory)
        world = World(engine)
        empty = Empty()
        scene.constitution_tabular().insert(empty)
        mesh = uipc.geometry.trimesh(x, np.array([[0, 1, 2]], dtype=np.int32))
        uipc.geometry.label_surface(mesh)
        empty.apply_to(mesh, mass_density=100., thickness=.001)
        key = {"paper_eq5": "pd_cloth_paper_eq5_state", "velpro": "pd_cloth_velpro_velocity",
               "vel_inc": "pd_cloth_delta_velocity"}[mode]
        mesh.meta().create(key, str(path))
        uipc.view(mesh.vertices().find(uipc.builtin.velocity)).reshape(-1, 3)[:] = v
        uipc.view(mesh.vertices().find(uipc.builtin.is_fixed)).reshape(-1)[2] = 1
        scene.objects().create("cloth").geometries().create(mesh)
        world.init(scene)
        feature = world.features().find(FiniteElementStateAccessorFeature)
        state = feature.create_geometry()
        pos = state.vertices().create(uipc.builtin.position, np.zeros(3))
        vel = state.vertices().create(uipc.builtin.velocity, np.zeros(3))
        for step, dv in enumerate(delta_v, 1):
            previous_fine_x = x.copy()
            if paper:
                predicted = coarse_x[step - 1] + dt * coarse_v[step - 1] + dt**2 * acceleration
            elif mode == "velpro":
                predicted = x + dt * coarse_v[step - 1] + dt**2 * acceleration
            else:
                predicted = x + dt * (v + dv)
            x[:2] = predicted[:2]
            v = (x - previous_fine_x) / dt
            world.advance()
            world.sync()
            assert world.is_valid() and world.frame() == step
            feature.copy_to(state)
            x_gpu = pos.view().reshape(-1, 3)
            v_gpu = vel.view().reshape(-1, 3)
            print(f"{mode} step {step}: dx max {np.max(np.abs(x_gpu - x)):.3e} m, "
                  f"dv max {np.max(np.abs(v_gpu - v)):.3e} m/s")
            np.testing.assert_allclose(x_gpu, x, rtol=0, atol=1e-12)
            np.testing.assert_allclose(v_gpu, v, rtol=0, atol=1e-10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-dir", required=True, type=Path)
    backend = parser.parse_args().backend_dir
    global_config = uipc.default_config()
    global_config["module_dir"] = str(backend.resolve())
    uipc.init(global_config)
    for mode in ("vel_inc", "paper_eq5", "velpro"):
        check(mode)
