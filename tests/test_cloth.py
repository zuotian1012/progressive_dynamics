import unittest

import numpy as np

from cloth.energy import ShellEnergy, floor_barrier
from cloth.mesh import grid, lumped_mass, prolongation
from cloth.simulation import ClothSystem, predictor


class ClothTests(unittest.TestCase):
    def setUp(self):
        self.mesh = grid(3)
        self.shell = ShellEnergy(self.mesh, 3e4, 0.3, 0.01)
        self.rng = np.random.default_rng(42)

    def assert_gradient(self, fun, x, tolerance=2e-6):
        _, g = fun(x)
        for _ in range(5):
            d = self.rng.normal(size=x.shape)
            d /= np.linalg.norm(d)
            eps = 1e-6
            numeric = (fun(x+eps*d)[0] - fun(x-eps*d)[0]) / (2*eps)
            analytic = np.sum(g*d)
            self.assertLess(abs(numeric-analytic) / max(1, abs(numeric), abs(analytic)), tolerance)

    def test_membrane_gradient(self):
        x = self.mesh.rest + self.rng.normal(scale=0.025, size=self.mesh.rest.shape)
        self.assert_gradient(self.shell.membrane, x)

    def test_bending_gradient(self):
        x = self.mesh.rest + self.rng.normal(scale=0.045, size=self.mesh.rest.shape)
        self.assert_gradient(self.shell.bending, x)

    def test_barrier_gradient(self):
        x = self.mesh.rest.copy()
        x[:, 2] = self.rng.uniform(0.005, 0.022, len(x))
        mass = lumped_mass(self.mesh, 0.6)
        self.assert_gradient(lambda y: floor_barrier(y, mass, 0, 0.025, 3000), x)

    def test_rest_and_rigid_motion(self):
        x = self.mesh.rest
        q, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
        e, g = self.shell.value_grad(x @ q + [0.4, -0.2, 2])
        self.assertLess(abs(e), 1e-10)
        self.assertLess(np.max(np.abs(g)), 1e-10)

    def test_rotation_covariance_and_force_balance(self):
        x = self.mesh.rest + self.rng.normal(scale=0.03, size=self.mesh.rest.shape)
        q, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
        e, g = self.shell.value_grad(x)
        er, gr = self.shell.value_grad(x @ q + [1, 2, 3])
        np.testing.assert_allclose(er, e, rtol=1e-10)
        np.testing.assert_allclose(gr, g @ q, atol=1e-9)
        np.testing.assert_allclose(g.sum(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.cross(x, g).sum(axis=0), 0, atol=1e-10)

    def test_hierarchy_and_proxy_gradient(self):
        fine = grid(7)
        p = prolongation(self.mesh, fine)
        np.testing.assert_allclose(p @ self.mesh.rest, fine.rest, atol=1e-15)
        np.testing.assert_allclose(np.asarray(p.sum(axis=1)), 1)
        self.assertEqual(np.linalg.matrix_rank(p.toarray()), len(self.mesh.uv))
        fine_shell = ShellEnergy(fine, 3e4, 0.3, 0.01)
        def proxy(x):
            e, g = fine_shell.value_grad(p @ x)
            return e, p.T @ g
        self.assert_gradient(proxy, self.mesh.rest + self.rng.normal(scale=0.02, size=self.mesh.rest.shape))
        self.assertAlmostEqual(lumped_mass(fine, 0.6).sum(), 0.6)

    def test_all_stencils_preserve_affine_constant_velocity(self):
        meshes = [grid(n) for n in [2, 3, 5]]
        maps = [prolongation(a, b) for a, b in zip(meshes, meshes[1:])]
        v = [np.tile([0.3, -0.1, 0.2], (len(m.uv), 1)) for m in meshes]
        h = 0.02
        trajectories = [np.stack([m.rest + k*h*vel for k in range(4)]) for m, vel in zip(meshes, v)]
        for method in ["independent", "semi", "full", "velpro"]:
            for k in range(3):
                for t in [0, 1, 2]:
                    np.testing.assert_allclose(predictor(method, k, t, trajectories, maps, h, v),
                                               trajectories[k][t+1], atol=1e-14)

    def test_single_step_free_fall(self):
        config = {"segments": [2], "dt": 0.02, "steps": 2, "initial_height": 1,
                  "pins": "none", "gravity": [0, 0, -9.81],
                  "material": {"young": 30000, "poisson": 0.3, "thickness": 0.003, "density": 200},
                  "solver": {"maxiter": 200, "gtol": 1e-8, "ftol": 1e-14}}
        system = ClothSystem(config)
        l = system.levels[0]
        x, report = system.step(0, l.initial, l.initial, 0.02)
        expected = l.initial + [0, 0, -9.81 * 0.02**2]
        np.testing.assert_allclose(x, expected, atol=1e-8)
        self.assertTrue(report["success"])

    def test_pins_and_plane_contact(self):
        config = {"segments": [2], "dt": 0.02, "steps": 20, "initial_height": 0.06,
                  "pins": "corners", "gravity": [0, 0, -9.81],
                  "material": {"young": 30000, "poisson": 0.3, "thickness": 0.003, "density": 200},
                  "floor": {"height": 0, "distance": 0.025, "stiffness": 3000},
                  "solver": {"maxiter": 300, "gtol": 2e-6, "ftol": 1e-13}}
        system = ClothSystem(config)
        tracks, reports = system.simulate("velpro")
        l = system.levels[0]
        self.assertGreater(tracks[0][:, :, 2].min(), 0)
        self.assertLess(tracks[0][-1, ~l.pins, 2].min(), config["floor"]["distance"])
        np.testing.assert_allclose(tracks[0][:, l.pins], np.broadcast_to(l.initial[l.pins], tracks[0][:, l.pins].shape))
        self.assertTrue(all(r["success"] for r in reports[0]))

    def test_vertical_flag_rest_and_spatial_forcing_gradient(self):
        config = {"segments": [2, 4], "dt": 0.04, "steps": 2, "initial_height": 0.2,
                  "orientation": "vertical", "pins": "left_edge", "gravity": [0, 0, -9.81],
                  "wind": {"acceleration": [0, 4, 1], "mean_acceleration": [0, 5, 0],
                           "frequency": 1.3, "spatial_phase": 2.0},
                  "material": {"young": 30000, "poisson": 0.3, "thickness": 0.0004, "density": 200},
                  "solver": {"maxiter": 300, "gtol": 2e-6, "ftol": 1e-13}}
        system = ClothSystem(config)
        for k, l in enumerate(system.levels):
            self.assertEqual(l.pins.sum(), l.mesh.segments+1)
            e, _ = system.shell.value_grad(l.to_fine @ l.initial)
            self.assertLess(abs(e), 1e-12)
            self.assert_gradient(lambda x: system.potential(k, x, 0.12), l.initial+self.rng.normal(scale=0.01, size=l.initial.shape))


if __name__ == "__main__":
    unittest.main()
