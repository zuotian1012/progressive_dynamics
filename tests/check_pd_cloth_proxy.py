"""Compare the production proxy assembly with finite differences and native FEM."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import io, sparse
import torch
import uipc
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, Empty, StrainLimitingBaraffWitkinShell
from uipc.core import Engine, Scene, World

from tools.assets.prepare_pd_cloth_proxy import prepare


DT = 1 / 120
RHO = 800.0
THICKNESS = 0.0001
STRETCH = 9800.0
SHEAR = 4.0
POISSON = 0.49
KAPPA = 1e-7
STRAIN_RATE = 100.0
MEMBRANE_STIFFNESS = (STRETCH * (2 * THICKNESS) / (1 - POISSON**2),
                      SHEAR / (2 * (1 + POISSON)), STRAIN_RATE)


def membrane(x, inverse, area, *, stiffness=MEMBRANE_STIFFNESS, dt=DT):
    stretch_stiffness, shear_stiffness, strain_rate = stiffness
    f = (x[1:] - x[0]).T @ inverse
    stretch = torch.linalg.vector_norm(f, dim=0) - 1
    shear = torch.dot(f[:, 0], f[:, 1])
    return area * dt**2 * (
        stretch_stiffness * torch.sum(stretch**2 + strain_rate * stretch.clamp_min(0)**3)
        + shear_stiffness * shear**2
    )


def bending(x, metric, *, stiffness=KAPPA, dt=DT):
    a, b, c, d = x
    edge = c - b
    n1 = torch.linalg.cross(b - a, c - a)
    n2 = torch.linalg.cross(c - d, b - d)
    signed_sin = torch.dot(torch.linalg.cross(n1, n2), edge) / torch.linalg.vector_norm(edge)
    angle = torch.atan2(signed_sin, torch.dot(n1, n2))
    return (dt**2 * stiffness) * metric * angle**2


def element_data(rest, faces):
    edges = {}
    for a, b, c in faces.tolist():
        for left, right, other in ((a, b, c), (b, c, a), (c, a, b)):
            edges.setdefault(tuple(sorted((left, right))), []).append(other)
    hinges = np.array([(min(op), a, b, max(op)) for (a, b), op in edges.items() if len(op) == 2])
    a, b, c = rest[faces].transpose(1, 0, 2)
    inverse = np.linalg.inv(np.stack(((b - a)[:, :2], (c - a)[:, :2]), axis=2))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    a, b, c, d = rest[hinges].transpose(1, 0, 2)
    area_sum = (np.linalg.norm(np.cross(b - a, c - a), axis=1)
                + np.linalg.norm(np.cross(c - d, b - d), axis=1)) / 2
    metric = 3 * np.sum((c - b)**2, axis=1) / area_sum
    return ((membrane, faces, (torch.from_numpy(inverse), torch.from_numpy(area))),
            (bending, hinges, (torch.from_numpy(metric),)))


def cpu_elastic(elements, positions, derivatives):
    x = torch.from_numpy(positions)
    energy = 0.0
    gradient = np.zeros(x.numel())
    hessian = sparse.csr_matrix((x.numel(), x.numel()))
    for kernel, indices, extras in elements:
        element_x = x[indices]
        energy += torch.vmap(kernel)(element_x, *extras).sum().item()
        if derivatives >= 1:
            dofs = (3 * indices[:, :, None] + np.arange(3)).reshape(len(indices), -1)
            local_g = torch.vmap(torch.func.grad(kernel))(element_x, *extras).numpy()
            np.add.at(gradient, dofs.ravel(), local_g.ravel())
        if derivatives == 2:
            h = torch.vmap(torch.func.hessian(kernel))(element_x, *extras).numpy()
            width = dofs.shape[1]
            rows = np.broadcast_to(dofs[:, :, None], (len(indices), width, width))
            cols = np.broadcast_to(dofs[:, None, :], rows.shape)
            hessian += sparse.coo_matrix((h.ravel(), (rows.ravel(), cols.ravel())),
                                         shape=hessian.shape).tocsr()
    return energy, gradient, hessian


def make_mesh(rest, faces):
    mesh = uipc.geometry.trimesh(rest, faces.astype(np.int32))
    uipc.geometry.label_surface(mesh)
    return mesh


def gpu_first_newton(directory, rest, faces, vertices, kind, asset=None):
    directory.mkdir()
    config = Scene.default_config()
    config["dt"] = DT
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = 0
    config["sanity_check"]["enable"] = 0
    config["newton"]["max_iter"] = 1
    config["extras"]["debug"]["dump_linear_system"] = 1
    scene = Scene(config)
    engine = Engine("cuda", str(directory))
    world = World(engine)
    mesh = make_mesh(rest, faces)
    if kind in ("native", "proxy"):
        shell = StrainLimitingBaraffWitkinShell()
        shell.apply_to(mesh, stretch_moduli=ElasticModuli2D.youngs_poisson(STRETCH, POISSON),
                       shear_moduli=ElasticModuli2D.youngs_poisson(SHEAR, POISSON),
                       mass_density=RHO, thickness=THICKNESS, strain_rate=STRAIN_RATE)
        if kind == "native":
            scene.constitution_tabular().insert(shell)
            bend = DiscreteShellBending()
            scene.constitution_tabular().insert(bend)
            bend.apply_to(mesh, bending_stiffness=KAPPA)
    raw_path = directory / "raw.mtx"
    if kind in ("empty", "proxy"):
        empty = Empty()
        scene.constitution_tabular().insert(empty)
        empty.apply_to(mesh, mass_density=RHO, thickness=THICKNESS)
    if kind == "proxy":
        mesh.meta().create("pd_cloth_proxy_asset", str(asset))
        mesh.meta().create("pd_cloth_proxy_kappa", KAPPA)
        mesh.meta().create("pd_cloth_proxy_validation", str(raw_path))
    # Preserve material and rest-volume attributes in both geometry slots.
    current_slot, _ = scene.objects().create("cloth").geometries().create(mesh)
    uipc.view(current_slot.geometry().positions())[:] = vertices.reshape(
        uipc.view(current_slot.geometry().positions()).shape
    )
    world.init(scene)
    assert world.is_valid()
    world.advance()
    world.sync()
    assert world.is_valid() and world.frame() == 1

    matrix_path, = directory.rglob("A.1.0.mtx")
    rhs_path, = directory.rglob("b.1.0.mtx")
    matrix = read_symmetric(matrix_path)
    rhs = io.mmread(rhs_path).ravel()
    if kind == "proxy":
        return matrix, rhs, read_symmetric(raw_path), float(
            raw_path.with_suffix(".mtx.energy").read_text()
        )
    return matrix, rhs


def read_symmetric(path):
    upper = io.mmread(path, spmatrix=True).tocsr()
    # libuipc exports all entries of a diagonal 3x3 block.
    entries = upper.tocoo()
    mask = (entries.row // 3) != (entries.col // 3)
    return upper + sparse.coo_matrix(
        (entries.data[mask], (entries.col[mask], entries.row[mask])), shape=upper.shape
    ).tocsr()


def infinity_error(actual, expected):
    error = actual - expected
    if sparse.issparse(error):
        return float(np.max(np.abs(error.data), initial=0))
    return float(np.max(np.abs(error)))


def check_level(hierarchy, level, work, samples):
    coarse = np.load(hierarchy / f"mesh_L{level}.npz")
    fine = np.load(hierarchy / "mesh_L3.npz")
    coarse_rest, fine_rest = coarse["rest"], fine["rest"]
    p = sparse.load_npz(hierarchy / f"P_direct_L{level}_to_L3.npz")
    asset = work / f"proxy_L{level}.json"
    prepare(hierarchy, level, asset)
    elements = element_data(fine_rest, fine["triangles"])
    records = []
    for seed in range(samples):
        rng = np.random.default_rng(seed + 100 * level)
        x = coarse_rest.ravel() + rng.normal(scale=0.001, size=coarse_rest.size)
        refined = (p @ x).reshape(-1, 3)
        energy, gradient, hessian = cpu_elastic(elements, refined, 2)
        expected_g = p.T @ gradient
        expected_raw = p.T @ hessian @ p

        # Directional central differences of the coarse energy and gradient.
        error_g = 0.0
        error_h = 0.0
        epsilon = 1e-6
        for _ in range(12):
            direction = rng.normal(size=x.size)
            direction /= np.linalg.norm(direction)
            e_plus, g_plus, _ = cpu_elastic(elements, (p @ (x + epsilon * direction)).reshape(-1, 3), 1)
            e_minus, g_minus, _ = cpu_elastic(elements, (p @ (x - epsilon * direction)).reshape(-1, 3), 1)
            error_g = max(error_g, abs((e_plus - e_minus) / (2 * epsilon) - expected_g @ direction))
            error_h = max(error_h, infinity_error(p.T @ (g_plus - g_minus) / (2 * epsilon),
                                                   expected_raw @ direction))

        prefix = work / f"L{level}_seed{seed}"
        prefix.mkdir()
        proxy_h, proxy_g, raw, proxy_energy = gpu_first_newton(
            prefix / "proxy", coarse_rest, coarse["triangles"], x.reshape(-1, 3), "proxy", asset
        )
        coarse_mass, inertial_g = gpu_first_newton(
            prefix / "coarse_mass", coarse_rest, coarse["triangles"], x.reshape(-1, 3), "empty"
        )
        native_h, native_g = gpu_first_newton(
            prefix / "native_fine", fine_rest, fine["triangles"], refined, "native"
        )
        fine_mass, fine_inertial_g = gpu_first_newton(
            prefix / "fine_mass", fine_rest, fine["triangles"], refined, "empty"
        )
        np.testing.assert_allclose(inertial_g, 0, atol=1e-15)
        np.testing.assert_allclose(fine_inertial_g, 0, atol=1e-15)
        projected = p.T @ (native_h - fine_mass) @ p
        record = {
            "level": level, "seed": seed,
            "cpu_fd_gradient": error_g,
            "cpu_fd_raw_hessian_action": error_h,
            "gpu_cpu_energy": abs(proxy_energy - energy),
            "gpu_cpu_gradient": infinity_error(proxy_g, expected_g),
            "gpu_cpu_raw_hessian": infinity_error(raw, expected_raw),
            "gpu_cpu_projected_hessian": infinity_error(proxy_h - coarse_mass, projected),
            "gpu_restricted_native_gradient": infinity_error(proxy_g, p.T @ native_g),
            "projected_vs_raw": infinity_error(projected, expected_raw),
        }
        print(json.dumps(record), flush=True)
        assert error_g < 1e-10 and error_h < 1e-8
        assert record["gpu_cpu_energy"] < 1e-12
        assert record["gpu_cpu_gradient"] < 1e-10
        assert record["gpu_cpu_raw_hessian"] < 1e-8
        assert record["gpu_cpu_projected_hessian"] < 1e-10
        assert record["gpu_restricted_native_gradient"] < 1e-10
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--levels", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    torch.set_num_threads(2)
    config = uipc.default_config()
    config["module_dir"] = str(args.backend_dir.resolve())
    uipc.init(config)
    with tempfile.TemporaryDirectory() as directory:
        results = []
        for level in args.levels:
            results.extend(check_level(args.hierarchy, level, Path(directory), args.samples))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
