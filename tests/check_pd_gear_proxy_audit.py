"""Independent CPU differentiation, spectral projection and GPU routing checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy import sparse
import torch
import uipc

from check_pd_cloth_proxy import (
    DT, KAPPA, POISSON, SHEAR, STRAIN_RATE, STRETCH, THICKNESS, bending, cpu_elastic,
    element_data, gpu_first_newton, infinity_error,
)
from tools.assets.generate_pd_square_cloth import generate
from tools.assets.prepare_pd_cloth_proxy import prepare
from tools.pd.run_integrator_benchmark import read_mesh
from tools.pd.run_square_fold import runtime_environment
from tools.pd.run_velpro_drape import gear_mesh


def positive_part(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 0)[..., None, :]) @ np.swapaxes(vectors, -2, -1)


def cpu_projected(elements, x):
    """Diagonalize each fine strain/stretch contribution and each fine hinge."""
    out = sparse.csr_matrix((x.size, x.size))
    for kernel, ids, extras in elements:
        local = torch.from_numpy(x[ids])
        if kernel is bending:
            derivatives = torch.vmap(torch.func.hessian(kernel))(local, *extras).numpy()
            blocks = positive_part(derivatives.reshape(len(ids), 12, 12))
        else:
            inverse, area = (a.numpy() for a in extras)
            f = ((x[ids][:, 1:] - x[ids][:, :1]).transpose(0, 2, 1) @ inverse)
            columns = torch.from_numpy(f.transpose(0, 2, 1).reshape(-1, 6))
            stretch = STRETCH * (2 * THICKNESS) / (1 - POISSON**2)
            shear = SHEAR / (2 * (1 + POISSON))

            def stretching(flat):
                delta = torch.linalg.vector_norm(flat.reshape(2, 3), dim=1) - 1
                return stretch * torch.sum(delta**2 + STRAIN_RATE * delta.clamp_min(0)**3)

            def shearing(flat):
                return shear * (flat[:3] @ flat[3:])**2

            hs, ht = (positive_part(torch.vmap(torch.func.hessian(term))(columns).numpy())
                      for term in (stretching, shearing))
            edges = np.array([[-1, -1], [1, 0], [0, 1]])
            jacobian = np.stack([np.kron((edges @ q).T, np.eye(3)) for q in inverse])
            blocks = (area * DT**2)[:, None, None] * (jacobian.transpose(0, 2, 1) @ (hs + ht) @ jacobian)
        dofs = (3 * ids[:, :, None] + np.arange(3)).reshape(len(ids), -1)
        rows = np.broadcast_to(dofs[:, :, None], blocks.shape)
        cols = np.broadcast_to(dofs[:, None, :], blocks.shape)
        out += sparse.coo_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())), shape=out.shape)
    return out


def numerical_case(case, cells):
    case.mkdir(exist_ok=True)
    hierarchy = case / "hierarchy"
    manifest = generate(hierarchy, cells, side_length=1., center=(0., 0., .4))
    cr, cf = read_mesh(hierarchy, 0)
    fr, ff = read_mesh(hierarchy, 1)
    asset = (case / "proxy.json").resolve()
    prepare(hierarchy, 0, asset)
    p = sparse.load_npz(hierarchy / "P_direct_L0_to_L1.npz")
    assert p.indices.min() == 0 and p.indices.max() < cr.size and p.data.min() >= 0
    assert np.all(np.asarray(p.sum(axis=1)) == 1)
    rest_error = float(np.max(np.abs(p @ cr.ravel() - fr.ravel())))
    assert rest_error < 1e-12
    xy = cr[:, :2]
    current = cr.copy()
    current[:, 0] += 0.013 * cr[:, 0] + 0.06 * cr[:, 1]
    current[:, 1] -= 0.026 * cr[:, 1]
    current[:, 2] += 0.13 * np.exp(-24 * (xy[:, 0] - .19)**2) * np.sin(5 * xy[:, 1] + .2)
    current[np.argmin(np.linalg.norm(xy - [-.1, .1], axis=1)), 2] += .18
    refined = (p @ current.ravel()).reshape(fr.shape)
    cpu_fine = element_data(fr, ff)
    cpu_coarse = element_data(cr, cf)
    expected_e, expected_g, raw = cpu_elastic(cpu_fine, refined, 2)
    raw = p.T @ raw @ p
    restricted_g = p.T @ expected_g

    rng = np.random.default_rng(245)
    error_g = error_h = 0.
    eps = 1e-6
    for direction in rng.normal(size=(12, cr.size)):
        direction /= np.linalg.norm(direction)
        fine_dir = (p @ direction).reshape(fr.shape)
        energy_a, grad_a, _ = cpu_elastic(cpu_fine, refined + eps * fine_dir, 1)
        energy_b, grad_b, _ = cpu_elastic(cpu_fine, refined - eps * fine_dir, 1)
        error_g = max(error_g, abs((energy_a - energy_b) / (2 * eps) - restricted_g @ direction))
        error_h = max(error_h, infinity_error(p.T @ (grad_a - grad_b) / (2 * eps), raw @ direction))
    proj = p.T @ cpu_projected(cpu_fine, refined) @ p

    # The runtime reference has the same inertia but zero elasticity.
    proxy_h, proxy_g, gpu_raw, gpu_e = gpu_first_newton(case / "proxy", cr, cf, current, "proxy", asset)
    inertial_h, inertial_g = gpu_first_newton(case / "inertia", cr, cf, current, "empty")
    native_h, native_g = gpu_first_newton(case / "native", cr, cf, current, "native")
    second_projection = positive_part(raw.toarray())
    symmetry = infinity_error(proxy_h, proxy_h.T)
    min_elastic = float(np.linalg.eigvalsh((proxy_h - inertial_h).toarray()).min())
    min_total = float(np.linalg.eigvalsh(proxy_h.toarray()).min())
    direction_proxy = np.linalg.solve(proxy_h.toarray(), -proxy_g)
    direction_native = np.linalg.solve(native_h.toarray(), -native_g)
    energy_split = {
        label: {part: cpu_elastic((el,), x, 0)[0] for part, el in zip(("membrane", "bending"), elements)}
        for label, elements, x in (("proxy", cpu_fine, refined), ("native", cpu_coarse, current))
    }
    result = dict(cells=cells, rest_error_m=rest_error, row_unit_error=0., fd_gradient_error=error_g,
                  fd_raw_hessian_action_error=error_h, gradient_error=infinity_error(proxy_g, restricted_g),
                  line_search_energy_error=abs(gpu_e - expected_e), raw_hessian_error=infinity_error(gpu_raw, raw),
                  projected_hessian_error=infinity_error(proxy_h - inertial_h, proj),
                  after_restriction_projection_difference=infinity_error(proj, second_projection),
                  symmetry_error=symmetry, elastic_min_eigenvalue=min_elastic, total_min_eigenvalue=min_total,
                  inertia_gradient_max=float(np.max(np.abs(inertial_g))), energy_split=energy_split,
                  proxy_gradient_l2=float(np.linalg.norm(proxy_g)), native_gradient_l2=float(np.linalg.norm(native_g)),
                  newton_direction_distance_l2_m=float(np.linalg.norm(direction_proxy - direction_native)),
                  proxy_descent=float(proxy_g @ direction_proxy), native_descent=float(native_g @ direction_native))
    print(json.dumps(result), flush=True)
    assert error_g < 1e-9 and error_h < 1e-8
    assert result["line_search_energy_error"] < 1e-12 and result["gradient_error"] < 1e-10
    assert result["raw_hessian_error"] < 1e-8 and result["projected_hessian_error"] < 1e-10
    assert symmetry < 1e-12 and min_elastic > -1e-12 and min_total > 0
    assert result["inertia_gradient_max"] < 1e-15
    assert result["after_restriction_projection_difference"] > 1e-8
    assert result["newton_direction_distance_l2_m"] > 1e-8
    assert result["proxy_descent"] < 0 and result["native_descent"] < 0
    assert abs(energy_split["proxy"]["bending"] - energy_split["native"]["bending"]) > 1e-12
    membrane_error = abs(energy_split["proxy"]["membrane"] - energy_split["native"]["membrane"])
    assert membrane_error < 1e-12 if cells[1] % cells[0] == 0 else membrane_error > 1e-9
    return result


def runtime_case(root, backend):
    directory = root / "runtime_gear"
    directory.mkdir()
    hierarchy = directory / "hierarchy"
    generate(hierarchy, (5, 16), side_length=1., center=(0.032, -0.018, 0.40))
    prepare(hierarchy, 0, directory / "proxy.json")
    vertices, faces = gear_mesh()
    collider = directory / "gear.npz"
    np.savez(collider, vertices=vertices, faces=faces)
    with (directory / "solver.log").open("w") as stream:
        subprocess.run([sys.executable, "-B", "-m", "tools.pd.run_integrator_benchmark",
                        "--worker", "--solver-info", "--case", str(directory), "--meshes", str(hierarchy),
                        "--backend", str(backend), "--steps", "24", "--level", "0", "--collider",
                        str(collider), "--proxy", str(directory / "proxy.json"), "--audit-step", "22",
                        "--dt", ".01"], stdout=stream, stderr=subprocess.STDOUT,
                       env=runtime_environment(), check=True)
    log = (directory / "solver.log").read_text()
    required = ["proxies=1 native_energy_slots=0 native_gradient_blocks=0 native_hessian_blocks=0",
                "PD proxy material:", "PD proxy compute_energy:", "PD proxy assemble: frame=22 Newton=0"]
    for marker in required:
        assert marker in log, f"Missing runtime evidence: {marker}"
    contact = [(int(g), int(h)) for g, h in re.findall(
        r"<[^>]*IPCSimplexFrictionalContact[^>]*> DyTopo Grad3 count: (\d+), DyTopo Hess3x3 count: (\d+)", log)]
    assert any(g > 0 and h > 0 for g, h in contact), "No active frictional IPC assembly in the proxy scene"
    assert (directory / "audit/x_newton.f64").stat().st_size == 36 * 3 * 8
    settings = json.loads((directory / "settings.json").read_text())
    return {"log": str(directory / "solver.log"), "markers": required,
            "maximum_ipc_gradient_blocks": max(g for g, _ in contact),
            "maximum_ipc_hessian_blocks": max(h for _, h in contact),
            "effective_material": settings["effective_material"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, backend = args.output.resolve(), args.backend.resolve()
    output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    uipc.init({"module_dir": str(backend)})
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    cases = [numerical_case(output / name, cells)
             for name, cells in (("nested", (5, 10)), ("nonnested", (5, 8)))]
    evidence = runtime_case(output, backend)
    with (backend / "libuipc_backend_cuda.so").open("rb") as library:
        digest = hashlib.file_digest(library, "sha256").hexdigest()
    (output / "gate.json").write_text(json.dumps(
        {"passed": True, "backend_sha256": digest, "numerics": cases, "runtime": evidence}, indent=2) + "\n")


if __name__ == "__main__":
    main()
