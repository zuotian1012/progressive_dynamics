"""Check the PSQ gradient against the full transfer and the SPD search metric."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import uipc

from check_pd_cloth_proxy import gpu_first_newton, infinity_error
from tools.assets.prepare_psq_hierarchy import shell_prolongation
from tools.pd.run_integrator_benchmark import read_mesh, read_p
from tools.pd.run_psq_shell import load_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    args = parser.parse_args()
    uipc.init({"module_dir": str(args.backend_dir.resolve())})
    uipc.Logger.set_level(uipc.Logger.Level.Warn)
    meshes = args.run / "hierarchy"
    coarse, coarse_faces = read_mesh(meshes, 0)
    fine, fine_faces = read_mesh(meshes, 2)
    p = read_p(meshes, "direct", 0, 2)
    state = np.fromfile(args.run / "preview/audit/x_newton.f64", dtype="<f8").reshape(coarse.shape)
    refined = shell_prolongation(state, *load_binding(meshes, "direct", 0, 2))
    with tempfile.TemporaryDirectory() as temp:
        case = Path(temp)
        ph, pg, _, _ = gpu_first_newton(case / "proxy", coarse, coarse_faces, state, "proxy",
                                       (meshes / "proxy_L0.json").resolve())
        cm, _ = gpu_first_newton(case / "coarse_mass", coarse, coarse_faces, state, "empty")
        fh, fg = gpu_first_newton(case / "fine", fine, fine_faces, refined, "native")
        fm, _ = gpu_first_newton(case / "fine_mass", fine, fine_faces, refined, "empty")
    binding = load_binding(meshes, "direct", 0, 2)
    rng = np.random.default_rng(527)
    errors = []
    for _ in range(12):
        direction = rng.normal(size=state.shape)
        direction /= np.linalg.norm(direction)
        h = 1e-7
        tangent = (shell_prolongation(state + h*direction, *binding)
                   - shell_prolongation(state - h*direction, *binding))/(2*h)
        errors.append(abs(pg @ direction.ravel() - fg @ tangent.ravel()))
    result = {"gradient": float(max(errors)),
              "projected_hessian": infinity_error(ph - cm, p.T @ (fh - fm) @ p)}
    print(json.dumps(result), flush=True)
    assert result["gradient"] < 1e-10
    assert result["projected_hessian"] < 1e-10
    (args.run / "derivative_check.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
