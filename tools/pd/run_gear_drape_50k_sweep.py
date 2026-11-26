"""Run paired native and progressive cloth simulations on a shared gear."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.assets.generate_pd_square_cloth import generate
from tools.assets.prepare_pd_cloth_proxy import prepare
from tools.pd.run_integrator_benchmark import read_mesh, read_p
from tools.pd.run_velpro_drape import BACKEND, gear_mesh, run_case


# Paired groups share intermediate meshes. B also provides a nested-grid control.
GROUPS = {"A": (21, 56, 224), "B": (28, 56, 224),
          "C": (49, 112, 224), "D": (70, 112, 224)}
CHAIN = ("proxy_coarse", "velpro_middle", "velpro_fine")
CENTER = (0.032, -0.018, 0.40)


def setup(directory, cells):
    directory.mkdir(parents=True, exist_ok=True)
    meshes = directory / "hierarchy"
    if not (meshes / "manifest.json").is_file():
        generate(meshes, cells, side_length=1., center=CENTER)
    manifest = json.loads((meshes / "manifest.json").read_text())
    if [row["cells_per_side"] for row in manifest["levels"]] != list(cells):
        raise ValueError(f"Unexpected saved resolutions in {meshes}")
    rests = [read_mesh(meshes, level)[0].ravel() for level in range(3)]
    for kind, source, target in (("direct", 0, 2), ("adjacent", 0, 1), ("adjacent", 1, 2)):
        p = read_p(meshes, kind, source, target).tocsr()
        p.check_format(full_check=True)
        if (np.any(p.data < 0) or not np.all(np.asarray(p.sum(axis=1)) == 1)
                or np.max(np.abs(p @ rests[source] - rests[target])) >= 1e-12):
            raise ValueError(f"Invalid {kind} operator from L{source} to L{target} in {meshes}")
    for level in range(2):
        asset = meshes / f"proxy_L{level}.json"
        if not asset.is_file():
            prepare(meshes, level, asset)
    vertices, faces = gear_mesh()
    np.savez(directory / "gear.npz", vertices=vertices, faces=faces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=BACKEND)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    args.backend = args.backend.resolve()
    args.output = args.output.resolve()
    args.solver_info = True
    gate = json.loads(args.gate.read_text())
    with (args.backend / "libuipc_backend_cuda.so").open("rb") as library:
        digest = hashlib.file_digest(library, "sha256").hexdigest()
    if not gate["passed"] or digest != gate["backend_sha256"]:
        raise ValueError("The CUDA backend does not match the verified proxy audit")
    for name, cells in GROUPS.items():
        setup(args.output / name, cells)
    (args.output / "setup.json").write_text(json.dumps(
        dict(groups=GROUPS, steps=args.steps, max_iter=args.max_iter,
             gate=str(args.gate.resolve()), backend_sha256=digest), indent=2) + "\n")
    expected_material = gate["runtime"]["effective_material"]
    for name, cells in GROUPS.items():
        directory = args.output / name
        args.cells = cells
        print(f"Group {name}: {cells}", flush=True)
        run_case(directory, "direct_coarse", 0, "bdf1", args)
        previous = run_case(directory, CHAIN[0], 0, "bdf1", args, proxy=True)
        previous = run_case(directory, CHAIN[1], 1, "velpro", args, previous, proxy=True)
        run_case(directory, CHAIN[2], 2, "velpro", args, previous)
        if name == "A":
            run_case(directory, "direct_fine", 2, "bdf1", args)
        for settings in directory.glob("*/settings.json"):
            if json.loads(settings.read_text())["effective_material"] != expected_material:
                raise ValueError(f"Cloth material changed in {settings}")


if __name__ == "__main__":
    main()
