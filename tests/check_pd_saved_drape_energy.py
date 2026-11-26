"""Evaluate native and proxy energies on identical poses from the gear sweep."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from check_pd_cloth_proxy import bending, element_data, membrane
from tools.pd.run_gear_drape_50k_sweep import GROUPS
from tools.pd.run_integrator_benchmark import lift, read_mesh, read_p


def split_energy(elements, positions, material):
    params = (material["stretch"], material["shear"], material["strain_rate"])
    by_type = {
        membrane: partial(membrane, dt=1., stiffness=params),
        bending: partial(bending, dt=1., stiffness=material["kappa"]),
    }
    x = torch.from_numpy(positions)
    return [torch.vmap(by_type[kernel])(x[ids], *args).sum().item() for kernel, ids, args in elements]


def check(root):
    torch.set_num_threads(1)
    fine = element_data(*read_mesh(root / "A/hierarchy", 2))
    frames = (20, 30, 46, 80, 120)
    results = {}
    for name, cells in GROUPS.items():
        directory = root / name
        meshes = directory / "hierarchy"
        coarse = element_data(*read_mesh(meshes, 0))
        transfer = read_p(meshes, "direct", 0, 2)
        rows = []
        for case in ("direct_coarse", "proxy_coarse"):
            run = directory / case
            settings = json.loads((run / "settings.json").read_text())
            material = settings["effective_material"]
            with np.load(run / "state.npz", allow_pickle=False) as data:
                positions = data["x"][list(frames)]
            for frame, x, y in zip(frames, positions, lift(transfer, positions), strict=True):
                native_membrane, native_bending = split_energy(coarse, x, material)
                proxy_membrane, proxy_bending = split_energy(fine, y, material)
                rows.append(dict(case=case, frame=frame, native_membrane_J=native_membrane,
                                 proxy_membrane_J=proxy_membrane, native_bending_J=native_bending,
                                 proxy_bending_J=proxy_bending))
        max_membrane_difference = max(abs(row["native_membrane_J"] - row["proxy_membrane_J"]) for row in rows)
        if cells[-1] % cells[0] == 0 and max_membrane_difference > 1e-12:
            raise ValueError(f"Nested hierarchy {name} changed the membrane energy")
        results[name] = dict(max_membrane_difference_J=max_membrane_difference, poses=rows)
        print(f"{name}: maximum same-pose membrane energy change {max_membrane_difference:.6g} J", flush=True)
    filename = root / "same_pose_energy.json"
    pending = filename.with_suffix(".tmp")
    pending.write_text(json.dumps(results, indent=2) + "\n")
    pending.replace(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
