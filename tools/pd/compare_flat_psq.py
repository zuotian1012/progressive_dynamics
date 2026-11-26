"""Compare barycentric and zero-offset PSQ paths on the identical six-level gear scene."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.assets.generate_pd_square_cloth import barycentric_stencil, generate
from tools.assets.prepare_pd_cloth_proxy import prepare
from tools.assets.prepare_psq_hierarchy import write_binding
from tools.pd.run_integrator_benchmark import read_mesh, read_p
from tools.pd.run_velpro_drape import BACKEND, CELLS, gear_mesh, run_case


def make_assets(directory, use_psq):
    meshes = directory / "hierarchy"
    directory.mkdir(parents=True, exist_ok=True)
    gear = directory / "gear.npz"
    manifest_path = meshes / "manifest.json"
    if gear.is_file() and manifest_path.is_file():
        return
    manifest = generate(meshes, CELLS, side_length=1., center=(0.032, -0.018, 0.40))
    last = len(CELLS) - 1
    if use_psq:
        for entry in manifest["operators"]:
            kind, source, target = (entry[key] for key in ("kind", "source_level", "target_level"))
            if source == target:
                continue
            coarse, coarse_faces = read_mesh(meshes, source)
            fine, fine_faces = read_mesh(meshes, target)
            columns, _ = barycentric_stencil(CELLS[source], CELLS[target])
            # Use the existing triangle orientation for the PSQ frame.
            face_lookup = {tuple(sorted(row)): i for i, row in enumerate(coarse_faces)}
            support = np.array([face_lookup[tuple(sorted(row))] for row in columns])
            u = read_p(meshes, kind, source, target)[::3, ::3].tocsr()
            write_binding(meshes, kind, source, target, coarse, fine, coarse_faces, u, support,
                          proxy=fine_faces if kind == "direct" else None, zero_offset=True)
    else:
        for level in range(last):
            prepare(meshes, level, meshes / f"proxy_L{level}.json")
    vertices, faces = gear_mesh()
    np.savez(gear, vertices=vertices, faces=faces)


def compare(root, matrices, steps):
    result = {"levels": []}
    names = ["preview_L0"] + [f"velpro_L{i}" for i in range(1, len(CELLS))]
    for level, name in enumerate(names):
        left, right = (root / mode / name for mode in ("barycentric", "psq"))
        stats = {"cells": CELLS[level]}
        with np.load(left / "state.npz", allow_pickle=False) as bary, np.load(
            right / "state.npz", allow_pickle=False
        ) as psq:
            if not (bary["dt"] == psq["dt"] and len(bary["x"]) == len(psq["x"]) == steps + 1):
                raise ValueError(f"Different time grids at {name}")
            for field, key in (("x", "position_max_abs_m"), ("v", "velocity_max_abs_m_s")):
                a, b = bary[field], psq[field]
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"Invalid {name} states")
                stats[key] = float(np.max(np.abs(b - a)))
                stats[f"{field}_equal"] = bool(np.array_equal(a, b))
                if field == "x":
                    stats["position_rms_m"] = float(np.sqrt(np.mean(np.sum((b - a)**2, axis=-1))))
            stats["steps"] = steps
        for mode, case in (("barycentric", left), ("psq", right)):
            stats[f"{mode}_newton_limit_steps"] = (case / "solver.log").read_text().count(
                "Newton Iteration Exits with Max Iteration:"
            )
        if level:
            a = np.memmap(left / "predictor.f64", dtype="<f8", mode="r")
            b = np.memmap(right / "predictor.f64", dtype="<f8", mode="r")
            stats["predictor_equal"] = bool(np.array_equal(a, b))
        result["levels"].append(stats)
    result["intrinsic_operators_checked"] = matrices
    repeat = root / "barycentric/repeat_finest"
    if (repeat / "state.npz").is_file():
        original = root / "barycentric" / f"velpro_L{len(CELLS)-1}"
        predictor_equal = np.array_equal(
            np.memmap(original / "predictor.f64", dtype="<f8", mode="r"),
            np.memmap(repeat / "predictor.f64", dtype="<f8", mode="r"),
        )
        with np.load(original / "state.npz", allow_pickle=False) as a, np.load(
            repeat / "state.npz", allow_pickle=False
        ) as b:
            difference = a["x"] - b["x"]
        result["same_input_repeat_finest"] = {
            "predictor_equal": bool(predictor_equal),
            "position_max_abs_m": float(np.abs(difference).max()),
            "position_rms_m": float(np.sqrt(np.mean(np.sum(difference**2, axis=-1)))),
        }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def video(root):
    from tools.pd.report_velpro_drape import render_comparison

    last = len(CELLS) - 1
    assets = root / "barycentric"
    _, faces = read_mesh(assets / "hierarchy", last)
    with np.load(assets / "gear.npz", allow_pickle=False) as data:
        gear = data["vertices"][data["faces"]]
    states = []
    for mode in ("barycentric", "psq"):
        with np.load(root / mode / f"velpro_L{last}/state.npz", allow_pickle=False) as data:
            states.append(data["x"])
    render_comparison(gear, states, (faces, faces), ("Barycentric | VelPro", "PSQ | VelPro"),
                      ("#087e7c", "#087e7c"), root / "velpro_flat_comparison.mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=BACKEND)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--repeat-finest", action="store_true",
                        help="Repeat the barycentric finest solve with the same saved velocity input")
    args = parser.parse_args()
    if args.steps < 2 or args.max_iter < 1:
        parser.error("Require at least two steps and a positive Newton limit")
    output = args.output.resolve()
    bary, psq = (output / name for name in ("barycentric", "psq"))
    make_assets(bary, use_psq=False)
    make_assets(psq, use_psq=True)
    entries = json.loads((bary / "hierarchy/manifest.json").read_text())["operators"]
    for entry in entries:
        keys = (entry["kind"], entry["source_level"], entry["target_level"])
        a, b = (read_p(case / "hierarchy", *keys) for case in (bary, psq))
        if (a != b).nnz:
            raise ValueError(f"Different intrinsic maps: {keys}")
        kind, source, target = keys
        if source == target:
            continue
        with np.load(psq / "hierarchy" / f"PSQ_{kind}_L{source}_to_L{target}.npz",
                     allow_pickle=False) as binding:
            if np.any(binding["local"] != 0):
                raise ValueError(f"Nonzero PSQ offsets: {keys}")
        if kind == "direct":
            composed = read_p(psq / "hierarchy", "adjacent", source, source + 1)
            for level in range(source + 1, target):
                composed = read_p(psq / "hierarchy", "adjacent", level, level + 1) @ composed
            if (composed != b).nnz:
                raise ValueError(f"Composed intrinsic map differs: {keys}")
    for level in range(len(CELLS)):
        for case in (bary, psq):
            if level == 0:
                run_case(case, "preview_L0", 0, "bdf1", args, proxy=True)
            else:
                previous = "preview_L0" if level == 1 else f"velpro_L{level-1}"
                run_case(case, f"velpro_L{level}", level, "velpro", args,
                         previous=case / previous / "state.npz", proxy=level < len(CELLS) - 1)
    if args.repeat_finest:
        last = len(CELLS) - 1
        run_case(bary, "repeat_finest", last, "velpro", args,
                 previous=bary / f"velpro_L{last-1}/state.npz")
    compare(output, len(entries), args.steps)
    if args.video:
        video(output)


if __name__ == "__main__":
    main()
